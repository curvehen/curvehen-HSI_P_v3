"""Walk-forward backtest, self-learning ensemble, live next-day prediction.

Self-learning = (1) every fold and every live run retrains all models on the
latest data; (2) ensemble weights are re-computed from each model's trailing
out-of-sample log-loss, using only labels already known at prediction time.
"""
import argparse
import json
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from src import config
from src.features import builder
from src.models.common import prepare, walk_forward_splits
from src.models.deep import get_sequence_models
from src.models.tabular import LGBMModel, get_tabular_models

log = logging.getLogger(__name__)

EPS = 1e-6
WEIGHT_WINDOW = 126      # trailing OOS rows used to score each model
WEIGHT_TEMP = 20.0       # higher = weights concentrate on recent winners
MIN_HIST = 20
NEUTRAL_BAND = 0.02      # |p - 0.5| below this => NEUTRAL
METRICS_PATH = config.DATA_PROC / "oos_metrics.csv"


def _all_models(names: Optional[List[str]] = None) -> list:
    models = get_tabular_models() + get_sequence_models()
    return [m for m in models if not names or m.name in names]


def _predict(model, X: pd.DataFrame, start: int, end: int) -> np.ndarray:
    """P(up) for rows [start, end) using only rows <= each row."""
    if model.kind == "tabular":
        return model.predict_proba(X.iloc[start:end])
    c0 = max(0, start - config.SEQ_LEN + 1)
    pos = np.arange(start - c0, end - c0)
    return model.predict_proba(X.iloc[c0:end], pos)


# ------------------------------------------------------------------ backtest
def run_backtest(X: pd.DataFrame, y: np.ndarray, n_lab: int,
                 feats: pd.DataFrame, names: Optional[List[str]] = None) -> pd.DataFrame:
    preds: Dict[str, List[np.ndarray]] = {}
    idx_parts = []
    splits = list(walk_forward_splits(
        n_lab, config.WF_MIN_TRAIN, config.WF_STEP, config.WF_MAX_TRAIN))
    for k, (tr0, s, e) in enumerate(splits, 1):
        log.info("Fold %d/%d | train rows %d-%d | test %s -> %s",
                 k, len(splits), tr0, s, X.index[s].date(), X.index[e - 1].date())
        for m in _all_models(names):
            try:
                m.fit(X.iloc[tr0:s], y[tr0:s])
                p = _predict(m, X, s, e)
            except Exception as exc:
                log.warning("Model %s failed in fold %d: %s", m.name, k, exc)
                p = np.full(e - s, np.nan)
            preds.setdefault(m.name, []).append(p)
        idx_parts.append(X.index[s:e])

    idx = pd.DatetimeIndex(np.concatenate([i.values for i in idx_parts]))
    oos = pd.DataFrame({f"p_{k}": np.concatenate(v) for k, v in preds.items()}, index=idx)
    oos["target_dir"] = feats.loc[idx, "target_dir"].values
    oos["target_ret"] = feats.loc[idx, "target_ret"].values
    return add_ensembles(oos)


# ------------------------------------------------------------------ ensemble
def _pcols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("p_")]


def _loss_frame(oos: pd.DataFrame, pcols: List[str]) -> pd.DataFrame:
    p = oos[pcols].clip(EPS, 1 - EPS).values
    y = oos["target_dir"].values[:, None]
    return pd.DataFrame(-(y * np.log(p) + (1 - y) * np.log(1 - p)),
                        index=oos.index, columns=pcols)


def _weights(trailing: pd.DataFrame, avail: pd.DataFrame) -> pd.DataFrame:
    """Softmax over -trailing log-loss; equal weights where no history."""
    score = -WEIGHT_TEMP * trailing
    score = score.sub(score.max(axis=1), axis=0)
    w = np.exp(score).where(avail)
    no_hist = w.isna().all(axis=1)
    eq = avail.astype(float).where(avail)
    w.loc[no_hist] = eq.loc[no_hist]
    return w.div(w.sum(axis=1), axis=0)


COST_BPS = 5.0  # one-way cost per unit of position change (bps). Replace with your real costs.


def add_ensembles(oos: pd.DataFrame) -> pd.DataFrame:
    pc = _pcols(oos)
    P = oos[pc]
    oos["ens_equal"] = P.mean(axis=1)
    # label of row t-1 is known when predicting row t -> shift(1)
    trailing = (
        _loss_frame(oos, pc)
        .rolling(WEIGHT_WINDOW, min_periods=MIN_HIST).mean().shift(1)
    )
    w = _weights(trailing, P.notna())
    ens = (P.fillna(0) * w.fillna(0)).sum(axis=1)
    oos["ens_weighted"] = ens.where(P.notna().any(axis=1))
    return oos


# ------------------------------------------------------------------ metrics
def _stats(name: str, p: np.ndarray, y: np.ndarray, r: np.ndarray,
           pos: Optional[np.ndarray] = None) -> dict:
    if pos is None:
        pos = np.where(p > 0.5 + NEUTRAL_BAND, 1.0,
                       np.where(p < 0.5 - NEUTRAL_BAND, -1.0, 0.0))
    turn = np.abs(np.diff(pos, prepend=0.0))
    net = pos * r - turn * COST_BPS / 1e4
    sd = net.std(ddof=1)
    cum = np.cumsum(net)
    traded = pos != 0
    return {
        "model": name,
        "n": len(y),
        "accuracy": float(((p > 0.5) == (y == 1)).mean()),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(p)) > 1 and len(np.unique(y)) > 1 else np.nan,
        "logloss": float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1])),
        "hit_rate_traded": float((np.sign(r)[traded] == pos[traded]).mean()) if traded.any() else np.nan,
        "sharpe": float(net.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan,
        "max_dd_log": float((cum - np.maximum.accumulate(cum)).min()),
        "total_log_ret": float(cum[-1]),
        "exposure": float(traded.mean()),
    }


def evaluate(oos: pd.DataFrame) -> pd.DataFrame:
    """OOS metrics per model and ensemble, plus a majority-class / buy-and-hold baseline.
    Strategy: long if p > 0.5+band, short if p < 0.5-band, else flat; costs on position changes."""
    rows = []
    for c in _pcols(oos) + ["ens_equal", "ens_weighted"]:
        d = oos[[c, "target_dir", "target_ret"]].dropna()
        if len(d) < 30:
            continue
        rows.append(_stats(c.replace("p_", ""), d[c].values,
                           d["target_dir"].values, d["target_ret"].values))
    d = oos[["target_dir", "target_ret"]].dropna()
    if len(d) >= 30:
        rows.append(_stats("baseline_buyhold", np.full(len(d), d["target_dir"].mean()),
                           d["target_dir"].values, d["target_ret"].values,
                           pos=np.ones(len(d))))
    return pd.DataFrame(rows).set_index("model")


# ------------------------------------------------------------------ live
def _live_weights(oos: Optional[pd.DataFrame], names: List[str],
                  last_labeled: pd.Timestamp) -> pd.Series:
    """Weights from trailing OOS log-loss. All labels in oos are already known."""
    cols = [f"p_{n}" for n in names]
    equal = pd.Series(1.0 / len(names), index=names)
    if oos is None or oos.empty:
        return equal
    if oos.index[-1] < last_labeled:
        log.warning("OOS file ends %s, older than last label %s: re-run backtest.",
                    oos.index[-1].date(), last_labeled.date())
    tail = oos.dropna(subset=["target_dir"]).tail(WEIGHT_WINDOW)
    have = [c for c in cols if c in tail.columns]
    if not have:
        return equal
    loss = _loss_frame(tail, have)
    mean_loss = loss.mean().where(loss.notna().sum() >= MIN_HIST)
    trailing = pd.DataFrame([mean_loss]).reindex(columns=cols)
    present = pd.DataFrame([[True] * len(cols)], columns=cols)
    w = _weights(trailing, present).iloc[0]
    w.index = names
    return w


def _f(v) -> Optional[float]:
    v = float(v)
    return round(v, 6) if np.isfinite(v) else None


def _direction(p: float) -> str:
    if p >= 0.5 + NEUTRAL_BAND:
        return "UP"
    if p <= 0.5 - NEUTRAL_BAND:
        return "DOWN"
    return "NEUTRAL"


def run_live(X: pd.DataFrame, y: np.ndarray, n_lab: int, feats: pd.DataFrame,
             names: Optional[List[str]] = None,
             oos: Optional[pd.DataFrame]
