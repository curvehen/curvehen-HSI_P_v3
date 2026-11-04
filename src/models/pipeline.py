"""Walk-forward backtest, self-learning ensemble, live next-day prediction.

Self-learning = (1) every fold and every live run retrains all models on the
latest data (optionally re-tuning LightGBM / XGBoost on that window only);
(2) ensemble weights are re-computed from each model's trailing out-of-sample
log-loss, using only labels already known at prediction time.
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
from src.models import tuning
from src.models.common import prepare, walk_forward_splits
from src.models.deep import get_sequence_models
from src.models.tabular import LGBMModel, get_tabular_models

log = logging.getLogger(__name__)

EPS = 1e-6
WEIGHT_WINDOW = 126      # trailing OOS rows used to score each model
WEIGHT_TEMP = 20.0       # higher = weights concentrate on recent winners
MIN_HIST = 20
NEUTRAL_BAND = 0.02      # |p - 0.5| below this => NEUTRAL
METRICS_PATH = config.METRICS_PATH
COST_BPS = 5.0           # one-way cost per unit of position change (bps). PLACEHOLDER: use your real costs.


def _all_models(names: Optional[List[str]] = None, tuned: Optional[dict] = None) -> list:
    models = get_tabular_models(tuned) + get_sequence_models()
    return [m for m in models if not names or m.name in names]


def _tune_all(X: pd.DataFrame, y: np.ndarray, names: Optional[List[str]]):
    """Tune each tunable model on THIS window only. Returns (params, info)."""
    params, info = {}, {}
    for n in config.TUNE_MODELS:
        if names and n not in names:
            continue
        p, i = tuning.tune(n, X, y)
        if p:
            params[n] = p
        info[n] = i
    return params, info


def _predict(model, X: pd.DataFrame, start: int, end: int) -> np.ndarray:
    """P(up) for rows [start, end) using only rows <= each row."""
    if model.kind == "tabular":
        return model.predict_proba(X.iloc[start:end])
    c0 = max(0, start - config.SEQ_LEN + 1)
    pos = np.arange(start - c0, end - c0)
    return model.predict_proba(X.iloc[c0:end], pos)


# ------------------------------------------------------------------ backtest
def run_backtest(X: pd.DataFrame, y: np.ndarray, n_lab: int,
                 feats: pd.DataFrame, names: Optional[List[str]] = None,
                 tune: bool = False) -> pd.DataFrame:
    preds: Dict[str, List[np.ndarray]] = {}
    idx_parts = []
    tuned: dict = {}
    tune_log: list = []
    splits = list(walk_forward_splits(
        n_lab, config.WF_MIN_TRAIN, config.WF_STEP, config.WF_MAX_TRAIN))
    for k, (tr0, s, e) in enumerate(splits, 1):
        log.info("Fold %d/%d | train rows %d-%d | test %s -> %s",
                 k, len(splits), tr0, s, X.index[s].date(), X.index[e - 1].date())
        if tune and (k - 1) % config.TUNE_EVERY_FOLDS == 0:
            tuned, info = _tune_all(X.iloc[tr0:s], y[tr0:s], names)
            tune_log.append({"fold": k, "train_end": str(X.index[s - 1].date()),
                             "params": tuned, "info": info})
        for m in _all_models(names, tuned):
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
    oos["target_dir"] = feats.loc[idx, config.HSI_DIR_COL].values
    oos["target_ret"] = feats.loc[idx, config.HSI_RET_COL].values
    if tune:
        config.TUNING_LOG_PATH.write_text(json.dumps(tune_log, indent=2, default=str))
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
    if config.HSI_FLAT:   # open -> close: entry and exit every day
        turn = 2.0 * np.abs(pos)
    else:
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
    """OOS metrics per model and ensemble, plus a daily long baseline.
    Strategy: long if p > 0.5+band, short if p < 0.5-band, else flat; costs on trades."""
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


def _saved_tuned() -> bool:
    try:
        return bool(json.loads(config.TUNED_PARAMS_PATH.read_text()))
    except Exception:
        return False


def run_live(X: pd.DataFrame, y: np.ndarray, n_lab: int, feats: pd.DataFrame,
             names: Optional[List[str]] = None,
             oos: Optional[pd.DataFrame] = None, tune: bool = False) -> dict:
    """Retrain every model on the latest window and predict the live row."""
    if len(X) != n_lab + 1:
        raise RuntimeError("No unlabelled live row: the last panel row must have a NaN target.")
    tr0 = 0 if config.WF_MAX_TRAIN is None else max(0, n_lab - config.WF_MAX_TRAIN)

    if tune:
        tuned, _ = _tune_all(X.iloc[tr0:n_lab], y[tr0:n_lab], names)
        config.TUNED_PARAMS_PATH.write_text(json.dumps(tuned, indent=2))
        config.TUNED_META_PATH.write_text(json.dumps({
            "tuned_at": pd.Timestamp.now(tz="Asia/Hong_Kong").strftime("%Y-%m-%d"),
            "n_train_rows": int(n_lab - tr0)}))
    elif config.TUNED_PARAMS_PATH.exists():
        tuned = json.loads(config.TUNED_PARAMS_PATH.read_text())
    else:
        tuned = {}
    tuned_at = None
    if tuned and config.TUNED_META_PATH.exists():
        try:
            tuned_at = json.loads(config.TUNED_META_PATH.read_text()).get("tuned_at")
        except Exception:
            pass

    preds: Dict[str, float] = {}
    lgbm = None
    for m in _all_models(names, tuned):
        try:
            m.fit(X.iloc[tr0:n_lab], y[tr0:n_lab])
            preds[m.name] = float(_predict(m, X, n_lab, n_lab + 1)[0])
            if isinstance(m, LGBMModel):
                lgbm = m
        except Exception as exc:
            log.warning("Model %s failed on live fit: %s", m.name, exc)
    if not preds:
        raise RuntimeError("All models failed on the live fit.")

    ok = list(preds)
    w = _live_weights(oos, ok, X.index[n_lab - 1]).reindex(ok).fillna(0.0)
    w = w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(ok), index=ok)
    p_ens = float(sum(w[k] * preds[k] for k in ok))
    p_eq = float(np.mean(list(preds.values())))
    side = p_ens > 0.5
    agreement = float(np.mean([(p > 0.5) == side for p in preds.values()]))

    top = {}
    if lgbm is not None:
        for f in lgbm.importance().head(10).index:
            top[f] = _f(X.iloc[-1][f])

    result = {
        "generated_at": pd.Timestamp.now(tz="Asia/Hong_Kong").isoformat(timespec="seconds"),
        "target_mode": config.HSI_TARGET,
        "as_of": str(X.index[-1].date()),
        "last_labeled": str(X.index[n_lab - 1].date()),
        "target_date": str(pd.Timestamp(feats["next_date"].iloc[-1]).date())
        if "next_date" in feats.columns else None,
        "p_up": _f(p_ens),
        "p_up_equal": _f(p_eq),
        "direction": _direction(p_ens),
        "agreement": _f(agreement),
        "models": {k: _f(v) for k, v in preds.items()},
        "weights": {k: _f(w[k]) for k in ok},
        "top_features": top,
        "n_train_rows": int(n_lab - tr0),
        "tuned": bool(tuned),
        "tuned_at": tuned_at,
        "tuned_oos_weights": bool(tuned) and oos is not None
                             and config.TUNING_LOG_PATH.exists(),
    }
    return result


# ------------------------------------------------------------------ CLI
def _load_oos() -> Optional[pd.DataFrame]:
    if config.OOS_PATH.exists():
        return pd.read_parquet(config.OOS_PATH)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="HSI walk-forward backtest + live prediction")
    ap.add_argument("--mode", choices=["backtest", "live", "both"], default="both")
    ap.add_argument("--models", default=None,
                    help="comma-separated subset, e.g. logit,lgbm,xgb")
    ap.add_argument("--tune", action="store_true",
                    help="re-tune LightGBM/XGBoost with Optuna (slow)")
    args = ap.parse_args()
    names = [n.strip() for n in args.models.split(",")] if args.models else None

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    feats = builder.load_features()
    X, y, n_lab = prepare(feats)
    log.info("Target mode: %s | labelled rows: %d | features: %d",
             config.HSI_TARGET, n_lab, X.shape[1])

    oos = None
    if args.mode in ("backtest", "both"):
        oos = run_backtest(X, y, n_lab, feats, names, tune=args.tune)
        oos.to_parquet(config.OOS_PATH)
        metrics = evaluate(oos)
        metrics.to_csv(METRICS_PATH)
        log.info("OOS metrics:\n%s", metrics.round(4).to_string())

    if args.mode in ("live", "both"):
        if oos is None:
            oos = _load_oos()
        result = run_live(X, y, n_lab, feats, names, oos=oos, tune=args.tune)
        config.LIVE_PATH.write_text(json.dumps(result, indent=2, default=str))
        log.info("Live: %s p_up=%.4f (%s)", result["target_date"],
                 result["p_up"], result["direction"])


if __name__ == "__main__":
    main()

