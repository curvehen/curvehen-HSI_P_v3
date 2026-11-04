"""B3-3d: retrain on the latest window, score the live date, save the ranked list."""
import argparse
import json
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src import config
from src.models.common import Preprocessor
from src.models.tabular import get_tabular_models
from src.multi import features_long as fl
from src.multi.pooled import LABEL
from src.models import tuning





log = logging.getLogger(__name__)
HORIZON = {
    "oc": "open(target_date) -> close(target_date)",
    "cc": "close(feature_date) -> close(target_date)",
}


def _train_mask(idx: pd.DatetimeIndex, has: np.ndarray) -> np.ndarray:
    dates = idx[has].unique().sort_values()
    if config.MULTI_WF_MAX_DATES:
        dates = dates[-config.MULTI_WF_MAX_DATES:]
    return np.asarray(idx >= dates[0]) & has

def _tuned_at() -> Optional[str]:
    try:
        return json.loads(config.MULTI_TUNED_META_PATH.read_text()).get("tuned_at")
    except Exception:
        return None


def _tuned_params(df: pd.DataFrame, fc: List[str], trm: np.ndarray,
                  names: Optional[List[str]], tune: bool) -> dict:
    """--tune: re-tune on the latest window and save. Else reuse the saved file."""
    if tune:
        X, y = df.loc[trm, fc].astype(float), df.loc[trm, LABEL].values
        dts = df.index[trm].values
        out = {}
        for n in config.TUNE_MODELS:
            if names and n not in names:
                continue
            p, _ = tuning.tune(n, X, y, dates=dts)
            if p:
                out[n] = p
        config.MULTI_TUNED_PARAMS_PATH.write_text(json.dumps(out, indent=2))
        config.MULTI_TUNED_META_PATH.write_text(json.dumps({
            "tuned_at": pd.Timestamp.now(tz="Asia/Hong_Kong").strftime("%Y-%m-%d")}))
        return out

    if config.MULTI_TUNED_PARAMS_PATH.exists():
        return json.loads(config.MULTI_TUNED_PARAMS_PATH.read_text())
    return {}


def _tabular(df: pd.DataFrame, fc: List[str], trm: np.ndarray, live_m: np.ndarray,
             names: Optional[List[str]], tuned: Optional[dict] = None) -> Dict[str, np.ndarray]:
    Xtr, ytr = df.loc[trm, fc].astype(float), df.loc[trm, LABEL].values
    Xlv = df.loc[live_m, fc].astype(float)
    out = {}
    for m in get_tabular_models(tuned):
        if names and m.name not in names:
            continue
        try:
            m.fit(Xtr, ytr)
            out[m.name] = m.predict_proba(Xlv)
        except Exception as exc:
            log.warning("Model %s failed on live fit: %s", m.name, exc)
    return out


def _gru(df: pd.DataFrame, fc: List[str], seed: int) -> pd.Series:
    from src.multi import pooled_deep as pdp
    g = df.sort_values("name", kind="stable")       # stock blocks, date order inside each
    X = g[fc].astype(float)
    y = g[LABEL].values.astype(np.float32)
    has = ~np.isnan(y)
    gidx = g.index
    start = pdp._block_start(g["name"].values)
    trm = _train_mask(gidx, has)
    dts = gidx[trm].unique().sort_values()
    cut = dts[int(len(dts) * (1 - config.VAL_FRAC))]
    tr = np.where(trm & np.asarray(gidx < cut))[0]
    va = np.where(trm & np.asarray(gidx >= cut))[0]
    pre = Preprocessor().fit(X.iloc[np.where(trm)[0]])
    Z = pre.transform(X)
    net = pdp._fit(Z, start, tr, va, y, seed)
    pos = np.where(g["is_live"].values == 1)[0]
    return pd.Series(pdp._predict(net, Z, start, pos), index=g["name"].values[pos])


def build_ranking(df: pd.DataFrame, names: Optional[List[str]], use_gru: bool,
                  k: int, seed: int, tune: bool = False):

    df = df.sort_index(kind="stable")
    fc = fl.long_feature_columns(df)
    live_m = (df["is_live"] == 1).values
    if not live_m.any():
        raise RuntimeError("No live rows: HK data for the latest date is not available yet.")
    trm = _train_mask(df.index, df[LABEL].notna().values)
    live = df.loc[live_m]
    warns: List[str] = []

    P = pd.DataFrame(index=live["name"].values)
    tuned = _tuned_params(df, fc, trm, names, tune)
    for name, p in _tabular(df, fc, trm, live_m, names, tuned).items():
        P[name] = p

    if use_gru and (not names or "gru" in names):
        try:
            P["gru"] = _gru(df, fc, seed).reindex(P.index).values
        except Exception as exc:
            log.warning("GRU failed on live fit: %s", exc)
            warns.append(f"GRU failed: {type(exc).__name__}")
    if P.empty:
        raise RuntimeError("All models failed on the live fit.")

    score = P.mean(axis=1)
    agree = P.gt(0.5).eq(score.gt(0.5), axis=0).mean(axis=1)
    live_i = live.set_index("name")
    r = pd.DataFrame({
        "name": P.index,
        "ticker": [config.UNIVERSE.get(n, "") for n in P.index],
        "score": score.values,
        "agreement": agree.values,
        "ref_close": live_i["ref_close"].reindex(P.index).values,
        "live_nan": live_i[fc].isna().sum(axis=1).reindex(P.index).values,
    })
    for c in P.columns:
        r[f"p_{c}"] = P[c].values
    r = r.sort_values("score", ascending=False).reset_index(drop=True)
    r["rank"] = np.arange(1, len(r) + 1)
    r["side"] = "FLAT"
    n = len(r)
    if n >= max(2 * k, config.MIN_CROSS_SECTION):
        r.loc[r["rank"] <= k, "side"] = "LONG"
        r.loc[r["rank"] > n - k, "side"] = "SHORT"
    else:
        warns.append(f"Only {n} live stocks: no positions assigned.")

    feature_date = live.index.max()
    today = pd.Timestamp.now(tz="Asia/Hong_Kong").tz_localize(None).normalize()
    if (today - feature_date).days > 4:
        warns.append(f"Live date {feature_date.date()} is more than 4 days old: data may be stale.")
    total = df["name"].nunique()
    if n < total:
        warns.append(f"{total - n} of {total} stocks have no live row (suspended or missing data).")
    if r["live_nan"].mean() / max(len(fc), 1) > 0.10:
        warns.append("More than 10% of live features are missing: overnight data may not be updated.")

    meta = {
        "generated_at": pd.Timestamp.now(tz="Asia/Hong_Kong").isoformat(timespec="seconds"),
        "feature_date": str(feature_date.date()),
        "target_date": str(pd.Timestamp(live["next_date"].max()).date()),
        "target_type": config.MULTI_TARGET,
        "horizon": HORIZON[config.MULTI_TARGET],
        "models": list(P.columns),
        "tuned_models": sorted(tuned),
        "tuned_at": _tuned_at() if tuned else None,
        "top_k": k,
        "n_stocks": n,
        "n_train_rows": int(trm.sum()),
        "n_train_dates": int(df.index[trm].nunique()),
        "n_features": len(fc),
        "warnings": warns,
    }
    return r, meta


def save(r: pd.DataFrame, meta: dict) -> None:
    out = r.round(6)
    out.to_csv(config.MULTI_LIVE_CSV, index=False)
    recs = out.astype(object).where(out.notna(), None).to_dict("records")
    config.MULTI_LIVE_JSON.write_text(
        json.dumps({**meta, "ranking": recs}, indent=2, ensure_ascii=False))

    log_cols = ["name", "score", "rank", "side", "ref_close"]
    new = r[log_cols].copy()
    new.insert(0, "target_date", meta["target_date"])
    new.insert(0, "feature_date", meta["feature_date"])
    if config.MULTI_RANK_LOG.exists():
        old = pd.read_csv(config.MULTI_RANK_LOG)
        old = old[old["feature_date"] != meta["feature_date"]]   # same-day rerun replaces
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(config.MULTI_RANK_LOG, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pooled multi-stock live ranking")
    ap.add_argument("--models", nargs="*", help="subset: logit lgbm xgb gru")
    ap.add_argument("--k", type=int, default=config.MULTI_TOPK)
    ap.add_argument("--no-gru", action="store_true")
    ap.add_argument("--cached", action="store_true",
                    help="reuse saved long features (may be stale)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--tune", action="store_true",
                    help="re-tune LightGBM/XGBoost on the latest window")
    a = ap.parse_args()

    df = fl.build_long_features(refresh=True, cached=a.cached)
    r, meta = build_ranking(df, a.models, not a.no_gru, a.k, a.seed, tune=a.tune)

    save(r, meta)

    pd.set_option("display.width", 200)
    cols = ["rank", "name", "ticker", "score", "agreement", "side", "ref_close", "live_nan"]
    print(f"Features as of {meta['feature_date']} -> target {meta['target_date']} ({meta['horizon']})")
    print(r[cols].round(4).to_string(index=False))
    for w in meta["warnings"]:
        print("WARNING:", w)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
