"""B3-4a: settle the ranking log against realised returns and score it."""
import argparse
import json
import logging

import numpy as np
import pandas as pd

from src import config
from src.llm.tracker import MIN_SAMPLE, _wilson
from src.multi import evaluate as ev
from src.multi import universe

log = logging.getLogger(__name__)


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, 5) if np.isfinite(v) else None


def load_log() -> pd.DataFrame:
    if not config.MULTI_RANK_LOG.exists():
        return pd.DataFrame()
    df = pd.read_csv(config.MULTI_RANK_LOG)
    df["fd"] = pd.to_datetime(df["feature_date"])
    return df


def settle(rank_log: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    b = (base[["name", config.MULTI_RET, config.MULTI_XS]]
         .rename(columns={config.MULTI_RET: "actual_ret", config.MULTI_XS: "actual_xs"})
         .rename_axis("fd").reset_index())
    m = rank_log.merge(b, on=["fd", "name"], how="left")
    n_real = m.groupby("fd")["actual_ret"].transform("count")
    m["settled"] = n_real >= config.MIN_CROSS_SECTION
    m["no_trade"] = m["settled"] & m["actual_ret"].isna()
    m["correct"] = np.where(
        m["actual_xs"].notna(), (m["score"] > 0.5) == (m["actual_xs"] > 0), np.nan)
    return m


def daily_results(m: pd.DataFrame, cost_bps: float):
    s = m[m["settled"]]
    if s.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    S = s.pivot(index="fd", columns="name", values="side")
    P = s.pivot(index="fd", columns="name", values="score").reindex_like(S)
    R = s.pivot(index="fd", columns="name", values="actual_ret").reindex_like(S)
    X = s.pivot(index="fd", columns="name", values="actual_xs").reindex_like(S)

    is_l, is_s = (S == "LONG"), (S == "SHORT")
    nl = is_l.sum(axis=1).replace(0, np.nan)
    ns = is_s.sum(axis=1).replace(0, np.nan)
    W = (is_l.astype(float).div(nl, axis=0) - is_s.astype(float).div(ns, axis=0)).fillna(0.0)
    Rv = np.nan_to_num(R.values)

    gross, net, turn = ev._returns(W.values, Rv, cost_bps)
    d = pd.DataFrame({
        "long_ret": (np.where(W.values > 0, W.values, 0.0) * Rv).sum(axis=1),
        "short_ret": (np.where(W.values < 0, -W.values, 0.0) * Rv).sum(axis=1),
        "gross": gross, "net": net, "turnover": turn,
        "n_long": is_l.sum(axis=1).values, "n_short": is_s.sum(axis=1).values,
    }, index=S.index)
    ic = ev.daily_ic(P, X)
    d["ic"] = ic.reindex(d.index)
    return d, ic


def _picks(s: pd.DataFrame, side: str, sign: int) -> dict:
    d = s[(s["side"] == side) & s["actual_xs"].notna()]
    n = len(d)
    k = float((d["actual_xs"] * sign > 0).sum())
    lo, hi = _wilson(k, n)
    return {"n": n, "hit_rate": _f(k / n) if n else None,
            "ci95_low": _f(lo), "ci95_high": _f(hi)}


def scorecard(m: pd.DataFrame, d: pd.DataFrame, ic: pd.Series) -> dict:
    out = {"target_type": config.MULTI_TARGET,
           "n_feature_dates_logged": int(m["fd"].nunique()),
           "n_days_settled": int(len(d))}
    if d.empty:
        out["note"] = "No settled days yet."
        return out
    s = m[m["settled"]]
    sd = ic.std(ddof=1) if len(ic) > 1 else np.nan
    out.update({
        "ic_mean": _f(ic.mean()),
        "ic_t": _f(ic.mean() / sd * np.sqrt(len(ic))) if sd and sd > 0 else None,
        "ic_pos_share": _f((ic > 0).mean()),
        "long_picks_beat_median": _picks(s, "LONG", 1),
        "short_picks_lag_median": _picks(s, "SHORT", -1),
        "gross_sharpe": _f(ev._sharpe(d["gross"].values)),
        "net_sharpe": _f(ev._sharpe(d["net"].values)),
        "net_cum_log_ret": _f(d["net"].sum()),
        "max_dd_log": _f(ev._max_dd(d["net"].values)),
        "mean_long_ret": _f(d["long_ret"].mean()),
        "mean_short_ret": _f(d["short_ret"].mean()),
        "no_trade_rows": int(m["no_trade"].sum()),
        "sample_warning": (
            f"Only {len(d)} settled days (< {MIN_SAMPLE}): results are not statistically meaningful."
            if len(d) < MIN_SAMPLE else ""),
    })
    return out


def run(cost_bps: float = config.COST_BPS_STOCK) -> dict:
    rank_log = load_log()
    if rank_log.empty:
        sc = {"note": "Ranking log is empty: run src.multi.live first."}
    else:
        m = settle(rank_log, universe.load_long_base())
        d, ic = daily_results(m, cost_bps)
        m.drop(columns=["fd"]).to_csv(config.MULTI_SETTLED_PATH, index=False)
        d.to_csv(config.MULTI_DAILY_PATH)
        sc = scorecard(m, d, ic)
        log.info("Settled %d day(s) of %d logged.", len(d), m["fd"].nunique())
    config.MULTI_SCORE_PATH.write_text(json.dumps(sc, indent=2, ensure_ascii=False))
    return sc


def main() -> None:
    ap = argparse.ArgumentParser(description="Settle and score the pooled ranking log")
    ap.add_argument("--cost-bps", type=float, default=config.COST_BPS_STOCK)
    print(json.dumps(run(ap.parse_args().cost_bps), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
