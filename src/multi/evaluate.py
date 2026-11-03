"""B3-3b: evaluate pooled out-of-sample predictions.

Selection on date T uses only that date's predictions. Stocks with a missing
target (suspended next day) are still selectable and contribute 0 return,
so the backtest never picks stocks based on future availability.
"""
import argparse
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config

log = logging.getLogger(__name__)
ANN = 252
FLAT = config.MULTI_TARGET == "oc"   # positions are closed every day
LABEL = config.MULTI_LABEL


def _wide(o: pd.DataFrame, col: str) -> pd.DataFrame:
    return o.pivot(index="date", columns="name", values=col)


def ls_weights(A: np.ndarray, k: int) -> np.ndarray:
    """+1/k on the k highest scores, -1/k on the k lowest, per row. NaN = not eligible."""
    W = np.zeros_like(A, dtype=float)
    for i in range(len(A)):
        ok = np.where(~np.isnan(A[i]))[0]
        if len(ok) < max(2 * k, config.MIN_CROSS_SECTION):
            continue
        o = ok[np.argsort(A[i, ok], kind="stable")]
        W[i, o[-k:]] = 1.0 / k
        W[i, o[:k]] = -1.0 / k
    return W


def _returns(W: np.ndarray, R: np.ndarray, cost_bps: float):
    gross = (W * R).sum(axis=1)
    if FLAT:   # enter at open, exit at close: every weight is traded twice per day
        turn = 2.0 * np.abs(W).sum(axis=1)
    else:
        turn = np.abs(np.diff(W, axis=0, prepend=np.zeros((1, W.shape[1])))).sum(axis=1)
    return gross, gross - turn * cost_bps / 1e4, turn



def _sharpe(x: np.ndarray) -> float:
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / sd * np.sqrt(ANN)) if sd > 0 else np.nan


def _max_dd(x: np.ndarray) -> float:
    cum = np.cumsum(x)
    return float((cum - np.maximum.accumulate(cum)).min())


def daily_ic(P: pd.DataFrame, X: pd.DataFrame) -> pd.Series:
    out = {}
    for d in P.index:
        a, b = P.loc[d], X.loc[d]
        m = a.notna() & b.notna()
        if m.sum() >= config.MIN_CROSS_SECTION:
            out[d] = a[m].rank().corr(b[m].rank())
    return pd.Series(out, dtype=float).dropna()


def evaluate(oos: pd.DataFrame, k: int, cost_bps: float, sims: int, seed: int = 42):
    o = oos.rename_axis("date").reset_index()
    R = _wide(o, config.MULTI_RET)
    X = _wide(o, config.MULTI_XS)
    Rv = np.nan_to_num(R.values)
    cols = [c for c in o.columns if c.startswith("p_")] + ["ens_equal"]

    rows: List[dict] = []
    daily: Dict[str, pd.Series] = {}
    ics: Dict[str, pd.Series] = {}
    for c in cols:
        P = _wide(o, c)
        ic = daily_ic(P, X)
        ics[c] = ic
        W = ls_weights(P.values, k)
        gross, net, turn = _returns(W, Rv, cost_bps)
        daily[c] = pd.Series(net, index=P.index)

        d = o.dropna(subset=[c, LABEL])
        both = len(np.unique(d[c])) > 1 and d[LABEL].nunique() > 1

        sd = ic.std(ddof=1)
        rows.append({
            "model": c.replace("p_", ""),
            "n_rows": len(d), "n_dates": int(P.shape[0]),
            "auc": float(roc_auc_score(d[LABEL], d[c])) if both else np.nan,
            "acc": float(((d[c] > 0.5) == (d[LABEL] == 1)).mean()),
            "ic_mean": float(ic.mean()), "ic_std": float(sd),
            "icir": float(ic.mean() / sd * np.sqrt(ANN)) if sd > 0 else np.nan,
            "ic_t": float(ic.mean() / sd * np.sqrt(len(ic))) if sd > 0 else np.nan,
            "ic_pos": float((ic > 0).mean()),
            "ls_gross_sharpe": _sharpe(gross), "ls_net_sharpe": _sharpe(net),
            "ls_net_ann_ret": float(net.mean() * ANN), "ls_max_dd": _max_dd(net),
            "turnover_per_day": float(turn.mean()),
        })

    # baseline 1: equal-weight universe long (no cost; it is nearly buy-and-hold)
    ew = np.nanmean(R.values, axis=1)
    ew = np.nan_to_num(ew)
    rows.append({"model": "baseline_universe_long", "n_dates": len(ew),
                 "ls_gross_sharpe": _sharpe(ew), "ls_net_sharpe": _sharpe(ew),
                 "ls_net_ann_ret": float(ew.mean() * ANN), "ls_max_dd": _max_dd(ew)})

    # baseline 2: random top-k / bottom-k picks, same eligibility and costs
    if sims > 0:
        rng = np.random.default_rng(seed)
        avail = ~np.isnan(_wide(o, "ens_equal").values)
        sh = []
        for _ in range(sims):
            A = np.where(avail, rng.random(avail.shape), np.nan)
            sh.append(_sharpe(_returns(ls_weights(A, k), Rv, cost_bps)[1]))
        sh = np.array(sh)
        rows.append({"model": "baseline_random_mean", "ls_net_sharpe": float(np.nanmean(sh))})
        rows.append({"model": "baseline_random_p95", "ls_net_sharpe": float(np.nanpercentile(sh, 95))})

    return pd.DataFrame(rows).set_index("model"), pd.DataFrame(daily), ics


def by_year(daily: pd.DataFrame, ics: Dict[str, pd.Series], model: str = "ens_equal") -> pd.DataFrame:
    ic, net = ics[model], daily[model]
    return pd.DataFrame({
        "ic_mean": ic.groupby(ic.index.year).mean(),
        "ls_net_ret": net.groupby(net.index.year).sum(),
        "n_dates": net.groupby(net.index.year).size(),
    })


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate pooled OOS predictions")
    ap.add_argument("--k", type=int, default=config.MULTI_TOPK)
    ap.add_argument("--cost-bps", type=float, default=config.COST_BPS_STOCK)
    ap.add_argument("--sims", type=int, default=100, help="random baseline draws; 0 = skip")
    ap.add_argument("--tuned", action="store_true", help="evaluate the *_tuned OOS file")
    args = ap.parse_args()

    sfx = config.tuned_path
    oos_p = sfx(config.MULTI_OOS_PATH) if args.tuned else config.MULTI_OOS_PATH
    if not oos_p.exists():
        raise SystemExit(f"Missing {oos_p}: run `python -m src.multi.pooled` first.")
    oos = pd.read_parquet(oos_p)
    metrics, daily, ics = evaluate(oos, args.k, args.cost_bps, args.sims)

    metrics.to_csv(sfx(config.MULTI_METRICS_PATH) if args.tuned else config.MULTI_METRICS_PATH)
    daily.to_parquet(sfx(config.MULTI_LS_PATH) if args.tuned else config.MULTI_LS_PATH)
    yr = by_year(daily, ics)
    yr.to_csv(sfx(config.MULTI_YEAR_PATH) if args.tuned else config.MULTI_YEAR_PATH)

    pd.set_option("display.width", 200)
    print(metrics.round(4).to_string())
    print("\nBy year (ens_equal):")
    print(yr.round(4).to_string())



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
