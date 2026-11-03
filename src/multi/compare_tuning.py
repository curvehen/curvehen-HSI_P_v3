"""Untuned vs tuned pooled OOS results, on the same (date, stock) rows.
The GRU is dropped from both files so the two ensembles use the same models."""
import argparse

import numpy as np
import pandas as pd

from src import config
from src.multi import evaluate as ev

SHOW = ["lgbm", "xgb", "ens_equal", "baseline_random_p95"]
COLS = ["auc", "ic_mean", "ic_t", "ls_gross_sharpe", "ls_net_sharpe"]


def _load(path) -> pd.DataFrame:
    o = pd.read_parquet(path).drop(columns=["p_gru", "ens_equal"], errors="ignore")
    pc = [c for c in o.columns if c.startswith("p_")]
    o["ens_equal"] = o[pc].mean(axis=1)
    return o.rename_axis("date").reset_index().set_index(["date", "name"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare untuned vs tuned pooled results")
    ap.add_argument("--k", type=int, default=config.MULTI_TOPK)
    ap.add_argument("--cost-bps", type=float, default=config.COST_BPS_STOCK)
    ap.add_argument("--sims", type=int, default=100)
    a = ap.parse_args()

    pa, pb = config.MULTI_OOS_PATH, config.tuned_path(config.MULTI_OOS_PATH)
    for p in (pa, pb):
        if not p.exists():
            raise SystemExit(f"Missing {p}: run src.multi.pooled with and without --tune.")
    A, B = _load(pa), _load(pb)
    common = A.index.intersection(B.index)
    A = A.loc[common].reset_index().set_index("date")
    B = B.loc[common].reset_index().set_index("date")

    ma, _, _ = ev.evaluate(A, a.k, a.cost_bps, a.sims)
    mb, _, _ = ev.evaluate(B, a.k, a.cost_bps, a.sims)
    rows = [r for r in SHOW if r in ma.index and r in mb.index]
    cols = [c for c in COLS if c in ma.columns]
    pd.set_option("display.width", 200)
    print(pd.concat({"untuned": ma.loc[rows, cols], "tuned": mb.loc[rows, cols]}, axis=1)
          .round(4).to_string())

    Ar, Br = A.reset_index(), B.reset_index()
    X = ev._wide(Ar, config.MULTI_XS)
    print("\nPaired daily rank IC (tuned minus untuned; positive = tuning helped):")
    for c in ("p_lgbm", "p_xgb", "ens_equal"):
        if c not in Ar.columns or c not in Br.columns:
            continue
        d = (ev.daily_ic(ev._wide(Br, c), X) - ev.daily_ic(ev._wide(Ar, c), X)).dropna()
        if len(d) < 30:
            continue
        t = d.mean() / d.std(ddof=1) * np.sqrt(len(d))
        print(f"  {c.replace('p_', '')}: mean {d.mean():+.5f} | t = {t:+.2f} | n_dates = {len(d)}")


if __name__ == "__main__":
    main()
