"""Compare untuned vs tuned out-of-sample results (same folds, same rows)."""
import numpy as np
import pandas as pd

from src import config

EPS = 1e-6
SHOW = ["lgbm", "xgb", "ens_equal", "ens_weighted"]
COLS = ["auc", "logloss", "sharpe"]


def _ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main() -> None:
    for p in (config.OOS_PATH, config.OOS_TUNED_PATH,
              config.METRICS_PATH, config.METRICS_TUNED_PATH):
        if not p.exists():
            raise SystemExit(f"Missing {p}: run the backtest with and without --tune.")
    ma = pd.read_csv(config.METRICS_PATH, index_col=0)
    mb = pd.read_csv(config.METRICS_TUNED_PATH, index_col=0)
    rows = [r for r in SHOW if r in ma.index and r in mb.index]
    cols = [c for c in COLS if c in ma.columns]
    t = pd.concat({"untuned": ma.loc[rows, cols], "tuned": mb.loc[rows, cols]}, axis=1)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string())

    a, b = pd.read_parquet(config.OOS_PATH), pd.read_parquet(config.OOS_TUNED_PATH)
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    y = a["target_dir"].values
    print("\nPaired log-loss gain (untuned minus tuned; positive = tuning helped):")
    for m in ("lgbm", "xgb"):
        c = f"p_{m}"
        if c not in a.columns or c not in b.columns:
            continue
        ok = a[c].notna().values & b[c].notna().values & np.isfinite(y)
        d = _ll(a[c].values[ok], y[ok]) - _ll(b[c].values[ok], y[ok])
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {m}: mean {d.mean():+.5f} | t = {d.mean() / se:+.2f} | n = {len(d)}")


if __name__ == "__main__":
    main()
