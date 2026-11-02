"""B3-3a: pooled walk-forward training on the long (date x stock) table.

Train rows: dates in [train_start, test_start) with a label.
Test rows : dates in [test_start, test_end]; labels may be NaN (stock suspended
            next day) and are dropped at evaluation time.
"""
import argparse
import logging
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from src import config
from src.models.tabular import get_tabular_models
from src.multi import features_long as fl

log = logging.getLogger(__name__)
LABEL = "target_xs_dir"


def date_splits(dates: pd.DatetimeIndex, min_train: int, step: int,
                max_train: Optional[int]) -> Iterator[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Yield (train_start_date, test_start_date, test_end_date), strictly forward."""
    n, start = len(dates), min_train
    while start < n:
        end = min(start + step, n)
        tr0 = 0 if max_train is None else max(0, start - max_train)
        yield dates[tr0], dates[start], dates[end - 1]
        start = end


def _models(names: Optional[List[str]]) -> list:
    return [m for m in get_tabular_models() if not names or m.name in names]


def run_backtest(df: pd.DataFrame, names: Optional[List[str]] = None) -> pd.DataFrame:
    fc = fl.long_feature_columns(df)
    idx = df.index
    has_label = df[LABEL].notna().values
    dates = idx[has_label].unique().sort_values()
    splits = list(date_splits(dates, config.MULTI_WF_MIN_DATES,
                              config.MULTI_WF_STEP, config.MULTI_WF_MAX_DATES))
    if not splits:
        raise RuntimeError("Not enough dated labels for a single fold.")

    parts = []
    for k, (d0, d1, d2) in enumerate(splits, 1):
        trm = (idx >= d0) & (idx < d1) & has_label
        tem = (idx >= d1) & (idx <= d2)
        Xtr = df.loc[trm, fc].astype(float)
        ytr = df.loc[trm, LABEL].values
        Xte = df.loc[tem, fc].astype(float)
        log.info("Fold %d/%d | train %d rows (%s -> %s) | test %d rows (%s -> %s)",
                 k, len(splits), len(Xtr), d0.date(), d1.date(),
                 len(Xte), d1.date(), d2.date())

        part = df.loc[tem, ["name"] + fl.TARGETS].copy()
        for m in _models(names):
            try:
                m.fit(Xtr, ytr)
                part[f"p_{m.name}"] = m.predict_proba(Xte)
            except Exception as exc:
                log.warning("Model %s failed in fold %d: %s", m.name, k, exc)
                part[f"p_{m.name}"] = np.nan
        parts.append(part)

    oos = pd.concat(parts)
    pcols = [c for c in oos.columns if c.startswith("p_")]
    oos["ens_equal"] = oos[pcols].mean(axis=1)
    return oos


def main() -> None:
    ap = argparse.ArgumentParser(description="Pooled multi-stock walk-forward backtest")
    ap.add_argument("--models", nargs="*", help="subset: logit lgbm xgb")
    ap.add_argument("--rebuild", action="store_true", help="rebuild long features first")
    args = ap.parse_args()

    df = fl.build_long_features(cached=True) if args.rebuild else fl.load_long_features()
    oos = run_backtest(df, args.models)
    oos.to_parquet(config.MULTI_OOS_PATH)

    d = oos.dropna(subset=[LABEL, "ens_equal"])
    acc = float(((d["ens_equal"] > 0.5) == (d[LABEL] == 1)).mean())
    log.info("OOS rows %d | dates %d | ensemble accuracy vs median: %.4f",
             len(d), d.index.nunique(), acc)
    print(f"Saved {config.MULTI_OOS_PATH}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
