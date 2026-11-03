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

import json
from src.models import tuning

log = logging.getLogger(__name__)
LABEL = config.MULTI_LABEL


def date_splits(dates: pd.DatetimeIndex, min_train: int, step: int,
                max_train: Optional[int]) -> Iterator[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Yield (train_start_date, test_start_date, test_end_date), strictly forward."""
    n, start = len(dates), min_train
    while start < n:
        end = min(start + step, n)
        tr0 = 0 if max_train is None else max(0, start - max_train)
        yield dates[tr0], dates[start], dates[end - 1]
        start = end


def _models(names: Optional[List[str]], tuned: Optional[dict] = None) -> list:
    return [m for m in get_tabular_models(tuned) if not names or m.name in names]



def run_backtest(df: pd.DataFrame, names: Optional[List[str]] = None,
                 tune: bool = False) -> pd.DataFrame:
    tuned: dict = {}
    tune_log: list = []
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

        if tune and (k - 1) % config.TUNE_EVERY_FOLDS == 0:
            tuned, info = {}, {}
            dts = df.index[trm].values
            for n in config.TUNE_MODELS:
                if names and n not in names:
                    continue
                p, i = tuning.tune(n, Xtr, ytr, dates=dts)
                if p:
                    tuned[n] = p
                info[n] = i
            tune_log.append({"fold": k, "train_end": str(d1.date()),
                             "params": tuned, "info": info})

                
        part = df.loc[tem, ["name"] + fl.TARGETS].copy()
        for m in _models(names, tuned):
            try:
                m.fit(Xtr, ytr)
                part[f"p_{m.name}"] = m.predict_proba(Xte)
            except Exception as exc:
                log.warning("Model %s failed in fold %d: %s", m.name, k, exc)
                part[f"p_{m.name}"] = np.nan
        parts.append(part)

    if tune:
        config.MULTI_TUNING_LOG_PATH.write_text(json.dumps(tune_log, indent=2, default=str))

    oos = pd.concat(parts)
    pcols = [c for c in oos.columns if c.startswith("p_")]
    oos["ens_equal"] = oos[pcols].mean(axis=1)
    return oos


def main() -> None:
    ap = argparse.ArgumentParser(description="Pooled multi-stock walk-forward backtest")
    ap.add_argument("--models", nargs="*", help="subset: logit lgbm xgb")
    ap.add_argument("--rebuild", action="store_true", help="rebuild long features first")
    ap.add_argument("--tune", action="store_true",
                    help="tune LightGBM/XGBoost per fold (writes *_tuned files)")

    args = ap.parse_args()

    df = fl.build_long_features(cached=True) if args.rebuild else fl.load_long_features()
    oos = run_backtest(df, args.models, tune=args.tune)
    out = config.tuned_path(config.MULTI_OOS_PATH) if args.tune else config.MULTI_OOS_PATH
    oos.to_parquet(out)


    d = oos.dropna(subset=[LABEL, "ens_equal"])
    acc = float(((d["ens_equal"] > 0.5) == (d[LABEL] == 1)).mean())
    log.info("OOS rows %d | dates %d | ensemble accuracy vs median: %.4f",
             len(d), d.index.nunique(), acc)

    print(f"Saved {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
