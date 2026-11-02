"""B3-2: features for the long (date x stock) table.

Leakage control:
  - Per-stock features use only that stock's rows <= T.
  - Cross-sectional features use only same-date (T) values.
  - Market features are the Part 2 table, already aligned to row T.
Targets are never used as inputs.
"""
import argparse
import logging
from typing import List

import numpy as np
import pandas as pd

from src import config
from src.data import build_dataset
from src.features import builder, candles, technical
from src.multi import universe

log = logging.getLogger(__name__)

FEATURES_LONG_PATH = config.MULTI_DATA / "features_long.parquet"
TARGETS = ["target_ret", "target_xs", "target_dir", "target_xs_dir"]
META = {"name", "next_date", "is_live", "ref_close", "date"}
CS_COLS = [
    "s_ret1_lag", "s_ret_5d", "s_ret_20d", "s_rv20", "s_vol_z20", "s_rsi14",
    "s_dist_sma50", "s_turn_z60", "rel_ret1", "rel_ret5",
]


def long_feature_columns(df: pd.DataFrame) -> List[str]:
    skip = set(META)
    if not config.USE_STOCK_ID:
        skip.add("stock_id")
    return [c for c in df.columns if c not in skip and not c.startswith("target_")]


def _stock_features(d: pd.DataFrame, hsi
