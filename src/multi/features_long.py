"""B3-2: features for the long (date x stock) table.

Leakage control:
  - Per-stock features use only that stock's rows <= T.
  - Cross-sectional features use only same-date (T) values.
  - Market features are the Part 2 table, already aligned to row T.
Targets are never used as inputs. Raw OHLCV is not kept (ref_close only).
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
TARGETS = ["target_ret", "target_xs", "target_dir", "target_xs_dir",
           "target_oc_ret", "target_oc_xs", "target_oc_dir", "target_oc_xs_dir"]

META = {"name", "next_date", "is_live", "ref_close", "date"}
CS_COLS = [
    "s_ret1_lag", "s_ret_5d", "s_ret_20d", "s_rv20", "s_vol_z20", "s_rsi14",
    "s_dist_sma50", "s_turn_z60", "rel_ret1", "rel_ret5",
]
# Stable ids from the full universe list, so ids do not shift if a ticker fails.
STOCK_IDS = {n: i for i, n in enumerate(sorted(config.UNIVERSE))}


def long_feature_columns(df: pd.DataFrame) -> List[str]:
    skip = set(META)
    if not config.USE_STOCK_ID:
        skip.add("stock_id")
    return [c for c in df.columns if c not in skip and not c.startswith("target_")]


def _stock_features(d: pd.DataFrame, hsi_lc: pd.Series) -> pd.DataFrame:
    """Own technicals + candles + relative-to-HSI features for one stock."""
    x = d[["open", "high", "low", "close", "volume"]].copy()
    x.columns = [f"s_{c}" for c in x.columns]
    f = pd.concat([technical.hsi_technical(x, "s"),
                   candles.detect_patterns(x, "s")], axis=1)

    lc = np.log(d["close"])
    hs = hsi_lc.reindex(d.index)
    r1, h1 = lc.diff(), hs.diff()       # both span the same rows, so gaps match
    f["rel_ret1"] = r1 - h1
    f["rel_ret5"] = lc.diff(5) - hs.diff(5)
    f["rel_ret20"] = lc.diff(20) - hs.diff(20)

    var = h1.rolling(60, min_periods=40).var().replace(0, np.nan)
    beta = r1.rolling(60, min_periods=40).cov(h1) / var
    f["s_beta60"] = beta
    f["s_idio_vol20"] = (r1 - beta * h1).rolling(20, min_periods=15).std()

    dv = np.log((d["close"] * d["volume"]).where(lambda s: s > 0))   # log HKD turnover
    f["s_dolvol_log"] = dv.rolling(20, min_periods=10).mean()
    m60 = dv.rolling(60, min_periods=30)
    f["s_turn_z60"] = (dv - m60.mean()) / m60.std()
    return f


def _add_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """Same-date rank / z-score of selected features, plus universe aggregates."""
    cs = [c for c in CS_COLS if c in df.columns]
    grp = df.groupby(level=0)
    ok = (grp["name"].transform("count") >= config.MIN_CROSS_SECTION).values

    rank = grp[cs].rank(pct=True)
    mean = grp[cs].transform("mean")
    sd = grp[cs].transform("std").replace(0, np.nan)
    z = (df[cs] - mean) / sd

    new = {}
    for c in cs:
        new[f"cs_rank_{c}"] = np.where(ok, rank[c].values, np.nan)
        new[f"cs_z_{c}"] = np.where(ok, z[c].clip(-5, 5).values, np.nan)

    r1 = df["s_ret1_lag"]
    up = (r1 > 0).astype(float).where(r1.notna())
    new["uni_ret1_mean"] = grp["s_ret1_lag"].transform("mean").values
    new["uni_breadth"] = up.groupby(level=0).transform("mean").values
    new["uni_disp"] = grp["s_ret1_lag"].transform("std").values
    new["uni_rv20_mean"] = grp["s_rv20"].transform("mean").values
    return df.assign(**new)


def build_long_features(refresh: bool = True, cached: bool = False) -> pd.DataFrame:
    if config.TARGET_NAME != "hsi":
        raise RuntimeError("Run B3 with the default target: unset TARGET_NAME / TARGET_TICKER.")
    if cached:
        panel = build_dataset.load_panel()
        long = universe.load_long_base()
    else:
        panel = build_dataset.build_panel(refresh=refresh)
        long = universe.build_long_base(panel, refresh=refresh)

    hsi_lc = np.log(panel["hsi_close"].where(lambda s: s > 0))

    # 1) per-stock features
    parts = []
    for name, g in long.groupby("name"):
        f = _stock_features(g, hsi_lc)
        meta = g[["name", "next_date", "is_live"] + TARGETS].copy()
        meta["ref_close"] = g["close"]
        meta["stock_id"] = STOCK_IDS.get(name, -1)
        parts.append(pd.concat([meta, f], axis=1))
    df = pd.concat(parts).sort_index(kind="stable")
    df.index.name = "date"

    # 2) market features (Part 2 table), same for every stock on a date.
    #    Inner join also drops the Part 2 warm-up rows.
    feats = builder.build_features(panel)
    mkt = feats[builder.feature_columns(feats)].add_prefix("m_")
    df = df.join(mkt, how="inner")

    # 3) drop each stock's own warm-up rows (e.g. recent listings)
    if "s_dist_sma200" in df.columns:
        df = df[df["s_dist_sma200"].notna()]
    df = df.replace([np.inf, -np.inf], np.nan)

    # 4) cross-sectional features
    df = _add_cross_section(df)

    # 5) drop sparse / constant columns (missing share measured on recent dates)
    df = df.sort_index(kind="stable")
    fc = long_feature_columns(df)
    dates = df.index.unique()
    recent = df[df.index >= dates[-min(config.RECENT_ROWS, len(dates))]]
    miss = recent[fc].isna().mean()
    sparse = miss[miss > config.MAX_MISSING_FRAC].index.tolist()
    const = [c for c in fc if df[c].nunique(dropna=True) <= 1]
    drop = sorted(set(sparse) | set(const))
    if drop:
        log.warning("Dropping %d columns (sparse/constant): %s",
                    len(drop), ", ".join(drop[:25]))
        df = df.drop(columns=drop)

    fc = long_feature_columns(df)
    live = df[df["is_live"] == 1]
    log.info("Long features: %d rows | %d stocks | %d features | live rows %d | "
             "labelled target_xs %d",
             len(df), df["name"].nunique(), len(fc), len(live),
             int(df["target_xs"].notna().sum()))
    if len(live):
        log.info("Live-row NaN per stock (mean): %.1f", live[fc].isna().sum(axis=1).mean())
    else:
        log.warning("No live rows: HK data may not be updated yet.")

    df.to_parquet(FEATURES_LONG_PATH)
    return df


def load_long_features() -> pd.DataFrame:
    if not FEATURES_LONG_PATH.exists():
        return build_long_features()
    return pd.read_parquet(FEATURES_LONG_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Build the long feature table")
    ap.add_argument("--cached", action="store_true",
                    help="reuse the saved panel and long_base instead of re-downloading")
    build_long_features(refresh=True, cached=ap.parse_args().cached)

