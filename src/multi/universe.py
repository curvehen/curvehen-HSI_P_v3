"""B3-1: load the stock universe and build a long-format base table.

One row per (date T, stock). Targets refer to the next HSI trading date T'.
Rows where the stock is suspended / zero-volume on T are dropped. If the stock
is suspended on T', the target is NaN (never a multi-day return).
The last date has NaN targets and is_live = 1.

Requires the Part 1 panel built with the default target (TARGET_NAME = hsi).
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src import config
from src.data import build_dataset, loaders

log = logging.getLogger(__name__)

OHLCV = ["open", "high", "low", "close", "volume"]
BASE_PATH = config.MULTI_DATA / "long_base.parquet"


def load_universe(refresh: bool = True) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for name, ticker in config.UNIVERSE.items():
        df = loaders.load_yf(ticker, refresh=refresh)
        if df.empty or "close" not in df.columns:
            log.warning("Skipping %s (%s): no data", name, ticker)
            continue
        out[name] = df
    return out


def _one(name: str, df: pd.DataFrame, cal: pd.DatetimeIndex,
         next_dates: pd.Series) -> pd.DataFrame:
    cols = [c for c in OHLCV if c in df.columns]
    d = df.reindex(cal)[cols].copy()

    # suspended / invalid bars -> missing
    bad = d["close"].isna() | (d["close"] <= 0)
    if "volume" in d.columns:
        bad |= d["volume"].fillna(0) <= 0
    d.loc[bad, :] = np.nan

    c = d["close"]
    d["target_ret"] = np.log(c.shift(-1) / c)   # next calendar row = next HSI trading date
    o_next = d["open"].shift(-1).where(lambda s: s > 0)   # open of T'; NaN if suspended / invalid
    d["target_oc_ret"] = np.log(c.shift(-1) / o_next)     # open(T') -> close(T'), tradable
    d["next_date"] = next_dates.values
    d["name"] = name
    return d.dropna(subset=["close"])



def build_long_base(panel: Optional[pd.DataFrame] = None,
                    refresh: bool = True) -> pd.DataFrame:
    if config.TARGET_NAME != "hsi":
        raise RuntimeError("Run B3 with the default target: unset TARGET_NAME / TARGET_TICKER.")
    if panel is None:
        panel = build_dataset.build_panel(refresh=refresh)

    cal = panel.index
    nxt = pd.Series(pd.DatetimeIndex(panel["next_date"]), index=cal)

    frames = []
    for name, df in load_universe(refresh).items():
        d = _one(name, df, cal, nxt)
        n_lab = int(d["target_ret"].notna().sum())
        if n_lab < config.MIN_STOCK_ROWS:
            log.warning("Skipping %s: only %d labelled rows", name, n_lab)
            continue
        frames.append(d)
    if not frames:
        raise RuntimeError("No stock in the universe has enough data.")

    long = pd.concat(frames).sort_index()
    long.index.name = "date"

    g    g = long.groupby(level=0)["target_ret"]
    n_cs = g.transform("count")
    med = g.transform("median")
    long["target_xs"] = (long["target_ret"] - med).where(n_cs >= config.MIN_CROSS_SECTION)
    long["target_dir"] = (long["target_ret"] > 0).astype(float).where(long["target_ret"].notna())
    long["target_xs_dir"] = (long["target_xs"] > 0).astype(float).where(long["target_xs"].notna())
    long["is_live"] = (long.index == cal[-1]).astype(int)

    # open-to-close targets (same construction, tradable entry at the T' open)
    g2 = long.groupby(level=0)["target_oc_ret"]
    n_cs2, med2 = g2.transform("count"), g2.transform("median")
    long["target_oc_xs"] = (long["target_oc_ret"] - med2).where(n_cs2 >= config.MIN_CROSS_SECTION)
    long["target_oc_dir"] = (long["target_oc_ret"] > 0).astype(float).where(long["target_oc_ret"].notna())
    long["target_oc_xs_dir"] = (long["target_oc_xs"] > 0).astype(float).where(long["target_oc_xs"].notna())

    live = long[long["is_live"] == 1]

    log.info("Long base: %d rows | %d stocks | labelled %d | live rows %d | %s -> %s",
             len(long), long["name"].nunique(), int(long["target_xs"].notna().sum()),
             len(live), long.index.min().date(), long.index.max().date())
    if len(live) < config.MIN_CROSS_SECTION:
        log.warning("Only %d stocks on the live date: HK data may not be updated yet.", len(live))

    long.to_parquet(BASE_PATH)
    return long


def load_long_base() -> pd.DataFrame:
    if not BASE_PATH.exists():
        return build_long_base()
    return pd.read_parquet(BASE_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    build_long_base()
