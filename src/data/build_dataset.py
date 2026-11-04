"""Build a leakage-safe raw panel indexed by HK trading date T.

Row T = everything known before the HK open of the next trading day T'.
Target = log(HSI close T' / HSI close T).

Alignment groups:
  - HK / Asia (closed before or at HK close): value dated D is used on row D.
  - "Late" group (US session, FX, futures, FRED, southbound): value dated D is
    only available from D+1 (calendar) and is matched to the NEXT HK trading
    date T'. This also handles HK holidays correctly.
"""
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from src import config
from src.data import loaders

log = logging.getLogger(__name__)

OHLC = ["open", "high", "low", "close"]
LATE_FX = {"usdcnh", "usdhkd"}  # Yahoo FX daily bar closes ~08:00 HKT next day
PANEL_PATH = config.DATA_PROC / "panel.parquet"


def _prep(df: pd.DataFrame, name: str, keep_volume: bool) -> pd.DataFrame:
    cols = OHLC + (["volume"] if keep_volume else [])
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    out.columns = [f"{name}_{c}" for c in cols]
    return out


def _collect(assets: Dict[str, Tuple[str, bool]], refresh: bool) -> Dict[str, pd.DataFrame]:
    frames = {}
    for name, (ticker, keep_vol) in assets.items():
        df = loaders.load_yf(ticker, refresh=refresh)
        if df.empty:
            log.warning("Skipping %s (%s): no data", name, ticker)
            continue
        frames[name] = _prep(df, name, keep_vol)
    return frames


def _align(keys: pd.DatetimeIndex, right: pd.DataFrame, avail_shift_days: int) -> pd.DataFrame:
    """As-of join: for each key take the latest row with (date + shift) <= key."""
    if right.empty:
        return pd.DataFrame(index=keys)
    r = right.sort_index().ffill(limit=config.MAX_STALE_DAYS)
    r = r.copy()
    r["_avail"] = r.index + pd.Timedelta(days=avail_shift_days)
    left = pd.DataFrame({"_key": keys})
    merged = pd.merge_asof(
        left, r.reset_index(drop=True),
        left_on="_key", right_on="_avail",
        direction="backward",
        tolerance=pd.Timedelta(days=config.MAX_STALE_DAYS),
    )
    merged = merged.drop(columns=["_key", "_avail"])
    return merged


def _drop_partial_today(hsi: pd.DataFrame) -> pd.DataFrame:
    """Avoid using an unfinished intraday HSI bar as a daily close."""
    now_hk = pd.Timestamp.now(tz="Asia/Hong_Kong")
    today = now_hk.tz_localize(None).normalize()
    if now_hk.hour < 17:
        return hsi[hsi.index < today]
    return hsi


def build_panel(refresh: bool = True) -> pd.DataFrame:
    hsi_raw = loaders.load_yf(config.TARGET_TICKER, refresh=refresh)
    if hsi_raw.empty:
        raise RuntimeError("Cannot load HSI data - aborting.")

    tgt = config.TARGET_NAME
    hsi_raw = _drop_partial_today(hsi_raw)
    if tgt != "hsi" and "volume" in hsi_raw.columns:
        hsi_raw = hsi_raw[hsi_raw["volume"] > 0]   # drop suspended / zero-volume days
    hsi = _prep(hsi_raw, tgt, True).dropna(subset=[f"{tgt}_close"])

    hk_idx = hsi.index

    # Next HK trading date (T'); last row falls back to the next business day.
    nxt = hk_idx.to_series().shift(-1)
    nxt.iloc[-1] = hk_idx[-1] + BDay(1)
    next_keys = pd.DatetimeIndex(nxt.values)

    # ---- Group 1: HK-session + Asia (same-date) ----
    same_assets = dict(config.HK_ASSETS)
    late_assets = {}
    for name, spec in config.ASIA_ASSETS.items():
        (late_assets if name in LATE_FX else same_assets)[name] = spec
    same_frames = _collect(same_assets, refresh)
    same_df = pd.concat(same_frames.values(), axis=1) if same_frames else pd.DataFrame()
    same_aligned = _align(hk_idx, same_df, avail_shift_days=0)
    same_aligned.index = hk_idx

    # ---- Group 2: late group (US session, FX, FRED, southbound) ----
    late_assets.update(config.US_ASSETS)
    late_frames = _collect(late_assets, refresh)

    for name, sid in config.FRED_SERIES.items():
        s = loaders.load_fred(sid, refresh=refresh)
        if s.empty:
            log.warning("Skipping FRED %s (%s)", name, sid)
            continue
        # conservative publication lag, in observations
        late_frames[f"fred_{name}"] = s.shift(config.FRED_LAG_OBS).to_frame(f"fred_{name}")

    sb = loaders.load_southbound(refresh=refresh)
    if not sb.empty:
        late_frames["southbound"] = sb.to_frame("southbound")

    late_df = pd.concat(late_frames.values(), axis=1) if late_frames else pd.DataFrame()
    late_aligned = _align(next_keys, late_df, avail_shift_days=1)
    late_aligned.index = hk_idx

    panel = pd.concat([hsi, same_aligned, late_aligned], axis=1)
    panel = panel.loc[:, ~panel.columns.duplicated()]

    # ---- Target (last row is NaN = live inference row) ----
    close = panel[f"{config.TARGET_NAME}_close"]
    panel["target_ret"] = np.log(close.shift(-1) / close)
    panel["target_dir"] = (panel["target_ret"] > 0).astype(float)
    panel.loc[panel["target_ret"].isna(), "target_dir"] = np.nan
    opn = panel[f"{config.TARGET_NAME}_open"].where(lambda s: s > 0)
    panel["target_oc_ret"] = np.log(close.shift(-1) / opn.shift(-1))   # open(T') -> close(T')
    panel["target_oc_dir"] = (panel["target_oc_ret"] > 0).astype(float)
    panel.loc[panel["target_oc_ret"].isna(), "target_oc_dir"] = np.nan
    panel["next_date"] = next_keys


    _report_quality(panel)
    panel.to_parquet(PANEL_PATH)
    log.info("Panel saved: %s rows x %d cols -> %s", len(panel), panel.shape[1], PANEL_PATH)
    return panel


def _report_quality(panel: pd.DataFrame) -> None:
    miss = panel.isna().mean().sort_values(ascending=False)
    bad = miss[miss > 0.5]
    if not bad.empty:
        log.warning("Columns with >50%% missing: %s", ", ".join(bad.index[:20]))
    log.info("Date range: %s -> %s", panel.index.min().date(), panel.index.max().date())



def load_panel() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        return build_panel(refresh=True)
    return pd.read_parquet(PANEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    build_panel(refresh=True)
