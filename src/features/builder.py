"""Assemble the model-ready feature table from the Part 1 panel.

Output: one row per HK trading date T. Features use only information known
before the next HK open. Targets (target_ret, target_dir) refer to T' (next
trading day). The last row has NaN targets and is used for live inference.
"""
import logging

import numpy as np
import pandas as pd

from src import config
from src.data import build_dataset
from src.features import candles, technical

log = logging.getLogger(__name__)

# FRED series expressed in rate / index points: use level + differences.
RATE_LIKE = {
    "fred_us10y", "fred_t10y2y", "fred_hy_spread", "fred_fed_funds",
    "fred_vix_fred", "fred_vxn_fred", "fred_ovx",
}
LEVEL_ASSETS = {"vix", "vxn"}  # Yahoo vol indices: keep level as well as return
META_PREFIXES = ("target_", "ref_")
META_COLS = {"next_date"}


def _price_prefixes(panel: pd.DataFrame) -> list:
    return sorted(
        {c[:-6] for c in panel.columns
         if c.endswith("_close") and c[:-6] != config.TARGET_NAME}
    )


def _cross_asset(panel: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for p in _price_prefixes(panel):
        c = panel[f"{p}_close"].where(lambda s: s > 0)
        lc = np.log(c)
        out[f"{p}_ret1"] = lc.diff()
        out[f"{p}_ret5"] = lc.diff(5)
        if p in LEVEL_ASSETS:
            out[f"{p}_lvl"] = c
        vcol = f"{p}_volume"
        if vcol in panel.columns:
            v = panel[vcol].replace(0, np.nan)
            out[f"{p}_volz"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    return pd.DataFrame(out, index=panel.index)


def _fred(panel: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in [c for c in panel.columns if c.startswith("fred_")]:
        s = panel[c]
        if c in RATE_LIKE:
            out[f"{c}_lvl"] = s
            out[f"{c}_d1"] = s.diff()
            out[f"{c}_d5"] = s.diff(5)
        else:
            ls = np.log(s.where(s > 0))
            out[f"{c}_ret1"] = ls.diff()
            out[f"{c}_ret5"] = ls.diff(5)
    return pd.DataFrame(out, index=panel.index)


def _southbound(panel: pd.DataFrame) -> pd.DataFrame:
    if "southbound" not in panel.columns:
        return pd.DataFrame(index=panel.index)
    s = panel["southbound"]
    sd = s.rolling(20).std().replace(0, np.nan)
    return pd.DataFrame(
        {
            "sb_net": s,
            "sb_z20": (s - s.rolling(20).mean()) / sd,
            "sb_sum5": s.rolling(5).sum(),
        },
        index=panel.index,
    )


def _adr_premium(panel: pd.DataFrame) -> pd.DataFrame:
    """US ADR close converted to HKD vs the HK line's close on row T.
    Ratios in config.ADR_PAIRS are unverified: check before relying on this."""
    out = {}
    if "usdhkd_close" in panel.columns:
        fx = panel["usdhkd_close"].where(lambda s: (s > 7.7) & (s < 7.9))
        fx = fx.fillna(config.FX_FALLBACK_HKD)
    else:
        fx = pd.Series(config.FX_FALLBACK_HKD, index=panel.index)
    for adr, (hk, ratio) in config.ADR_PAIRS.items():
        a, h = f"{adr}_close", f"{hk}_close"
        if a not in panel.columns or h not in panel.columns:
            continue
        adr_px = panel[a].where(lambda s: s > 0)
        hk_px = panel[h].where(lambda s: s > 0)
        prem = adr_px * fx / (ratio * hk_px) - 1
        out[f"adr_{adr}_prem"] = prem
        out[f"adr_{adr}_prem_d1"] = prem.diff()
    return pd.DataFrame(out, index=panel.index)


def _composites(cross: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for g, names in config.COMPOSITES.items():
        cols = [f"{n}_ret1" for n in names if f"{n}_ret1" in cross.columns]
        if not cols:
            continue
        blk = cross[cols]
        out[f"comp_{g}_ret1"] = blk.mean(axis=1)
        out[f"comp_{g}_breadth"] = (blk > 0).astype(float).where(blk.notna()).mean(axis=1)
    return pd.DataFrame(out, index=cross.index)


def feature_columns(df: pd.DataFrame) -> list:
    """Model input columns = everything except targets / reference columns."""
    return [
        c for c in df.columns
        if c not in META_COLS and not c.startswith(META_PREFIXES)
    ]


def build_features(panel: pd.DataFrame = None) -> pd.DataFrame:
    if panel is None:
        panel = build_dataset.load_panel()

    cross = _cross_asset(panel)
    parts = [
        technical.hsi_technical(panel, config.TARGET_NAME)
        technical.calendar_features(panel.index, panel["next_date"]),
        candles.add_candle_features(panel, config.CANDLE_ASSETS),
        cross,
        _fred(panel),
        _southbound(panel),
        _adr_premium(panel),
        _composites(cross),
    ]
    feats = pd.concat(parts, axis=1)
    feats = feats.loc[:, ~feats.columns.duplicated()]
    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.iloc[config.WARMUP_ROWS:]

    # drop columns that are mostly missing recently, or constant
    recent_miss = feats.tail(config.RECENT_ROWS).isna().mean()
    sparse = recent_miss[recent_miss > config.MAX_MISSING_FRAC].index.tolist()
    const = [c for c in feats.columns if feats[c].nunique(dropna=True) <= 1]
    drop = sorted(set(sparse) | set(const))
    if drop:
        log.warning("Dropping %d columns (sparse/constant): %s", len(drop), ", ".join(drop[:25]))
        feats = feats.drop(columns=drop)

    # targets and references (aligned to row T)
    feats["target_ret"] = panel.loc[feats.index, "target_ret"]
    feats["target_dir"] = panel.loc[feats.index, "target_dir"]
    feats["next_date"] = panel.loc[feats.index, "next_date"]
    feats["ref_hsi_close"] = panel.loc[feats.index, f"{config.TARGET_NAME}_close"]

    n_feat = len(feature_columns(feats))
    last_nan = int(feats[feature_columns(feats)].iloc[-1].isna().sum())
    log.info("Features: %d rows x %d features | live-row NaN: %d", len(feats), n_feat, last_nan)

    feats.to_parquet(config.FEATURES_PATH)
    return feats


def load_features() -> pd.DataFrame:
    if not config.FEATURES_PATH.exists():
        return build_features()
    return pd.read_parquet(config.FEATURES_PATH)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    build_features()
