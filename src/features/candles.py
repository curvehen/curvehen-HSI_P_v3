"""Rule-based candlestick pattern detection (pure pandas, no TA-Lib).

Output columns are signed: +1 bullish, -1 bearish. Neutral patterns are 0/1.
Thresholds scale with each candle's own range, so they work across assets.
Every pattern uses only the current and prior candles (no look-ahead).
"""
from typing import List

import numpy as np
import pandas as pd


def _ohlc(df: pd.DataFrame, p: str):
    return (df[f"{p}_open"], df[f"{p}_high"], df[f"{p}_low"], df[f"{p}_close"])


def detect_patterns(df: pd.DataFrame, p: str) -> pd.DataFrame:
    o, h, l, c = _ohlc(df, p)
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    avg_rng = rng.rolling(14, min_periods=5).mean()
    bull, bear = c > o, c < o

    # short-term trend context, measured BEFORE the current candle
    trend = c.shift(1) / c.shift(6) - 1
    down, up = trend < 0, trend > 0

    po, pc, ph, pl = o.shift(1), c.shift(1), h.shift(1), l.shift(1)
    pbody, pbull, pbear = body.shift(1), bull.shift(1, fill_value=False), bear.shift(1, fill_value=False)
    pmid = (po + pc) / 2

    doji = body <= 0.1 * rng
    spinning = (body > 0.1 * rng) & (body <= 0.3 * rng) & (upper > body) & (lower > body)
    maru_bull = bull & (body >= 0.9 * rng)
    maru_bear = bear & (body >= 0.9 * rng)

    long_lower = (lower >= 0.6 * rng) & (upper <= 0.1 * rng)
    long_upper = (upper >= 0.6 * rng) & (lower <= 0.1 * rng)
    hammer = long_lower & down
    hanging = long_lower & up
    inv_hammer = long_upper & down
    shooting = long_upper & up

    engulf_bull = pbear & bull & (o <= pc) & (c >= po) & (body > pbody) & down
    engulf_bear = pbull & bear & (o >= pc) & (c <= po) & (body > pbody) & up

    harami_bull = pbear & bull & (np.maximum(o, c) <= po) & (np.minimum(o, c) >= pc) & down
    harami_bear = pbull & bear & (np.maximum(o, c) <= pc) & (np.minimum(o, c) >= po) & up

    piercing = pbear & bull & (o < pl) & (c > pmid) & (c < po) & down
    dark_cloud = pbull & bear & (o > ph) & (c < pmid) & (c > po) & up

    tol = 0.05 * avg_rng
    tweezer_bot = pbear & bull & ((l - pl).abs() <= tol) & down
    tweezer_top = pbull & bear & ((h - ph).abs() <= tol) & up

    # ---- three-candle patterns (candle 1 = shift(2), candle 2 = shift(1)) ----
    p2o, p2c = o.shift(2), c.shift(2)
    p2bull = bull.shift(2, fill_value=False)
    p2bear = bear.shift(2, fill_value=False)
    p2body, p2mid = body.shift(2), (p2o + p2c) / 2
    prng, p2rng = rng.shift(1), rng.shift(2)
    trend0 = c.shift(2) / c.shift(7) - 1  # trend up to candle 1's close
    down0, up0 = trend0 < 0, trend0 > 0

    morning_star = (
        p2bear & (p2body >= 0.5 * p2rng) & (pbody <= 0.3 * prng)
        & (np.maximum(po, pc) < p2c) & bull & (c > p2mid) & down0
    )
    evening_star = (
        p2bull & (p2body >= 0.5 * p2rng) & (pbody <= 0.3 * prng)
        & (np.minimum(po, pc) > p2c) & bear & (c < p2mid) & up0
    )
    big3 = (body >= 0.5 * rng) & (pbody >= 0.5 * prng) & (p2body >= 0.5 * p2rng)
    soldiers = (
        bull & pbull & p2bull & (c > pc) & (pc > p2c)
        & (o > po) & (o < pc) & (po > p2o) & (po < p2c) & big3
    )
    crows = (
        bear & pbear & p2bear & (c < pc) & (pc < p2c)
        & (o < po) & (o > pc) & (po < p2o) & (po > p2c) & big3
    )

    # ---- structure / gaps ----
    inside_bar = (h < ph) & (l > pl)
    outside_bar = (h > ph) & (l < pl)
    gap_up, gap_down = o > ph, o < pl

    def sgn(a: pd.Series, b: pd.Series) -> pd.Series:
        return a.astype(int) - b.astype(int)

    out = pd.DataFrame(
        {
            # signed: +1 bullish, -1 bearish
            "marubozu": sgn(maru_bull, maru_bear),
            "hammer_hanging": sgn(hammer, hanging),
            "invhammer_shooting": sgn(inv_hammer, shooting),
            "engulfing": sgn(engulf_bull, engulf_bear),
            "harami": sgn(harami_bull, harami_bear),
            "piercing_darkcloud": sgn(piercing, dark_cloud),
            "tweezer": sgn(tweezer_bot, tweezer_top),
            "star": sgn(morning_star, evening_star),
            "three_soldiers_crows": sgn(soldiers, crows),
            "gap": sgn(gap_up, gap_down),
            # neutral: 0/1
            "doji": doji.astype(int),
            "spinning_top": spinning.astype(int),
            "inside_bar": inside_bar.astype(int),
            "outside_bar": outside_bar.astype(int),
        },
        index=df.index,
    )

    signed_cols = [
        "marubozu", "hammer_hanging", "invhammer_shooting", "engulfing", "harami",
        "piercing_darkcloud", "tweezer", "star", "three_soldiers_crows", "gap",
    ]
    out["bull_count"] = (out[signed_cols] > 0).sum(axis=1)
    out["bear_count"] = (out[signed_cols] < 0).sum(axis=1)
    out["net_score"] = out["bull_count"] - out["bear_count"]

    # continuous candle-shape features (useful for the deep-learning models)
    out["body_ratio"] = (c - o) / rng          # signed, -1..+1
    out["upper_ratio"] = upper / rng
    out["lower_ratio"] = lower / rng
    out["close_pos"] = (c - l) / rng           # 0 = closed at low, 1 = at high
    out["range_vs_avg"] = rng / avg_rng

    # mask rows where the candle is invalid or the warm-up window is incomplete
    valid = rng.notna() & avg_rng.notna() & c.shift(7).notna()
    out = out.where(valid)
    out.columns = [f"{p}_cdl_{col}" for col in out.columns]
    return out


def pattern_columns(p: str) -> List[str]:
    """Names of the pattern columns produced for asset prefix p."""
    return [f"{p}_cdl_{k}" for k in (
        "marubozu", "hammer_hanging", "invhammer_shooting", "engulfing", "harami",
        "piercing_darkcloud", "tweezer", "star", "three_soldiers_crows", "gap",
        "doji", "spinning_top", "inside_bar", "outside_bar",
    )]


def add_candle_features(df: pd.DataFrame, assets: List[str]) -> pd.DataFrame:
    """Run detection for every asset that has full OHLC columns in df."""
    import logging
    log = logging.getLogger(__name__)
    parts = []
    for p in assets:
        need = [f"{p}_{k}" for k in ("open", "high", "low", "close")]
        if not all(n in df.columns for n in need):
            log.warning("Candle detection skipped for %s: OHLC columns missing", p)
            continue
        parts.append(detect_patterns(df, p))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
