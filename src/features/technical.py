"""Single-asset technical indicators for HSI + calendar features.
Everything uses data up to and including row T only (no look-ahead)."""
import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    out = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return out.mask((dn == 0) & up.notna(), 100.0)


def _streak(r1: pd.Series) -> pd.Series:
    """Signed count of consecutive up (+) / down (-) days."""
    sign = np.sign(r1).fillna(0)
    grp = (sign != sign.shift()).cumsum()
    return sign * (sign.groupby(grp).cumcount() + 1)


def hsi_technical(df: pd.DataFrame, p: str = "hsi") -> pd.DataFrame:
    o, h, l, c = (df[f"{p}_{k}"] for k in ("open", "high", "low", "close"))
    lc = np.log(c)
    r1 = lc.diff()
    out = {}

    # returns and lags (ret1 = today's close-to-close return)
    for k in range(1, 6):
        out[f"{p}_ret{k}_lag"] = r1.shift(k - 1)
    for n in (5, 10, 20):
        out[f"{p}_ret_{n}d"] = lc.diff(n)
    out[f"{p}_gap"] = np.log(o / c.shift(1))
    out[f"{p}_oc"] = np.log(c / o)
    out[f"{p}_hl"] = np.log(h / l)

    # volatility
    for n in (5, 10, 20):
        out[f"{p}_rv{n}"] = r1.rolling(n).std()
    out[f"{p}_pk20"] = np.sqrt((np.log(h / l) ** 2).rolling(20).mean() / (4 * np.log(2)))
    out[f"{p}_rv_ratio"] = out[f"{p}_rv5"] / out[f"{p}_rv20"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    out[f"{p}_atr14_pct"] = tr.rolling(14).mean() / c
    out[f"{p}_ret_z"] = r1 / out[f"{p}_rv20"]

    # trend / mean reversion
    smas = {n: c.rolling(n).mean() for n in (5, 10, 20, 50, 200)}
    for n, s in smas.items():
        out[f"{p}_dist_sma{n}"] = c / s - 1
    out[f"{p}_sma20_50"] = smas[20] / smas[50] - 1
    out[f"{p}_sma50_200"] = smas[50] / smas[200] - 1
    out[f"{p}_drawdown60"] = c / c.rolling(60).max() - 1
    out[f"{p}_streak"] = _streak(r1)

    # oscillators
    out[f"{p}_rsi14"] = rsi(c, 14)
    out[f"{p}_rsi5"] = rsi(c, 5)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    out[f"{p}_macd"] = macd / c
    out[f"{p}_macd_hist"] = (macd - sig) / c
    m20, s20 = smas[20], c.rolling(20).std()
    out[f"{p}_bb_pctb"] = (c - (m20 - 2 * s20)) / (4 * s20).replace(0, np.nan)
    out[f"{p}_bb_width"] = 4 * s20 / m20
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    out[f"{p}_stoch_k"] = 100 * (c - lo14) / (hi14 - lo14).replace(0, np.nan)

    # volume (Yahoo index volume is sometimes 0 -> treat as missing)
    if f"{p}_volume" in df.columns:
        v = df[f"{p}_volume"].replace(0, np.nan)
        out[f"{p}_vol_z20"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
        out[f"{p}_vol_chg"] = np.log(v / v.shift(1))

    return pd.DataFrame(out, index=df.index)


def is_settle_day(dates) -> np.ndarray:
    """Approx. HSI futures settlement: 2nd-last business day of the month.
    Ignores HK public holidays, so it can be off by one in some months."""
    idx = pd.DatetimeIndex(dates)
    month_end = idx + pd.offsets.BMonthEnd(0)
    return np.asarray(idx == (month_end - pd.offsets.BDay(1))).astype(int)


def calendar_features(index: pd.DatetimeIndex, next_dates: pd.Series) -> pd.DataFrame:
    """Calendar flags for today (T) and for the target day (T')."""
    nd = pd.DatetimeIndex(next_dates)
    return pd.DataFrame(
        {
            "cal_dow_next": nd.dayofweek,
            "cal_month_next": nd.month,
            "cal_gap_days": (nd - index).days,           # >1 weekend / holiday
            "cal_settle_today": is_settle_day(index),
            "cal_settle_next": is_settle_day(nd),
            "cal_month_end_next": (nd == (nd + pd.offsets.BMonthEnd(0))).astype(int),
        },
        index=index,
    )
