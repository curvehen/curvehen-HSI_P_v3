"""Raw data loaders with incremental parquet caching."""
import io
import logging
import time
from datetime import timedelta
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

from src import config

log = logging.getLogger(__name__)


def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    return df[~df.index.duplicated(keep="last")].sort_index()


def _download_yf(ticker: str, start: str, retries: int = 3) -> pd.DataFrame:
    for i in range(retries):
        try:
            df = yf.download(
                ticker, start=start, auto_adjust=True,
                progress=False, threads=False,
            )
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [str(c).lower() for c in df.columns]
                return _clean_index(df)
        except Exception as exc:  # network / yahoo errors
            log.warning("yfinance %s attempt %d failed: %s", ticker, i + 1, exc)
        time.sleep(2 * (i + 1))
    return pd.DataFrame()


def load_yf(ticker: str, refresh: bool = True) -> pd.DataFrame:
    """Load OHLCV for a Yahoo ticker; incrementally update the local cache."""
    safe = ticker.replace("^", "").replace("=", "_").replace("/", "_")
    path = config.DATA_RAW / f"yf_{safe}.parquet"
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame()

    if refresh or cached.empty:
        start = config.START_DATE
        if not cached.empty:
            start = (cached.index.max() - timedelta(days=7)).strftime("%Y-%m-%d")
        new = _download_yf(ticker, start)
        if not new.empty:
            cached = pd.concat([cached, new])
            cached = cached[~cached.index.duplicated(keep="last")].sort_index()
            cached.to_parquet(path)
    if cached.empty:
        log.warning("No data for %s - skipped", ticker)
    return cached


def load_fred(series_id: str, refresh: bool = True) -> pd.Series:
    """Load a FRED series. Uses fredapi with a key, else the public CSV."""
    path = config.DATA_RAW / f"fred_{series_id}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)["value"]

    s: Optional[pd.Series] = None
    try:
        if config.FRED_API_KEY:
            from fredapi import Fred
            s = Fred(api_key=config.FRED_API_KEY).get_series(
                series_id, observation_start=config.START_DATE
            )
        else:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            raw = pd.read_csv(io.StringIO(r.text))
            raw.iloc[:, 0] = pd.to_datetime(raw.iloc[:, 0])
            s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
            s.index = raw.iloc[:, 0]
            s = s[s.index >= config.START_DATE]
    except Exception as exc:
        log.warning("FRED %s failed: %s", series_id, exc)

    if s is None or s.empty:
        return pd.read_parquet(path)["value"] if path.exists() else pd.Series(dtype=float)

    s = s.astype(float).dropna()
    s.index = pd.to_datetime(s.index).normalize()
    s.index.name = "date"
    s.to_frame("value").to_parquet(path)
    return s


def load_southbound(refresh: bool = True) -> pd.Series:
    """Optional: Southbound net buy via akshare. Column names are unverified
    and may change upstream, so failures degrade gracefully to an empty series."""
    path = config.DATA_RAW / "ak_southbound.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)["value"]
    try:
        import akshare as ak
        df = ak.stock_hsgt_hist_em(symbol="南向资金")
        date_col = "日期"
        val_col = next((c for c in df.columns if "净买额" in c and "当日" in c), None)
        if val_col is None or date_col not in df.columns:
            raise ValueError(f"unexpected columns: {list(df.columns)}")
        s = pd.Series(
            pd.to_numeric(df[val_col], errors="coerce").values,
            index=pd.to_datetime(df[date_col]).dt.normalize(), name="value",
        ).dropna()
        s.index.name = "date"
        s.to_frame("value").to_parquet(path)
        return s
    except Exception as exc:
        log.warning("Southbound (akshare) failed: %s", exc)
        if path.exists():
            return pd.read_parquet(path)["value"]
        return pd.Series(dtype=float)
