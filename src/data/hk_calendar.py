"""HK trading calendar: weekdays minus public holidays (python-holidays).
Falls back to plain weekdays, with a warning, if the package is missing.
Typhoon and black-rainstorm closures cannot be known in advance."""
import logging
from functools import lru_cache

import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _holidays() -> tuple:
    try:
        import holidays
        h = holidays.country_holidays("HK", years=range(2010, 2036))
        return tuple(sorted(pd.Timestamp(d) for d in h.keys()))
    except Exception as exc:
        log.warning("HK holidays unavailable (%s): weekdays only.", exc)
        return ()


def hk_bday() -> CustomBusinessDay:
    return CustomBusinessDay(holidays=list(_holidays()))


def next_trading_day(d) -> pd.Timestamp:
    return pd.Timestamp(d) + hk_bday()


def last_trading_day_of_month(dates) -> pd.DatetimeIndex:
    """Last trading day on or before the calendar month end of each date."""
    idx = pd.DatetimeIndex(dates)
    bd = hk_bday()
    return (idx + pd.offsets.MonthEnd(0) + bd) - bd


if __name__ == "__main__":
    # Compare this output with the official HKEX holiday calendar.
    today = pd.Timestamp.now().normalize()
    print("Holidays this year:")
    for h in _holidays():
        if h.year == today.year:
            print(" ", h.date(), h.day_name())
    d, out = today, []
    for _ in range(8):
        d = next_trading_day(d)
        out.append(str(d.date()))
    print("Next 8 trading days:", ", ".join(out))
