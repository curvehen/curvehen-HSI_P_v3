"""Global configuration. Secrets are read from environment variables only."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models_store"
for _p in (DATA_RAW, DATA_PROC, REPORT_DIR, MODEL_DIR):
    _p.mkdir(parents=True, exist_ok=True)

START_DATE = "2012-01-01"
TARGET_NAME = "hsi"
TARGET_TICKER = "^HSI"

# ---- Secrets (set in .env locally, GitHub Secrets in CI) ----
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
EMAIL_TO = os.getenv("EMAIL_TO", "")

# ---- HK-session assets: same-date alignment ----
# name -> (yahoo ticker, keep_volume)
HK_ASSETS = {
    "hstech": ("^HSTECH", False),
    "hscei": ("^HSCE", False),
    "etf2800": ("2800.HK", True),
    "tencent": ("0700.HK", True),
    "hsbc_hk": ("0005.HK", True),
    "alibaba_hk": ("9988.HK", True),
    "meituan": ("3690.HK", True),
    "aia": ("1299.HK", True),
    "ccb": ("0939.HK", True),
    "xiaomi": ("1810.HK", True),
}

# ---- Asia assets (close before / around HK close) ----
ASIA_ASSETS = {
    "nikkei": ("^N225", False),
    "kospi": ("^KS11", False),
    "shanghai": ("000001.SS", False),
    "usdcnh": ("CNH=X", False),
    "usdhkd": ("HKD=X", False),
}

# ---- US-session assets: US date D maps to HK date D (used to predict D+1) ----
US_ASSETS = {
    "spx": ("^GSPC", False), "nasdaq": ("^IXIC", False),
    "dow": ("^DJI", False), "russell": ("^RUT", False),
    "sox": ("^SOX", False), "vix": ("^VIX", False), "vxn": ("^VXN", False),
    "es_fut": ("ES=F", False), "nq_fut": ("NQ=F", False),
    "fxi": ("FXI", True), "kweb": ("KWEB", True), "mchi": ("MCHI", True),
    "ashr": ("ASHR", True), "ewh": ("EWH", True),
    "yinn": ("YINN", False), "yang": ("YANG", False),
    "baba": ("BABA", True), "jd": ("JD", True), "bidu": ("BIDU", True),
    "pdd": ("PDD", True), "ntes": ("NTES", True), "tcom": ("TCOM", True),
    "li": ("LI", True), "xpev": ("XPEV", True),
    "hsbc_adr": ("HSBC", False), "tcehy": ("TCEHY", False),
    "xlk": ("XLK", False), "xlf": ("XLF", False), "xle": ("XLE", False),
    "hyg": ("HYG", False), "lqd": ("LQD", False), "tlt": ("TLT", False),
    "uup": ("UUP", False), "dxy": ("DX-Y.NYB", False),
    "wti": ("CL=F", False), "gold": ("GC=F", False), "copper": ("HG=F", False),
}

# ---- FRED series ----
FRED_SERIES = {
    "us10y": "DGS10", "t10y2y": "T10Y2Y", "hy_spread": "BAMLH0A0HYM2",
    "fed_funds": "DFF", "vix_fred": "VIXCLS", "vxn_fred": "VXNCLS",
    "ovx": "OVXCLS", "wti_fred": "DCOILWTICO", "usdhkd_fred": "DEXHKUS",
    "usdcny_fred": "DEXCHUS", "usdjpy_fred": "DEXJPUS",
    "usd_broad": "DTWEXBGS",
}
# Conservative publication lag (in observations) applied to FRED data.
FRED_LAG_OBS = 1

# Max calendar days a stale value may be carried forward during alignment.
MAX_STALE_DAYS = 5
