"""Global configuration. Secrets are read from environment variables only."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed" / TARGET_NAME
REPORT_DIR = ROOT / "reports" / TARGET_NAME
MODEL_DIR = ROOT / "models_store"
for _p in (DATA_RAW, DATA_PROC, REPORT_DIR, MODEL_DIR):
    _p.mkdir(parents=True, exist_ok=True)

START_DATE = "2012-01-01"

TARGET_NAME = os.getenv("TARGET_NAME") or "hsi"
TARGET_TICKER = os.getenv("TARGET_TICKER") or "^HSI"
TARGET_LABEL = os.getenv("TARGET_LABEL") or TARGET_NAME.upper()

# ---- Secrets (set in .env locally, GitHub Secrets in CI) ----
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

SMTP_HOST = os.getenv("SMTP_HOST") or "smtp.gmail.com"
SMTP_PORT = int(os.getenv("SMTP_PORT") or "465")
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

# ---- Part 2: features ----
# ADR -> (HK line name in panel, HK shares per 1 ADR). Ratios are UNVERIFIED: check before use.
ADR_PAIRS = {
    "baba": ("alibaba_hk", 8.0),
    "tcehy": ("tencent", 1.0),
    "hsbc_adr": ("hsbc_hk", 5.0),
}
FX_FALLBACK_HKD = 7.8
# Assets on which candle patterns are detected (column prefix in panel)
CANDLE_ASSETS = ["hsi", "hstech", "spx", "nasdaq", "nq_fut"]
# Composite groups (mean of 1-day log returns)
COMPOSITES = {
    "us_index": ["spx", "nasdaq", "dow", "russell", "es_fut", "nq_fut"],
    "cn_etf": ["fxi", "kweb", "mchi", "ashr", "ewh"],
    "cn_adr": ["baba", "jd", "bidu", "pdd", "ntes", "tcom", "li", "xpev"],
    "hk_bluechip": ["tencent", "hsbc_hk", "alibaba_hk", "meituan", "aia", "ccb", "xiaomi"],
}
MAX_MISSING_FRAC = 0.5
WARMUP_ROWS = 200
FEATURES_PATH = DATA_PROC / "features.parquet"

# ---- Part 3: models ----
SEED = 42
SEQ_LEN = 20            # lookback window for LSTM / GRU / Transformer
WF_MIN_TRAIN = 750      # first training window (rows)
WF_STEP = 63            # retrain every 63 trading days in the backtest
WF_MAX_TRAIN = 2000     # rolling window cap (rows); None = expanding
VAL_FRAC = 0.15         # last 15% of each training window = early-stopping set
EPOCHS = 40
BATCH_SIZE = 64
PATIENCE = 6
OOS_PATH = DATA_PROC / "oos_predictions.parquet"
LIVE_PATH = DATA_PROC / "live_prediction.json"

LR = 1e-3
WEIGHT_DECAY = 1e-3

# ---- Part 4: LLM ----
LLM_MODEL = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"  # placeholder: set to a model you have access to
LLM_MAX_ADJ = 0.05      # max |adjusted_p_up - model p_up|
LLM_PATH = DATA_PROC / "llm_review.json"
TRACK_PATH = DATA_PROC / "prediction_log.csv"
METRICS_PATH = DATA_PROC / "oos_metrics.csv"

SCORE_PATH = DATA_PROC / "scorecard.json"
CHART_DIR = REPORT_DIR

# ---- Part 5: persistent state (committed to the repo) ----
STATE_DIR = ROOT / "state" / TARGET_NAME
STATE_DIR.mkdir(parents=True, exist_ok=True)
OOS_PATH = STATE_DIR / "oos_predictions.parquet"
METRICS_PATH = STATE_DIR / "oos_metrics.csv"
TRACK_PATH = STATE_DIR / "prediction_log.csv"
SCORE_PATH = STATE_DIR / "scorecard.json"

HK_ASSETS.pop(TARGET_NAME, None)
if TARGET_NAME != "hsi":
    HK_ASSETS["hsi"] = ("^HSI", True)   # HSI 變成 feature
CANDLE_ASSETS = list(dict.fromkeys([TARGET_NAME] + CANDLE_ASSETS))

# ---- B3: pooled multi-stock model ----
# Tickers are UNVERIFIED. Any ticker that fails to download is skipped with a log warning.
UNIVERSE = {
    "tencent": "0700.HK", "hsbc_hk": "0005.HK", "alibaba_hk": "9988.HK",
    "meituan": "3690.HK", "aia": "1299.HK", "ccb": "0939.HK",
    "xiaomi": "1810.HK", "china_mobile": "0941.HK", "hkex": "0388.HK",
    "icbc": "1398.HK", "ping_an": "2318.HK", "cnooc": "0883.HK",
    "byd": "1211.HK", "jd_hk": "9618.HK", "baidu_hk": "9888.HK",
}
MIN_STOCK_ROWS = 500       # skip stocks with fewer labelled rows
MIN_CROSS_SECTION = 5      # min stocks per date to define target_xs
MULTI_DATA = ROOT / "data" / "processed" / "multi"
MULTI_STATE = ROOT / "state" / "multi"
MULTI_DATA.mkdir(parents=True, exist_ok=True)
MULTI_STATE.mkdir(parents=True, exist_ok=True)
