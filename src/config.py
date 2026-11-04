"""Global configuration. Secrets come from environment variables only.
Order matters: target -> paths -> secrets -> LLM -> assets -> features -> models -> tuning -> pooled."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ================================================================ target
TARGET_NAME = os.getenv("TARGET_NAME") or "hsi"
TARGET_TICKER = os.getenv("TARGET_TICKER") or "^HSI"
TARGET_LABEL = os.getenv("TARGET_LABEL") or TARGET_NAME.upper()

HSI_TARGET = (os.getenv("HSI_TARGET") or "oc").lower()   # "oc" open-to-close, "cc" close-to-close
if HSI_TARGET not in ("oc", "cc"):
    raise ValueError("HSI_TARGET must be 'oc' or 'cc'")
HSI_FLAT = HSI_TARGET == "oc"
HSI_RET_COL, HSI_DIR_COL = (("target_oc_ret", "target_oc_dir") if HSI_FLAT
                            else ("target_ret", "target_dir"))
HSI_HORIZON = ("open(target_date) -> close(target_date)" if HSI_FLAT
               else "close(feature_date) -> close(target_date)")

# ================================================================ paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed" / TARGET_NAME
REPORT_DIR = ROOT / "reports" / TARGET_NAME
CHART_DIR = REPORT_DIR
MODEL_DIR = ROOT / "models_store"
STATE_DIR = ROOT / "state" / TARGET_NAME
MULTI_DATA = ROOT / "data" / "processed" / "multi"
MULTI_STATE = ROOT / "state" / "multi"
MULTI_REPORT_DIR = ROOT / "reports" / "multi"
for _p in (DATA_RAW, DATA_PROC, REPORT_DIR, MODEL_DIR, STATE_DIR,
           MULTI_DATA, MULTI_STATE, MULTI_REPORT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

START_DATE = "2012-01-01"

# ================================================================ secrets
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SMTP_HOST = os.getenv("SMTP_HOST") or "smtp.gmail.com"
SMTP_PORT = int(os.getenv("SMTP_PORT") or "465")
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
EMAIL_TO = os.getenv("EMAIL_TO", "")

# ================================================================ LLM
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "openai").lower()   # "openai" or "deepseek"
LLM_MODEL = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"           # placeholder
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-pro"
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
DEEPSEEK_THINKING = (os.getenv("DEEPSEEK_THINKING") or "1") == "1"
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT") or "180")
LLM_MAX_ADJ = 0.05                                               # HSI: max |adjusted_p - p|
LLM_MAX_ADJ_STOCK = float(os.getenv("LLM_MAX_ADJ_STOCK") or "0.01")

# ================================================================ data assets
# name -> (yahoo ticker, keep_volume)
HK_ASSETS = {
    "hstech": ("^HSTECH", False), "hscei": ("^HSCE", False),
    "etf2800": ("2800.HK", True), "tencent": ("0700.HK", True),
    "hsbc_hk": ("0005.HK", True), "alibaba_hk": ("9988.HK", True),
    "meituan": ("3690.HK", True), "aia": ("1299.HK", True),
    "ccb": ("0939.HK", True), "xiaomi": ("1810.HK", True),
}
# single-stock target: it must not also be a feature; HSI becomes a feature
HK_ASSETS.pop(TARGET_NAME, None)
if TARGET_NAME != "hsi":
    HK_ASSETS["hsi"] = ("^HSI", True)

ASIA_ASSETS = {
    "nikkei": ("^N225", False), "kospi": ("^KS11", False),
    "shanghai": ("000001.SS", False), "usdcnh": ("CNH=X", False),
    "usdhkd": ("HKD=X", False),
}

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

FRED_SERIES = {
    "us10y": "DGS10", "t10y2y": "T10Y2Y", "hy_spread": "BAMLH0A0HYM2",
    "fed_funds": "DFF", "vix_fred": "VIXCLS", "vxn_fred": "VXNCLS",
    "ovx": "OVXCLS", "wti_fred": "DCOILWTICO", "usdhkd_fred": "DEXHKUS",
    "usdcny_fred": "DEXCHUS", "usdjpy_fred": "DEXJPUS", "usd_broad": "DTWEXBGS",
}
FRED_LAG_OBS = 1        # conservative publication lag (observations)
MAX_STALE_DAYS = 5      # max calendar days a stale value is carried forward

# ================================================================ features
# ADR -> (HK line name in panel, HK shares per 1 ADR). Ratios are UNVERIFIED.
ADR_PAIRS = {
    "baba": ("alibaba_hk", 8.0),
    "tcehy": ("tencent", 1.0),
    "hsbc_adr": ("hsbc_hk", 5.0),
}
FX_FALLBACK_HKD = 7.8
CANDLE_ASSETS = ["hsi", "hstech", "spx", "nasdaq", "nq_fut"]
CANDLE_ASSETS = list(dict.fromkeys([TARGET_NAME] + CANDLE_ASSETS))
COMPOSITES = {
    "us_index": ["spx", "nasdaq", "dow", "russell", "es_fut", "nq_fut"],
    "cn_etf": ["fxi", "kweb", "mchi", "ashr", "ewh"],
    "cn_adr": ["baba", "jd", "bidu", "pdd", "ntes", "tcom", "li", "xpev"],
    "hk_bluechip": ["tencent", "hsbc_hk", "alibaba_hk", "meituan", "aia", "ccb", "xiaomi"],
}
MAX_MISSING_FRAC = 0.5
WARMUP_ROWS = 200
RECENT_ROWS = 750
FEATURES_PATH = DATA_PROC / "features.parquet"

# ================================================================ single-target models
SEED = 42
SEQ_LEN = 20
WF_MIN_TRAIN = 750
WF_STEP = 63
WF_MAX_TRAIN = 2000          # None = expanding
VAL_FRAC = 0.15
EPOCHS = 40
BATCH_SIZE = 64
PATIENCE = 6
LR = 1e-3
WEIGHT_DECAY = 1e-3

# data/processed (rebuilt every run)
LIVE_PATH = DATA_PROC / "live_prediction.json"
LLM_PATH = DATA_PROC / "llm_review.json"

# state/<target> (committed), suffix = HSI_TARGET so oc and cc never overwrite each other
_m = HSI_TARGET
OOS_PATH = STATE_DIR / f"oos_predictions_{_m}.parquet"
METRICS_PATH = STATE_DIR / f"oos_metrics_{_m}.csv"
OOS_TUNED_PATH = STATE_DIR / f"oos_predictions_tuned_{_m}.parquet"
METRICS_TUNED_PATH = STATE_DIR / f"oos_metrics_tuned_{_m}.csv"
TRACK_PATH = STATE_DIR / f"prediction_log_{_m}.csv"
SCORE_PATH = STATE_DIR / f"scorecard_{_m}.json"
TUNED_PARAMS_PATH = STATE_DIR / f"tuned_params_{_m}.json"
TUNING_LOG_PATH = STATE_DIR / f"tuning_log_{_m}.json"
TUNED_META_PATH = STATE_DIR / f"tuned_meta_{_m}.json"

# ================================================================ tuning
TUNE_TRIALS = 30
TUNE_TIMEOUT = 120           # seconds per study
TUNE_EVERY_FOLDS = 4
TUNE_METRIC = "logloss"      # "logloss" (recommended) or "auc"
TUNE_MODELS = ("lgbm", "xgb")
TUNE_STALE_DAYS = 120

# ================================================================ pooled multi-stock
# Tickers are UNVERIFIED. A ticker that fails to download is skipped with a warning.
UNIVERSE = {
    "tencent": "0700.HK", "hsbc_hk": "0005.HK", "alibaba_hk": "9988.HK",
    "meituan": "3690.HK", "aia": "1299.HK", "ccb": "0939.HK",
    "xiaomi": "1810.HK", "china_mobile": "0941.HK", "hkex": "0388.HK",
    "icbc": "1398.HK", "ping_an": "2318.HK", "cnooc": "0883.HK",
    "byd": "1211.HK", "jd_hk": "9618.HK", "baidu_hk": "9888.HK",
}
MIN_STOCK_ROWS = 500
MIN_CROSS_SECTION = 5
USE_STOCK_ID = True
MULTI_WF_MIN_DATES = 750
MULTI_WF_STEP = 63
MULTI_WF_MAX_DATES = 1500    # None = expanding
MULTI_EPOCHS = 20
MULTI_BATCH = 256
MULTI_TOPK = 3
COST_BPS_STOCK = 15.0        # PLACEHOLDER one-way cost (bps): replace with your real costs

MULTI_TARGET = (os.getenv("MULTI_TARGET") or "oc").lower()   # "oc" tradable, "cc" close-to-close
_COLS = {
    "cc": ("target_ret", "target_xs", "target_xs_dir"),
    "oc": ("target_oc_ret", "target_oc_xs", "target_oc_xs_dir"),
}
if MULTI_TARGET not in _COLS:
    raise ValueError("MULTI_TARGET must be 'oc' or 'cc'")
MULTI_RET, MULTI_XS, MULTI_LABEL = _COLS[MULTI_TARGET]
_t = MULTI_TARGET

# state/multi (committed)
MULTI_OOS_PATH = MULTI_STATE / f"oos_long_{_t}.parquet"
MULTI_METRICS_PATH = MULTI_STATE / f"oos_metrics_long_{_t}.csv"
MULTI_YEAR_PATH = MULTI_STATE / f"oos_by_year_long_{_t}.csv"
MULTI_LS_PATH = MULTI_STATE / f"ls_returns_{_t}.parquet"
MULTI_LIVE_JSON = MULTI_STATE / f"live_ranking_{_t}.json"
MULTI_LIVE_CSV = MULTI_STATE / f"live_ranking_{_t}.csv"
MULTI_RANK_LOG = MULTI_STATE / f"ranking_log_{_t}.csv"
MULTI_SETTLED_PATH = MULTI_STATE / f"ranking_settled_{_t}.csv"
MULTI_DAILY_PATH = MULTI_STATE / f"forward_daily_{_t}.csv"
MULTI_SCORE_PATH = MULTI_STATE / f"forward_scorecard_{_t}.json"
MULTI_TUNED_PARAMS_PATH = MULTI_STATE / f"tuned_params_long_{_t}.json"
MULTI_TUNING_LOG_PATH = MULTI_STATE / f"tuning_log_long_{_t}.json"
MULTI_TUNED_META_PATH = MULTI_STATE / f"tuned_meta_long_{_t}.json"
MULTI_LLM_PATH = MULTI_STATE / f"llm_ranking_{_t}.json"
MULTI_LLM_LOG = MULTI_STATE / f"llm_ranking_log_{_t}.csv"
MULTI_LLM_COMPARE = MULTI_STATE / f"llm_compare_{_t}.json"


# ================================================================ helpers
def tuned_path(p: Path) -> Path:
    """oos_long_oc.parquet -> oos_long_oc_tuned.parquet"""
    return p.with_name(p.stem + "_tuned" + p.suffix)


def _live_tuned(path: Path) -> bool:
    try:
        return bool(json.loads(path.read_text()).get("tuned_models"))
    except Exception:
        return False


def active_hsi_paths():
    """(oos_path, metrics_path) matching the models used by the latest HSI live run."""
    if _live_tuned(LIVE_PATH) and OOS_TUNED_PATH.exists() and METRICS_TUNED_PATH.exists():
        return OOS_TUNED_PATH, METRICS_TUNED_PATH
    return OOS_PATH, METRICS_PATH


def active_multi_metrics() -> Path:
    """Pooled metrics file matching the latest live ranking (tuned if it used tuned parameters)."""
    t = tuned_path(MULTI_METRICS_PATH)
    if _live_tuned(MULTI_LIVE_JSON) and t.exists():
        return t
    return MULTI_METRICS_PATH
