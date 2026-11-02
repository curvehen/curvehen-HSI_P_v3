"""Compact, factual context handed to the LLM. Numbers only, no outside facts."""
import json
from typing import Optional

import numpy as np
import pandas as pd

from src import config
from src.features import candles

KEY_FEATURES = [
    # overnight US / China proxies
    "spx_ret1", "nasdaq_ret1", "nq_fut_ret1", "sox_ret1", "kweb_ret1", "fxi_ret1",
    "baba_ret1", "yinn_ret1", "vix_lvl", "vix_ret1",
    "comp_us_index_ret1", "comp_cn_etf_ret1", "comp_cn_adr_ret1",
    "comp_hk_bluechip_ret1", "comp_cn_adr_breadth",
    # rates / credit / FX
    "fred_us10y_d1", "fred_hy_spread_d1", "usdcnh_ret1", "dxy_ret1",
    # flows / ADR premium
    "sb_net", "sb_z20", "adr_baba_prem", "adr_tcehy_prem",
    # HSI state
    f"{_T}_ret1_lag", f"{_T}_ret_5d", f"{_T}_rsi14", f"{_T}_rv20",
    f"{_T}_dist_sma20", f"{_T}_dist_sma200", f"{_T}_drawdown60",
    f"{_T}_bb_pctb", f"{_T}_streak",
    *([] if _T == "hsi" else ["hsi_ret1", "hsi_ret5"]),  # index as context
    # calendar
    "cal_dow_next", "cal_gap_days", "cal_settle_next", "cal_month_end_next",
]


def _v(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, 5) if np.isfinite(x) else None


def _oos_summary() -> dict:
    if not config.METRICS_PATH.exists():
        return {}
    m = pd.read_csv(config.METRICS_PATH, index_col=0)
    keep = ["accuracy", "auc", "logloss", "sharpe", "hit_rate_traded", "n"]
    out = {}
    for name in ("ens_weighted", "ens_equal", "baseline_buyhold"):
        if name in m.index:
            out[name] = {k: _v(m.loc[name, k]) for k in keep if k in m.columns}
    return out


def build_context(live: dict, feats: pd.DataFrame) -> str:
    last = feats.iloc[-1]
    snapshot = {k: _v(last[k]) for k in KEY_FEATURES if k in feats.columns}

    pats = [c for c in candles.pattern_columns(_T) if c in feats.columns]
    patterns = {c: int(last[c]) for c in pats if pd.notna(last[c]) and last[c] != 0}
    net = f"{_T}_cdl_net_score"
    if net in feats.columns:
        patterns[net] = _v(last[net])

    closes = feats["ref_hsi_close"].tail(6)  # holds the target's close
    history = [{"date": str(d.date()), "close": _v(c)} for d, c in closes.items()]

    ctx = {
        "task": "Review the quantitative model's call for the next HK trading day.",
        "units": "*_ret1 / *_ret5 are log returns (0.01 = 1%); *_d1 are level changes.",
        "model_output": {
            k: live.get(k) for k in (
                "feature_date", "target_date", "direction", "p_up_ensemble",
                "model_probs", "weights", "model_agreement", "live_row_nan",
            )
        },
        "target_recent_closes": history,
        "latest_feature_snapshot": snapshot,
        "target_candle_patterns_today": patterns,
        "top_model_features_live": live.get("top_features_live", {}),
        "out_of_sample_walkforward_stats": _oos_summary(),
    }
    return json.dumps(ctx, ensure_ascii=False, indent=1)
