"""L2: LLM challenge review of the pooled multi-stock ranking."""
import argparse
import json
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from src import config
from src.llm import critic
from src.multi import features_long as fl

log = logging.getLogger(__name__)
ACTIONS = {"KEEP", "UPGRADE", "DOWNGRADE", "FLAG"}
STOCK_FEATS = [
    "s_ret1_lag", "s_ret_5d", "s_ret_20d", "s_rv20", "s_rsi14", "s_dist_sma50",
    "s_dist_sma200", "s_drawdown60", "s_bb_pctb", "s_streak", "s_beta60",
    "s_turn_z60", "rel_ret1", "rel_ret5", "rel_ret20", "cs_rank_s_ret_5d",
    "s_cdl_net_score",
]
MKT_FEATS = [
    "m_spx_ret1", "m_nasdaq_ret1", "m_nq_fut_ret1", "m_sox_ret1", "m_kweb_ret1",
    "m_fxi_ret1", "m_vix_lvl", "m_vix_ret1", "m_comp_cn_adr_ret1",
    "m_comp_cn_etf_ret1", "m_comp_hk_bluechip_ret1", "m_usdcnh_ret1",
    "m_fred_us10y_d1", "m_cal_gap_days", "uni_ret1_mean", "uni_breadth", "uni_disp",
]

SYSTEM = """You are a sceptical senior risk reviewer for a pooled multi-stock Hong Kong \
equity ranking model. You receive ONLY the data below.
Model facts: each stock's score is the mean probability that it beats the same-day median \
return of the universe (score 0.5 = no view). The top-k are LONG and the bottom-k SHORT.
Rules:
1. Use ONLY the supplied numbers. Do not cite news, earnings, events or prices from memory. \
Anything that would matter but is not supplied (earnings dates, ex-dividend dates, liquidity, \
borrow availability) goes in risk_flags as "missing data".
2. Challenge the ranking: look for stocks whose score contradicts their own features or the \
market snapshot (overextended RSI, large recent move, high beta against a weak overnight tape), \
low model agreement, many missing live features, or weak out-of-sample stats \
(ic_t below about 2, net Sharpe not above the random baseline).
3. If out-of-sample evidence is weak, say so plainly in the summary.
4. For EVERY stock return an entry with action KEEP, UPGRADE, DOWNGRADE or FLAG, an adjusted_score \
within 0.01 of its model score, and a reason of at most 25 words.
Return ONE JSON object with exactly these keys:
stocks: list of {"name", "action", "adjusted_score", "reason"}
market_view: max 40 words
risk_flags: list of up to 5 short strings
summary: max 60 words
Example of the required JSON shape (placeholders only, not a prediction):
{"stocks": [{"name": "xxx", "action": "KEEP", "adjusted_score": 0.5, "reason": "..."}], "market_view": "...", "risk_flags": ["..."], "summary": "..."}"""


def _v(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, 5) if np.isfinite(x) else None


def _json(path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception as exc:
        log.warning("Cannot read %s: %s", path, exc)
        return {}


def _oos_stats() -> dict:
    mpath = config.active_multi_metrics()
    if not mpath.exists():
        return {}
    m = pd.read_csv(mpath, index_col=0)

    out = {}
    for row, cols in (("ens_equal", ["auc", "ic_mean", "ic_t", "ic_pos", "ls_net_sharpe"]),
                      ("baseline_random_p95", ["ls_net_sharpe"]),
                      ("baseline_universe_long", ["ls_net_sharpe"])):
        if row in m.index:
            out[row] = {c: _v(m.loc[row, c]) for c in cols if c in m.columns}
    return out


def build_context(live: dict, df: pd.DataFrame) -> str:
    lv = df[df["is_live"] == 1].set_index("name")
    market = {}
    if len(lv):
        first = lv.iloc[0]
        market = {c: _v(first[c]) for c in MKT_FEATS if c in lv.columns}
    stocks = []
    for r in live["ranking"]:
        n = r["name"]
        feats = ({c: _v(lv.at[n, c]) for c in STOCK_FEATS if c in lv.columns}
                 if n in lv.index else {})
        stocks.append({
            "name": n, "model_rank": r["rank"], "model_score": r["score"],
            "model_side": r["side"], "agreement": r["agreement"],
            "model_probs": {k[2:]: r[k] for k in r if k.startswith("p_")},
            "live_nan": r["live_nan"], "features": feats,
        })
    sc = _json(config.MULTI_SCORE_PATH)
    ctx = {
        "feature_date": live["feature_date"], "target_date": live["target_date"],
        "horizon": live["horizon"], "top_k": live["top_k"],
        "score_spread_std": _v(np.std([r["score"] for r in live["ranking"]])),
        "market_snapshot": market,
        "stocks": stocks,
        "backtest_out_of_sample": _oos_stats(),
        "forward_scorecard": {k: sc.get(k) for k in
                              ("n_days_settled", "ic_mean", "ic_t", "net_sharpe") if k in sc},
        "units": "*_ret* are log returns (0.01 = 1%); *_d1 are level changes.",
    }
    return json.dumps(ctx, ensure_ascii=False, indent=1)


def _assign(scores: pd.Series, k: int):
    order = scores.sort_values(ascending=False, kind="stable")
    n = len(order)
    side = pd.Series("FLAT", index=order.index)
    if n >= max(2 * k, config.MIN_CROSS_SECTION):
        side.iloc[:k] = "LONG"
        side.iloc[n - k:] = "SHORT"
    return order, side


def _finalize(live: dict, raw: dict, verdict: str, note: str = "") -> dict:
    cap, k = config.LLM_MAX_ADJ_STOCK, int(live["top_k"])
    items: Dict[str, dict] = {}
    if isinstance(raw.get("stocks"), list):
        items = {str(s["name"]): s for s in raw["stocks"] if isinstance(s, dict) and "name" in s}

    rk = live["ranking"]
    model = pd.Series({r["name"]: float(r["score"]) for r in rk})
    adj = model.copy()
    meta = {}
    for n, p in model.items():
        s = items.get(n, {})
        try:
            a = float(s.get("adjusted_score", p))
        except (TypeError, ValueError):
            a = p
        if not np.isfinite(a):
            a = p
        adj[n] = min(max(a, p - cap, 0.0), p + cap, 1.0)     # cap enforced in code
        act = str(s.get("action", "KEEP")).upper()
        meta[n] = (act if act in ACTIONS else "KEEP", str(s.get("reason", ""))[:240])

    order, side_f = _assign(adj, k)
    rows = []
    for i, (n, fs) in enumerate(order.items(), 1):
        r = next(x for x in rk if x["name"] == n)
        rows.append({
            "name": n, "model_rank": r["rank"], "model_side": r["side"],
            "model_score": round(float(model[n]), 6), "final_score": round(float(fs), 6),
            "final_rank": i, "final_side": side_f[n],
            "side_changed": side_f[n] != r["side"],
            "action": meta[n][0], "reason": meta[n][1],
        })
    return {
        "reviewed_at": pd.Timestamp.now(tz="Asia/Hong_Kong").isoformat(timespec="seconds"),
        "model": critic._provider()[1],
        "feature_date": live["feature_date"], "target_date": live["target_date"],
        "verdict": verdict, "cap": cap,
        "n_side_changed": int(sum(r["side_changed"] for r in rows)),
        "market_view": str(raw.get("market_view", ""))[:400],
        "risk_flags": [str(x)[:240] for x in raw.get("risk_flags", [])[:5]]
        if isinstance(raw.get("risk_flags"), list) else [],
        "summary": str(raw.get("summary", note))[:500],
        "stocks": rows,
    }


def review(live: dict, df: pd.DataFrame) -> dict:
    key, _, base_url, _ = critic._provider()
    if not key:
        log.warning("No API key for provider %s: stock review skipped.", config.LLM_PROVIDER)
        return _finalize(live, {}, "UNAVAILABLE", "LLM review skipped: no API key.")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
        messages = [{"role": "system", "content": SYSTEM},
                    
