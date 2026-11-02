"""LLM devil's-advocate review of the model's next-day call."""
import json
import logging
from typing import Any, List

import pandas as pd

from src import config
from src.features import builder
from src.llm.context import build_context

log = logging.getLogger(__name__)

NEUTRAL_BAND = 0.02  # keep equal to pipeline.NEUTRAL_BAND
VERDICTS = {"AGREE", "DISAGREE", "UNCERTAIN"}

SYSTEM = """You are a sceptical senior risk reviewer for a Hang Seng Index (HSI) \
next-day direction model. You receive ONLY the data below.
Rules:
1. Use ONLY the supplied numbers. Do not cite news, events, or prices from memory. \
If something would matter but is not supplied, list it under risk_flags as "missing data".
2. Your job is to challenge the model: build the strongest bear case and bull case, \
then find contradictions between the model call and the inputs (for example overnight \
US/China signals pointing the other way, high live_row_nan, low model_agreement, weak \
out-of-sample AUC or Sharpe versus baseline_buyhold).
3. Daily direction is close to a coin flip. If out-of-sample AUC is near 0.5, say so.
4. adjusted_p_up must stay within 0.05 of the model's p_up_ensemble.
Return ONE JSON object with exactly these keys:
verdict: "AGREE" | "DISAGREE" | "UNCERTAIN"
adjusted_p_up: number in [0,1]
bull_case: list of up to 4 short strings
bear_case: list of up to 4 short strings
contradictions: list of up to 4 short strings
risk_flags: list of up to 4 short strings
summary: max 60 words, plain text."""

SYSTEM = SYSTEM.replace("Hang Seng Index (HSI)", config.TARGET_LABEL)


def _direction(p: float) -> str:
    if p >= 0.5 + NEUTRAL_BAND:
        return "UP"
    if p <= 0.5 - NEUTRAL_BAND:
        return "DOWN"
    return "NEUTRAL"


def _strs(x: Any, n: int = 4) -> List[str]:
    if not isinstance(x, list):
        return []
    return [str(i)[:240] for i in x[:n]]


def _result(live: dict, verdict: str, adj: float, raw: dict, note: str = "") -> dict:
    p = float(live["p_up_ensemble"])
    final = _direction(adj)
    return {
        "reviewed_at": pd.Timestamp.now(tz="Asia/Hong_Kong").isoformat(timespec="seconds"),
        "model": config.LLM_MODEL,
        "verdict": verdict,
        "model_p_up": round(p, 4),
        "adjusted_p_up": round(adj, 4),
        "model_direction": live["direction"],
        "final_direction": final,
        "direction_changed": final != live["direction"],
        "bull_case": _strs(raw.get("bull_case")),
        "bear_case": _strs(raw.get("bear_case")),
        "contradictions": _strs(raw.get("contradictions")),
        "risk_flags": _strs(raw.get("risk_flags")),
        "summary": str(raw.get("summary", note))[:500],
    }


def _sanitize(raw: dict, live: dict) -> dict:
    p = float(live["p_up_ensemble"])
    verdict = str(raw.get("verdict", "UNCERTAIN")).upper()
    if verdict not in VERDICTS:
        verdict = "UNCERTAIN"
    try:
        adj = float(raw.get("adjusted_p_up", p))
    except (TypeError, ValueError):
        adj = p
    lo, hi = max(0.0, p - config.LLM_MAX_ADJ), min(1.0, p + config.LLM_MAX_ADJ)
    adj = min(max(adj, lo), hi)  # hard cap enforced in code, not by the prompt
    return _result(live, verdict, adj, raw)


def _call(client, messages: list) -> dict:
    kw = dict(model=config.LLM_MODEL, messages=messages,
              response_format={"type": "json_object"}, timeout=90)
    try:
        resp = client.chat.completions.create(temperature=0.2, **kw)
    except Exception as exc:  # some models reject a custom temperature
        log.info("Retrying without temperature: %s", exc)
        resp = client.chat.completions.create(**kw)
    return json.loads(resp.choices[0].message.content)


def review(live: dict, feats: pd.DataFrame) -> dict:
    p = float(live["p_up_ensemble"])
    if not config.OPENAI_API_KEY:
        log.warning("OPENAI_API_KEY not set: LLM review skipped.")
        return _result(live, "UNAVAILABLE", p, {}, "LLM review skipped: no API key.")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_context(live, feats)},
        ]
        return _sanitize(_call(client, messages), live)
    except Exception as exc:
        log.warning("LLM review failed: %s", exc)
        return _result(live, "UNAVAILABLE", p, {}, f"LLM review failed: {type(exc).__name__}")


def main() -> None:
    if not config.LIVE_PATH.exists():
        raise SystemExit("Run `python -m src.models.pipeline --mode live` first.")
    live = json.loads(config.LIVE_PATH.read_text())
    feats = builder.load_features()
    out = review(live, feats)
    config.LLM_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
