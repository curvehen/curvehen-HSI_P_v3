"""Forward tracking: log each live prediction, settle it once the next HK
trading day is known, and score the model vs the LLM-adjusted call.

Rows are keyed by feature_date. The actual outcome is the NEXT row in the
features table, so HK holidays are handled automatically.
"""
import argparse
import json
import logging

import numpy as np
import pandas as pd

from src import config
from src.features import builder

log = logging.getLogger(__name__)

COLS = [
    "feature_date", "target_date", "logged_at", "ref_hsi_close", "model_p_up",
    "model_direction", "llm_verdict", "llm_p_up", "final_direction",
    "direction_changed", "actual_date", "actual_close", "actual_ret",
    "actual_dir", "model_correct", "final_correct",
]
NUM = ["ref_hsi_close", "model_p_up", "llm_p_up", "direction_changed",
       "actual_close", "actual_ret", "actual_dir", "model_correct", "final_correct"]
MIN_SAMPLE = 100  # below this, any hit rate is statistical noise


def load_log() -> pd.DataFrame:
    if not config.TRACK_PATH.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(config.TRACK_PATH).reindex(columns=COLS)
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _hit(direction: str, up: bool) -> float:
    if direction == "UP":
        return float(up)
    if direction == "DOWN":
        return float(not up)
    return np.nan  # NEUTRAL is not scored


def log_prediction(live: dict, llm: dict) -> pd.DataFrame:
    p = float(live["p_up_ensemble"])
    if llm and abs(float(llm.get("model_p_up", -1)) - p) > 1e-3:
        log.warning("llm_review.json does not match live_prediction.json: ignoring LLM fields.")
        llm = {}
    row = {
        "feature_date": live["feature_date"],
        "target_date": live["target_date"],
        "logged_at": live["generated_at"],
        "ref_hsi_close": live["ref_hsi_close"],
        "model_p_up": p,
        "model_direction": live["direction"],
        "llm_verdict": llm.get("verdict", "UNAVAILABLE"),
        "llm_p_up": llm.get("adjusted_p_up", p),
        "final_direction": llm.get("final_direction", live["direction"]),
        "direction_changed": int(bool(llm.get("direction_changed", False))),
    }
    df = load_log()
    old = df[df["feature_date"] == row["feature_date"]]
    if old["actual_dir"].notna().any():
        return df  # never overwrite a settled prediction
    df = df[df["feature_date"] != row["feature_date"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True) if len(df) else pd.DataFrame([row])
    df = df.reindex(columns=COLS).sort_values("feature_date").reset_index(drop=True)
    df.to_csv(config.TRACK_PATH, index=False)
    return df


def settle(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    idx, close = feats.index, feats["ref_hsi_close"]
    n = 0
    for i in df.index[df["actual_dir"].isna()]:
        fd = pd.Timestamp(df.at[i, "feature_date"])
        if fd not in idx:
            continue
        pos = idx.get_loc(fd)
        if pos + 1 >= len(idx):
            continue  # outcome not available yet
        ret = float(np.log(close.iloc[pos + 1] / close.iloc[pos]))
        up = ret > 0
        df.at[i, "actual_date"] = str(idx[pos + 1].date())
        df.at[i, "actual_close"] = float(close.iloc[pos + 1])
        df.at[i, "actual_ret"] = ret
        df.at[i, "actual_dir"] = float(up)
        df.at[i, "model_correct"] = _hit(df.at[i, "model_direction"], up)
        df.at[i, "final_correct"] = _hit(df.at[i, "final_direction"], up)
        n += 1
    if n:
        df.to_csv(config.TRACK_PATH, index=False)
    log.info("Settled %d prediction(s).", n)
    return df


def _wilson(k: float, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - a) / d, (c + a) / d)


def _rate(s: pd.Series) -> dict:
    s = s.dropna()
    n = len(s)
    lo, hi = _wilson(float(s.sum()), n)
    f = lambda v: round(float(v), 4) if np.isfinite(v) else None
    return {"n": n, "hit_rate": f(s.mean()) if n else None, "ci95_low": f(lo), "ci95_high": f(hi)}


def scorecard(df: pd.DataFrame) -> dict:
    st = df[df["actual_dir"].notna()]
    out = {"n_logged": int(len(df)), "n_settled": int(len(st))}
    if st.empty:
        out["note"] = "No settled predictions yet."
        return out
    out["model"] = _rate(st["model_correct"])
    out["final_after_llm"] = _rate(st["final_correct"])

    ch = st[st["direction_changed"] == 1]
    out["llm_changed_days"] = {
        "n_changed": int(len(ch)),
        "model_hit": _rate(ch["model_correct"]),
        "final_hit": _rate(ch["final_correct"]),
    }
    y = st["actual_dir"]
    llm_ok = st[st["llm_verdict"] != "UNAVAILABLE"]
    out["brier"] = {
        "model": round(float(((st["model_p_up"] - y) ** 2).mean()), 5),
        "llm_adjusted": round(float(((llm_ok["llm_p_up"] - llm_ok["actual_dir"]) ** 2).mean()), 5)
        if len(llm_ok) else None,
        "coin_flip": 0.25,
    }
    out["verdict_counts"] = {k: int(v) for k, v in st["llm_verdict"].value_counts().items()}
    out["sample_warning"] = (
        f"Only {len(st)} settled rows (< {MIN_SAMPLE}): differences are not statistically meaningful."
        if len(st) < MIN_SAMPLE else ""
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Log, settle and score predictions")
    ap.add_argument("--settle-only", action="store_true")
    args = ap.parse_args()

    df = load_log()
    if not args.settle_only:
        if not config.LIVE_PATH.exists():
            raise SystemExit("Run the pipeline in live mode first.")
        live = json.loads(config.LIVE_PATH.read_text())
        llm = json.loads(config.LLM_PATH.read_text()) if config.LLM_PATH.exists() else {}
        df = log_prediction(live, llm)
    df = settle(df, builder.load_features())
    sc = scorecard(df)
    config.SCORE_PATH.write_text(json.dumps(sc, indent=2, ensure_ascii=False))
    print(json.dumps(sc, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
