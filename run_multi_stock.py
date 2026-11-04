"""Daily pooled multi-stock run:
data -> long features -> live ranking -> LLM review -> settle earlier days -> report -> email.

Usage:
  python run_multi_stock.py
  python run_multi_stock.py --no-gru --no-email
  python run_multi_stock.py --no-llm
  python run_multi_stock.py --models lgbm xgb --k 2
TARGET_NAME / TARGET_TICKER must be unset (the HSI panel is the trading calendar).
"""
import argparse
import json
import logging
import sys

from src import config
from src.multi import features_long as fl
from src.multi import live, llm_review, llm_track, report_multi, settle

log = logging.getLogger("run_multi_stock")


def main() -> int:
    ap = argparse.ArgumentParser(description="Pooled multi-stock daily run")
    ap.add_argument("--models", nargs="*", help="subset: logit lgbm xgb gru")
    ap.add_argument("--k", type=int, default=config.MULTI_TOPK)
    ap.add_argument("--no-gru", action="store_true")
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM review")
    ap.add_argument("--seed", type=int, default=config.SEED)
    a = ap.parse_args()

    if config.TARGET_NAME != "hsi":
        log.error("Unset TARGET_NAME / TARGET_TICKER before running the pooled model.")
        return 2

    # core steps: errors stop the run
    df = fl.build_long_features(refresh=True)
    r, meta = live.build_ranking(df, a.models, not a.no_gru, a.k, a.seed)
    live.save(r, meta)

    # LLM review: never blocks the run
    if not a.no_llm:
        try:
            live_json = json.loads(config.MULTI_LIVE_JSON.read_text())
            out = llm_review.review(live_json, df)
            config.MULTI_LLM_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
            llm_track.log_review(live_json, out)
            log.info("LLM review: %s, %d side change(s)", out["verdict"], out["n_side_changed"])
        except Exception as exc:
            log.warning("LLM review failed: %s: %s", type(exc).__name__, exc)

    # settle earlier days: a failure here must not block the report
    try:
        sc = settle.run()
        log.info("Forward scorecard: %s settled day(s)", sc.get("n_days_settled", 0))
        llm_track.compare()
    except Exception as exc:
        log.warning("Settlement failed: %s: %s", type(exc).__name__, exc)

    # report is a core step; email failure is only logged
    rep = report_multi.build_report()
    if a.no_email:
        log.info("Email skipped (--no-email). Report: %s", rep["path"])
    else:
        report_multi.send(rep)

    print(rep["text"])
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sys.exit(main())
