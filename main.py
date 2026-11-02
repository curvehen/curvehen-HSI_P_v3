"""Daily entry point:
data -> features -> models -> LLM review -> forward tracking -> report -> email

Usage:
  python main.py                   # daily run (re-runs backtest only if missing/stale)
  python main.py --backtest        # force a fresh walk-forward backtest
  python main.py --skip-data       # reuse cached features.parquet
  python main.py --no-email        # build the report but do not send it
  python main.py --models logit lgbm xgb
"""
import argparse
import json
import logging
import sys

import pandas as pd

from src import config
from src.data import build_dataset
from src.features import builder
from src.llm import critic, tracker
from src.models import pipeline
from src.models.common import prepare
from src.reporting import emailer, report

log = logging.getLogger("main")


def _load_oos():
    if not config.OOS_PATH.exists():
        return None
    try:
        return pd.read_parquet(config.OOS_PATH)
    except Exception as exc:
        log.warning("Cannot read OOS file (%s): will rebuild.", exc)
        return None


def _backtest_needed(oos, X: pd.DataFrame, n_lab: int) -> bool:
    if oos is None or oos.empty or "ens_weighted" not in oos.columns:
        return True
    if oos.index[-1] not in X.index:
        return True
    rows_since = (n_lab - 1) - X.index.get_loc(oos.index[-1])
    return rows_since > config.WF_STEP  # about one quarter of new labels


def main() -> int:
    ap = argparse.ArgumentParser(description="HSI daily pipeline")
    ap.add_argument("--backtest", action="store_true", help="force a fresh backtest")
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--models", nargs="*", help="subset: logit lgbm xgb lstm gru transformer")
    args = ap.parse_args()

    # 1) data + features
    if args.skip_data and config.FEATURES_PATH.exists():
        feats = builder.load_features()
    else:
        feats = builder.build_features(build_dataset.build_panel(refresh=True))
    X, y, n_lab = prepare(feats)
    log.info("Rows: %d labelled + 1 live | features: %d", n_lab, X.shape[1])

    # 2) walk-forward backtest (slow: only when forced, missing or stale)
    oos = _load_oos()
    if args.backtest or _backtest_needed(oos, X, n_lab):
        log.info("Running walk-forward backtest ...")
        oos = pipeline.run_backtest(X, y, n_lab, feats, args.models)
        oos.to_parquet(config.OOS_PATH)
        metrics = pipeline.evaluate(oos)
        metrics.to_csv(config.METRICS_PATH)
        print(metrics.round(4).to_string())
    else:
        log.info("Using stored backtest (ends %s).", oos.index[-1].date())

    # 3) live prediction (retrains every model on the latest window)
    live = pipeline.run_live(X, y, n_lab, feats, args.models, oos)

    # 4) LLM challenge review (never blocks the run)
    try:
        llm = critic.review(live, feats)
    except Exception as exc:
        log.warning("LLM step failed: %s", exc)
        llm = {}
    if llm:
        config.LLM_PATH.write_text(json.dumps(llm, indent=2, ensure_ascii=False))

    # 5) forward tracking: log today, settle earlier days, score
    df = tracker.log_prediction(live, llm)
    df = tracker.settle(df, feats)
    config.SCORE_PATH.write_text(
        json.dumps(tracker.scorecard(df), indent=2, ensure_ascii=False))

    # 6) report + email
    rep = report.build_report()
    if args.no_email:
        log.info("Email skipped (--no-email). Report: %s", rep["path"])
    else:
        emailer.send(rep)

    print(rep["text"])
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sys.exit(main())
