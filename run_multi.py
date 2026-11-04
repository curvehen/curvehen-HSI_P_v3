"""Run main.py once per target in targets.json, then send one digest email.

Usage:
  python run_multi.py                    # all targets
  python run_multi.py --only tencent     # one or more targets (comma-separated)
  python run_multi.py --backtest         # force fresh backtests (slow)
  python run_multi.py --backtest --tune  # fresh backtests with tuning (very slow)
  python run_multi.py --no-email         # skip the digest
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("run_multi")


def run_target(t: dict, backtest: bool, tune: bool = False,
               live_only: bool = False, deep_refit: bool = False) -> int:
    env = {
        **os.environ,
        "TARGET_NAME": t["name"],
        "TARGET_TICKER": t["ticker"],
        "TARGET_LABEL": t.get("label") or t["name"].upper(),
    }
    cmd = [sys.executable, "main.py", "--no-email"]
    if backtest:
        cmd.append("--backtest")
    if tune:
        cmd.append("--tune")
    models = (t.get("models") or "").split()
    if models:
        cmd += ["--models", *models]

    if live_only:
        cmd.append("--live-only")
    if deep_refit:
        cmd.append("--deep-refit")
    log.info("=== %s (%s) ===", t["name"], " ".join(cmd))
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
    log.info("%s finished: rc=%d, %.0fs", t["name"], rc, time.time() - t0)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-target daily run")
    ap.add_argument("--only", default="", help="comma-separated target names")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--tune", action="store_true", help="tune LightGBM/XGBoost (slow)")
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--live-only", action="store_true")
    ap.add_argument("--deep-refit", action="store_true")
    args = ap.parse_args()

    targets = json.loads((ROOT / "targets.json").read_text())
    names = [n.strip() for n in args.only.split(",") if n.strip()]
    if names:
        targets = [t for t in targets if t["name"] in names]
    if not targets:
        log.error("No matching targets.")
        return 2

    started = time.time()
    results = {t["name"]: run_target(t, args.backtest, args.tune,
                                     args.live_only, args.deep_refit) for t in targets}

    from src.reporting import digest  # imported late: config reads env at import
    rows = digest.collect(targets, results, started)
    print(digest.plain(rows))
    if not args.no_email:
        digest.send_digest(rows)

    failed = [n for n, rc in results.items() if rc != 0]
    if failed:
        log.warning("Failed targets: %s", ", ".join(failed))
    return 1 if len(failed) == len(results) else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
