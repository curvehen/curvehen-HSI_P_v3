"""One summary email for all targets, with each target's HTML report attached."""
import json
import logging
import smtplib
import time
from email.message import EmailMessage
from html import escape
from pathlib import Path
from typing import Dict, List

from src import config
from src.reporting import emailer

log = logging.getLogger(__name__)
ROOT = config.ROOT


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception as exc:
        log.warning("Cannot read %s: %s", path, exc)
        return {}


def _pct(v) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def collect(targets: List[dict], results: Dict[str, int], started: float) -> List[dict]:
    rows = []
    for t in targets:
        n = t["name"]
        live_p = ROOT / "data" / "processed" / n / "live_prediction.json"
        live = _read(live_p)
        fresh = live_p.exists() and live_p.stat().st_mtime >= started
        row = {"name": n, "label": t.get("label", n.upper()),
               "ok": results.get(n) == 0 and fresh and bool(live), "report": None}
        if not row["ok"]:
            rows.append(row)
            continue

        llm = _read(ROOT / "data" / "processed" / n / "llm_review.json")
        use_llm = (llm and llm.get("verdict") != "UNAVAILABLE"
                   and abs(float(llm.get("model_p_up", -1)) - float(live["p_up_ensemble"])) < 1e-3)
        sc = _read(ROOT / "state" / n / f"scorecard_{config.HSI_TARGET}.json").get("final_after_llm", {})

        reports = sorted((ROOT / "reports" / n).glob("*.html"), key=lambda p: p.stat().st_mtime)
        row.update(
            target_date=live.get("target_date", ""),
            direction=live.get("direction", ""),
            final=llm["final_direction"] if use_llm else live.get("direction", ""),
            verdict=llm.get("verdict", "n/a") if use_llm else "n/a",
            p_up=live.get("p_up_ensemble"),
            agreement=live.get("model_agreement"),
            fwd_n=sc.get("n"), fwd_hit=sc.get("hit_rate"),
            report=reports[-1] if reports and reports[-1].stat().st_mtime >= started else None,
        )
        rows.append(row)
    return rows


COLORS = {"UP": "#0a7a3d", "DOWN": "#c0392b"}


def render(rows: List[dict]) -> str:
    body = []
    for r in rows:
        if not r["ok"]:
            body.append(f"<tr><td>{escape(r['label'])}</td>"
                        f"<td colspan='7' style='color:#c0392b'>FAILED or stale: see job log</td></tr>")
            continue
        col = COLORS.get(r["final"], "#7f8c8d")
        fwd = f"{_pct(r['fwd_hit'])} (n={r['fwd_n']})" if r.get("fwd_n") else "n/a"
        body.append(
            f"<tr><td>{escape(r['label'])}</td><td>{escape(r['target_date'])}</td>"
            f"<td>{escape(r['direction'])}</td>"
            f"<td style='color:{col};font-weight:bold'>{escape(r['final'])}</td>"
            f"<td>{_pct(r['p_up'])}</td><td>{_pct(r['agreement'])}</td>"
            f"<td>{escape(r['verdict'])}</td><td>{fwd}</td></tr>")
    head = ("<tr><th>Target</th><th>Target day</th><th>Model</th><th>Final</th>"
            "<th>P(up)</th><th>Agreement</th><th>LLM</th><th>Forward hit</th></tr>")
    return (
        "<html><body style='font-family:Arial,sans-serif;font-size:13px'>"
        "<h3>Daily forecasts</h3>"
        "<table border='1' cellpadding='5' style='border-collapse:collapse'>"
        f"{head}{''.join(body)}</table>"
        "<p style='color:#666'>Forward hit rate is only meaningful after 100+ settled days. "
        "Per-target HTML reports are attached. Automated model output, not investment advice.</p>"
        "</body></html>")


def plain(rows: List[dict]) -> str:
    out = []
    for r in rows:
        if r["ok"]:
            out.append(f"{r['label']}: {r['final']} (P(up) {_pct(r['p_up'])}, target {r['target_date']})")
        else:
            out.append(f"{r['label']}: FAILED or stale")
    return "\n".join(out)


def send_digest(rows: List[dict]) -> bool:
    missing = [k for k in ("SMTP_USER", "SMTP_PASS", "EMAIL_TO") if not getattr(config, k)]
    if missing:
        log.warning("Digest skipped: missing %s", ", ".join(missing))
        return False
    ok_rows = [r for r in rows if r["ok"]]
    day = max((r["target_date"] for r in ok_rows), default=time.strftime("%Y-%m-%d"))
    subject = f"Daily forecasts {day}: " + ", ".join(
        f"{r['label']} {r['final']}" if r["ok"] else f"{r['label']} FAILED" for r in rows)

    msg = EmailMessage()
    msg["Subject"] = subject[:200]
    msg["From"] = config.SMTP_USER
    msg["To"] = ", ".join(emailer._recipients())
    msg.set_content(plain(rows))
    msg.add_alternative(render(rows), subtype="html")
    for r in rows:
        if r.get("report"):
            msg.add_attachment(Path(r["report"]).read_bytes(), maintype="text",
                               subtype="html", filename=f"{r['name']}_{Path(r['report']).name}")
    try:
        if config.SMTP_PORT == 465:
            with smtplib.SMTP_SSL(config.SMTP_HOST, config.SMTP_PORT, timeout=60) as s:
                s.login(config.SMTP_USER, config.SMTP_PASS)
                s.send_message(msg)
        else:
            with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=60) as s:
                s.starttls()
                s.login(config.SMTP_USER, config.SMTP_PASS)
                s.send_message(msg)
        log.info("Digest sent to %s", msg["To"])
        return True
    except Exception as exc:
        log.warning("Digest failed: %s: %s", type(exc).__name__, exc)
        return False
