"""HTML report + email for the pooled multi-stock ranking."""
import argparse
import json
import logging
import smtplib
from email.message import EmailMessage
from html import escape
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src import config
from src.reporting import emailer

log = logging.getLogger(__name__)
COLORS = {"LONG": "#0a7a3d", "SHORT": "#c0392b", "FLAT": "#7f8c8d"}
METRIC_COLS = ["auc", "acc", "ic_mean", "ic_t", "ic_pos", "ls_gross_sharpe",
               "ls_net_sharpe", "ls_net_ann_ret", "ls_max_dd", "turnover_per_day"]
CSS = ("body{font-family:Arial,sans-serif;background:#f4f5f7;color:#222;margin:0}"
       ".w{max-width:900px;margin:0 auto;padding:18px}"
       ".c{background:#fff;border:1px solid #e1e4e8;border-radius:8px;padding:14px 18px;margin:12px 0}"
       "h1{font-size:21px;margin:0 0 4px}h2{font-size:15px;margin:0 0 8px}"
       "table{border-collapse:collapse;width:100%;font-size:12.5px}"
       "th,td{border-bottom:1px solid #eee;padding:4px 7px;text-align:right}"
       "th:first-child,td:first-child{text-align:left}th{background:#fafbfc}"
       ".m{color:#666;font-size:12px}"
       ".warn{background:#fff8e1;border:1px solid #ffe08a;border-radius:6px;padding:7px 11px;"
       "margin:5px 0;font-size:13px}")


class Raw(str):
    """Pre-escaped HTML cell."""


def _num(v, d: int = 3) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if v != v else f"{v:,.{d}f}"


def _pct(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if v != v else f"{v * 100:.1f}%"


def _table(head: List[str], rows: List[list]) -> str:
    th = "".join(f"<th>{escape(h)}</th>" for h in head)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f"<td>{c if isinstance(c, Raw) else escape(str(c))}</td>" for c in r) + "</tr>"
    return f"<table><tr>{th}</tr>{body}</table>"


def _json(p: Path) -> dict:
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception as exc:
        log.warning("Cannot read %s: %s", p, exc)
        return {}


def _side(s: str) -> Raw:
    return Raw(f"<b style='color:{COLORS.get(s, '#7f8c8d')}'>{escape(s)}</b>")


def _warnings(live: dict, metrics: Optional[pd.DataFrame], score: dict) -> List[str]:
    w = list(live.get("warnings", []))
    if metrics is None or "ens_equal" not in metrics.index:
        w.append("No backtest metrics found: run pooled, evaluate first. Edge is unverified.")
    else:
        e = metrics.loc["ens_equal"]
        if pd.notna(e.get("ic_t")) and e["ic_t"] < 2:
            w.append(f"Backtest rank IC t-stat is {e['ic_t']:.2f} (below about 2): no demonstrated edge.")
        if "baseline_random_p95" in metrics.index:
            p95 = metrics.loc["baseline_random_p95", "ls_net_sharpe"]
            if pd.notna(e.get("ls_net_sharpe")) and e["ls_net_sharpe"] <= p95:
                w.append(f"Backtest net Sharpe {e['ls_net_sharpe']:.2f} does not beat the random-pick "
                         f"95th percentile ({p95:.2f}).")
    if score.get("sample_warning"):
        w.append(score["sample_warning"])
    if live.get("tuned_models"):
        if config.active_multi_metrics() == config.MULTI_METRICS_PATH:
            w.append("Tuned parameters are in use, but the metrics shown come from the untuned "
                     "backtest. Run `src.multi.pooled --tune` and `src.multi.evaluate --tuned`.")
        ta = live.get("tuned_at")
        if ta:
            age = (pd.Timestamp.now().normalize() - pd.Timestamp(ta)).days
            if age > config.TUNE_STALE_DAYS:
                w.append(f"Tuned parameters are {age} days old (limit {config.TUNE_STALE_DAYS}). "
                         "Re-tune with `src.multi.live --tune`.")
        else:
            w.append("Tuned parameters have no recorded date: re-tune to be sure they are current.")
    return w


def _llm_section(live: dict, llm: dict, cmp: dict) -> str:
    if not llm or llm.get("feature_date") != live["feature_date"]:
        return ""
    if llm.get("verdict") != "REVIEWED":
        return ("<div class='c'><h2>LLM review</h2><div class='m'>"
                "Not available for this run. Scores are the model's own.</div></div>")
    rows = [[s["final_rank"], s["name"], _side(s["model_side"]), _side(s["final_side"]),
             s["action"], s["reason"]] for s in llm["stocks"]]
    h = ("<div class='c'><h2>LLM review</h2>"
         f"<div class='m'>Model: {escape(str(llm['model']))} &middot; score cap "
         f"&plusmn;{llm['cap']} per stock &middot; sides changed: {llm['n_side_changed']}</div>"
         f"<p>{escape(llm['summary'])}</p>"
         f"<p class='m'>{escape(llm['market_view'])}</p>"
         + _table(["#", "Stock", "Model", "Final", "Action", "Reason"], rows))
    if llm.get("risk_flags"):
        h += "<ul>" + "".join(f"<li>{escape(x)}</li>" for x in llm["risk_flags"]) + "</ul>"
    a, b, p = cmp.get("model", {}), cmp.get("llm_adjusted", {}), cmp.get("paired")
    if p and "ic_mean" in a and "ic_mean" in b:
        h += _table(["Forward, same days", "Model only", "LLM adjusted"], [
            ["Days compared", cmp["n_days_compared"], cmp["n_days_compared"]],
            ["Rank IC mean", _num(a["ic_mean"], 4), _num(b["ic_mean"], 4)],
            ["Net Sharpe", _num(a["net_sharpe"], 2), _num(b["net_sharpe"], 2)],
            ["Net cum log return", _num(a["net_cum_log_ret"], 4), _num(b["net_cum_log_ret"], 4)],
        ])
        h += (f"<div class='m'>Paired net return diff t = {_num(p['net_ret_diff_t'], 2)}, "
              f"IC diff t = {_num(p['ic_diff_t'], 2)}. {escape(cmp.get('sample_warning', ''))}</div>")
    else:
        h += "<div class='m'>Forward comparison starts once reviewed days have settled.</div>"
    return h + "</div>"


def build_report() -> dict:
    live = _json(config.MULTI_LIVE_JSON)
    if not live:
        raise RuntimeError("live ranking JSON missing: run src.multi.live first.")
    score = _json(config.MULTI_SCORE_PATH)
    mpath = config.active_multi_metrics()
    metrics = pd.read_csv(mpath, index_col=0) if mpath.exists() else None
    tuned_lbl = "tuned" if (live.get("tuned_models") and mpath != config.MULTI_METRICS_PATH) else "untuned"
    daily = (pd.read_csv(config.MULTI_DAILY_PATH, index_col=0)
             if config.MULTI_DAILY_PATH.exists() else pd.DataFrame())

    rk = live["ranking"]
    mcols = [f"p_{m}" for m in live.get("models", [])]
    head = ["#", "Stock", "Ticker", "Score", "Agree", "Side", "Ref close", "NaN"] + \
           [c[2:] for c in mcols]
    rows = [[r["rank"], r["name"], r["ticker"], _num(r["score"], 3), _pct(r["agreement"]),
             _side(r["side"]), _num(r["ref_close"], 2), r["live_nan"]]
            + [_num(r.get(c), 3) for c in mcols] for r in rk]

    parts = [f"<h1>Stock ranking (pooled model)</h1>"
             f"<div class='m'>Features as of {escape(live['feature_date'])} &middot; target day "
             f"{escape(live['target_date'])} &middot; horizon: {escape(live['horizon'])} &middot; "
             f"{live['n_stocks']} stocks &middot; trained on {live['n_train_rows']} rows, "
             f"{live['n_features']} features</div>"]
    parts += [f"<div class='warn'>{escape(w)}</div>" for w in _warnings(live, metrics, score)]
    parts.append("<div class='c'><h2>Live ranking</h2>" + _table(head, rows) +
                 "<div class='m'>Score = mean P(beats same-day median). LONG = top "
                 f"{live['top_k']}, SHORT = bottom {live['top_k']}. Shorts are hypothetical: "
                 "borrow and short-sale rules are not modelled.</div></div>")
    parts.append(_llm_section(live, _json(config.MULTI_LLM_PATH), _json(config.MULTI_LLM_COMPARE)))

    if metrics is not None:
        keep = [c for c in METRIC_COLS if c in metrics.columns]
        mrows = [[i] + [_num(metrics.loc[i, c], 3) for c in keep] for i in metrics.index]
        parts.append(f"<div class='c'><h2>Walk-forward backtest ({tuned_lbl} parameters)</h2>"
                     + _table(["Model"] + keep, mrows)
                     + "<div class='m'>Compare ls_net_sharpe with baseline_random_p95. "
                       "Survivorship bias: the universe is today's large caps.</div></div>")

    if "ic_mean" in score:
        lp, sp = score["long_picks_beat_median"], score["short_picks_lag_median"]
        srows = [
            ["Settled days", score["n_days_settled"]],
            ["Rank IC mean / t-stat", f"{_num(score['ic_mean'], 4)} / {_num(score['ic_t'], 2)}"],
            ["LONG picks beating median", f"{_pct(lp['hit_rate'])} (n={lp['n']})"],
            ["SHORT picks lagging median", f"{_pct(sp['hit_rate'])} (n={sp['n']})"],
            ["Net Sharpe / cum log return", f"{_num(score['net_sharpe'], 2)} / {_num(score['net_cum_log_ret'], 4)}"],
            ["Max drawdown (log)", _num(score["max_dd_log"], 4)],
        ]
        parts.append("<div class='c'><h2>Forward scorecard</h2>" + _table(["Metric", "Value"], srows) + "</div>")
    else:
        parts.append("<div class='c'><h2>Forward scorecard</h2><div class='m'>"
                     f"{escape(score.get('note', 'No settled days yet.'))}</div></div>")

    if not daily.empty:
        t = daily.tail(10).iloc[::-1]
        drows = [[str(i)[:10], _pct(r["long_ret"]), _pct(r["short_ret"]), _pct(r["net"]),
                  _num(r["ic"], 3)] for i, r in t.iterrows()]
        parts.append("<div class='c'><h2>Recent settled days</h2>"
                     + _table(["Feature date", "Long leg", "Short leg", "Net", "IC"], drows) + "</div>")

    parts.append("<div class='m'>Automated model output, not investment advice. Judge the model by "
                 "backtest IC and net Sharpe versus random picks, and by the forward scorecard.</div>")
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>Stock ranking "
            f"{escape(live['feature_date'])}</title><style>{CSS}</style></head><body><div class='w'>"
            + "".join(parts) + "</div></body></html>")

    path = config.MULTI_REPORT_DIR / f"stock_ranking_{live['feature_date']}_{config.MULTI_TARGET}.html"
    path.write_text(html, encoding="utf-8")
    log.info("Report saved: %s", path)

    longs = [r["name"] for r in rk if r["side"] == "LONG"]
    shorts = [r["name"] for r in rk if r["side"] == "SHORT"]
    text = (f"Stock ranking for {live['target_date']} (features as of {live['feature_date']})\n"
            f"LONG: {', '.join(longs) or '-'}\nSHORT: {', '.join(shorts) or '-'}\n"
            "Automated model output, not investment advice. Open the attached HTML for details.")
    subject = (f"Stock ranking {live['target_date']}: LONG {','.join(longs) or '-'} | "
               f"SHORT {','.join(shorts) or '-'}")
    return {"path": path, "html": html, "text": text, "subject": subject[:200]}


def send(rep: dict) -> bool:
    missing = [k for k in ("SMTP_USER", "SMTP_PASS", "EMAIL_TO") if not getattr(config, k)]
    if missing:
        log.warning("Email skipped: missing %s", ", ".join(missing))
        return False
    msg = EmailMessage()
    msg["Subject"], msg["From"] = rep["subject"], config.SMTP_USER
    msg["To"] = ", ".join(emailer._recipients())
    msg.set_content(rep["text"])
    msg.add_alternative(rep["html"], subtype="html")
    p = Path(rep["path"])
    msg.add_attachment(p.read_bytes(), maintype="text", subtype="html", filename=p.name)
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
        log.info("Email sent to %s", msg["To"])
        return True
    except Exception as exc:
        log.warning("Email failed: %s: %s", type(exc).__name__, exc)
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Build (and optionally email) the stock ranking report")
    ap.add_argument("--email", action="store_true")
    a = ap.parse_args()
    rep = build_report()
    print(rep["path"])
    if a.email:
        print("sent" if send(rep) else "not sent")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
