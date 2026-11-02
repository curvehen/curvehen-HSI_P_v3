"""Build the daily HTML report.

Two renderings of the same content:
  - file : charts embedded as base64 (self-contained, saved to reports/)
  - email: charts referenced as cid: (Gmail/Outlook block data: URIs)
"""
import base64
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from jinja2 import Environment

from src import config
from src.llm import tracker
from src.reporting import charts

log = logging.getLogger(__name__)

CHART_TITLES = {
    "equity": "Walk-forward equity curve (out-of-sample)",
    "models": "Model comparison: AUC and Sharpe",
    "rolling": "Rolling out-of-sample accuracy",
    "forward": "Forward hit rate (live log)",
}
METRIC_COLS = ["n", "accuracy", "auc", "logloss", "hit_rate_traded",
               "sharpe", "max_dd_log", "total_log_ret", "exposure"]

TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>HSI report {{ live.feature_date }}</title>
<style>
body{font-family:Arial,Helvetica,sans-serif;background:#f4f5f7;color:#222;margin:0}
.wrap{max-width:860px;margin:0 auto;padding:20px}
.card{background:#fff;border-radius:8px;padding:16px 20px;margin:14px 0;border:1px solid #e1e4e8}
h1{margin:0 0 4px 0;font-size:22px} h2{font-size:16px;margin:0 0 10px 0}
.muted{color:#666;font-size:12px}
.big{font-size:30px;font-weight:bold}
.UP{color:#0a7a3d}.DOWN{color:#c0392b}.NEUTRAL{color:#7f8c8d}.UNAVAILABLE{color:#7f8c8d}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid #eee;padding:5px 8px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:#fafbfc}
.warn{background:#fff8e1;border:1px solid #ffe08a;border-radius:6px;padding:8px 12px;margin:6px 0;font-size:13px}
ul{margin:4px 0 8px 18px;padding:0;font-size:13px}
img{max-width:100%;height:auto}
</style></head><body><div class="wrap">

<h1>HSI next-day forecast</h1>
<div class="muted">Features as of {{ live.feature_date }} (HK close) &middot;
Target day {{ live.target_date }} &middot; Reference close {{ live.ref_hsi_close|num(2) }} &middot;
Generated {{ live.generated_at }}</div>

{% for w in warnings %}<div class="warn">{{ w }}</div>{% endfor %}

<div class="card">
<h2>Model call</h2>
<div class="big {{ live.direction }}">{{ live.direction }}</div>
<div>P(up), weighted ensemble: <b>{{ live.p_up_ensemble|pct }}</b>
&middot; equal-weight: {{ live.p_up_equal|pct }}
&middot; model agreement: {{ live.model_agreement|pct }}</div>
<div class="muted">Trained on {{ live.n_train_rows }} rows &middot; {{ live.n_features }} features &middot;
{{ live.live_row_nan }} missing on the live row</div>
<table style="margin-top:10px"><tr><th>Model</th><th>P(up)</th><th>Weight</th></tr>
{% for r in model_rows %}<tr><td>{{ r.name }}</td><td>{{ r.p|pct }}</td><td>{{ r.w|pct }}</td></tr>{% endfor %}
</table></div>

<div class="card">
<h2>LLM challenge review</h2>
{% if llm and llm.verdict != 'UNAVAILABLE' %}
<div>Verdict: <b>{{ llm.verdict }}</b> &middot; adjusted P(up): {{ llm.adjusted_p_up|pct }}
&middot; final call: <b class="{{ llm.final_direction }}">{{ llm.final_direction }}</b>
{% if llm.direction_changed %}(changed from {{ llm.model_direction }}){% endif %}</div>
<p>{{ llm.summary }}</p>
{% for title, key in [('Bull case','bull_case'),('Bear case','bear_case'),('Contradictions','contradictions'),('Risk flags','risk_flags')] %}
{% if llm[key] %}<b>{{ title }}</b><ul>{% for i in llm[key] %}<li>{{ i }}</li>{% endfor %}</ul>{% endif %}
{% endfor %}
<div class="muted">Model: {{ llm.model }}. The LLM sees only the numbers supplied by the pipeline, and its
probability adjustment is capped at &plusmn;{{ cap }}.</div>
{% else %}<div class="muted">LLM review not available for this run.</div>{% endif %}
</div>

{% if top_features %}<div class="card"><h2>Top model features (live values)</h2>
<table><tr><th>Feature</th><th>Value</th></tr>
{% for k, v in top_features.items() %}<tr><td>{{ k }}</td><td>{{ v|num(5) }}</td></tr>{% endfor %}
</table></div>{% endif %}

{% if metrics %}<div class="card"><h2>Walk-forward out-of-sample metrics</h2>
<table><tr><th>Model</th>{% for c in metric_cols %}<th>{{ c }}</th>{% endfor %}</tr>
{% for r in metrics %}<tr><td>{{ r.model }}</td>{% for c in metric_cols %}<td>{{ r[c]|num(3) }}</td>{% endfor %}</tr>{% endfor %}
</table>
<div class="muted">Strategy: long / short / flat from P(up), with costs. Compare against baseline_buyhold.</div></div>{% endif %}

{% if score %}<div class="card"><h2>Forward scorecard (live predictions)</h2>
<div>Logged {{ score.n_logged }} &middot; settled {{ score.n_settled }}</div>
{% if score.model %}
<table><tr><th>Call</th><th>n</th><th>Hit rate</th><th>95% CI</th></tr>
<tr><td>Model</td><td>{{ score.model.n }}</td><td>{{ score.model.hit_rate|pct }}</td><td>{{ score.model.ci95_low|pct }} - {{ score.model.ci95_high|pct }}</td></tr>
<tr><td>After LLM</td><td>{{ score.final_after_llm.n }}</td><td>{{ score.final_after_llm.hit_rate|pct }}</td><td>{{ score.final_after_llm.ci95_low|pct }} - {{ score.final_after_llm.ci95_high|pct }}</td></tr>
</table>{% endif %}
{% if score.sample_warning %}<div class="warn">{{ score.sample_warning }}</div>{% endif %}
</div>{% endif %}

{% if recent %}<div class="card"><h2>Recent predictions</h2>
<table><tr><th>Feature date</th><th>Target</th><th>Model</th><th>Final</th><th>Actual</th><th>Correct</th></tr>
{% for r in recent %}<tr><td>{{ r.feature_date }}</td><td>{{ r.target_date }}</td><td>{{ r.model_direction }}</td>
<td>{{ r.final_direction }}</td><td>{{ r.actual_ret|pct }}</td>
<td>{% if r.final_correct is none %}-{% else %}{{ 'yes' if r.final_correct == 1 else 'no' }}{% endif %}</td></tr>{% endfor %}
</table></div>{% endif %}

{% for key, title in chart_titles %}{% if key in images %}
<div class="card"><h2>{{ title }}</h2><img src="{{ images[key] }}" alt="{{ title }}"></div>
{% endif %}{% endfor %}

<div class="muted">Automated statistical model output, not investment advice. Daily direction is close to a
coin flip: judge the model by out-of-sample AUC, net Sharpe versus buy-and-hold, and the forward scorecard.</div>
</div></body></html>"""


def _num(v, d: int = 3) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    try:
        return f"{float(v):,.{d}f}"
    except (TypeError, ValueError):
        return str(v)


def _pct(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(v)


def _json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text()) if Path(path).exists() else {}
    except Exception as exc:
        log.warning("Cannot read %s: %s", path, exc)
        return {}


def _records(df: pd.DataFrame) -> List[dict]:
    return df.astype(object).where(df.notna(), None).to_dict("records")


def build_context() -> dict:
    live = _json(config.LIVE_PATH)
    if not live:
        raise RuntimeError("live_prediction.json missing: run the pipeline in live mode first.")
    warnings: List[str] = []

    llm = _json(config.LLM_PATH)
    if llm and abs(float(llm.get("model_p_up", -1)) - float(live["p_up_ensemble"])) > 1e-3:
        warnings.append("LLM review does not match today's model output (stale file): omitted.")
        llm = {}

    weights = live.get("weights", {})
    model_rows = sorted(
        ({"name": k, "p": v, "w": weights.get(k)} for k, v in live.get("model_probs", {}).items()),
        key=lambda r: -(r["w"] or 0),
    )

    metrics: List[dict] = []
    if config.METRICS_PATH.exists():
        m = pd.read_csv(config.METRICS_PATH, index_col=0)
        cols = [c for c in METRIC_COLS if c in m.columns]
        metrics = _records(m[cols].reset_index().rename(columns={"index": "model"}))
        ens = m.loc["ens_weighted"] if "ens_weighted" in m.index else None
        if ens is not None and pd.notna(ens.get("auc")) and ens["auc"] < 0.52:
            warnings.append(f"Out-of-sample ensemble AUC is {ens['auc']:.3f}, close to random (0.5).")

    nf = max(int(live.get("n_features", 1)), 1)
    if live.get("live_row_nan", 0) / nf > 0.10:
        warnings.append(f"{live['live_row_nan']} of {nf} live features are missing: "
                        "overnight data may not be updated yet.")
    if (live.get("model_agreement") or 1) < 0.6:
        warnings.append("Models disagree with each other (agreement below 60%).")

    score = _json(config.SCORE_PATH)
    df = tracker.load_log()
    recent = _records(df.tail(10).iloc[::-1]) if not df.empty else []

    return {
        "live": live, "llm": llm, "model_rows": model_rows, "metrics": metrics,
        "metric_cols": [c for c in METRIC_COLS if metrics and c in metrics[0]],
        "top_features": live.get("top_features_live", {}),
        "score": score, "recent": recent, "warnings": warnings,
        "cap": config.LLM_MAX_ADJ,
        "chart_titles": list(CHART_TITLES.items()),
    }


def final_direction(ctx: dict) -> str:
    llm = ctx["llm"]
    if llm and llm.get("verdict") != "UNAVAILABLE":
        return llm["final_direction"]
    return ctx["live"]["direction"]


def render(ctx: dict, images: Dict[str, str]) -> str:
    env = Environment(autoescape=True)
    env.filters["num"], env.filters["pct"] = _num, _pct
    return env.from_string(TEMPLATE).render(images=images, **ctx)


def _data_uri(path: str) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def plain_text(ctx: dict) -> str:
    lv = ctx["live"]
    lines = [
        f"HSI forecast for {lv['target_date']} (features as of {lv['feature_date']})",
        f"Model: {lv['direction']}  P(up)={_pct(lv['p_up_ensemble'])}  agreement={_pct(lv['model_agreement'])}",
        f"Final after LLM: {final_direction(ctx)}",
    ]
    lines += [f"WARNING: {w}" for w in ctx["warnings"]]
    lines.append("Automated model output, not investment advice. Open the HTML version for details.")
    return "\n".join(lines)


def build_report() -> dict:
    """Build charts and both HTML renderings; save the self-contained file."""
    ctx = build_context()
    imgs = charts.make_all()
    cids = {k: f"chart_{k}" for k in imgs}

    html_file = render(ctx, {k: _data_uri(p) for k, p in imgs.items()})
    html_email = render(ctx, {k: f"cid:{cids[k]}" for k in imgs})

    path = config.REPORT_DIR / f"hsi_report_{ctx['live']['feature_date']}.html"
    path.write_text(html_file, encoding="utf-8")
    log.info("Report saved: %s", path)

    lv = ctx["live"]
    return {
        "path": path,
        "html_email": html_email,
        "text": plain_text(ctx),
        "images": {cids[k]: imgs[k] for k in imgs},  # cid -> png path
        "subject": (f"HSI forecast {lv['target_date']}: {final_direction(ctx)} "
                    f"(P(up) {_pct(lv['p_up_ensemble'])})"),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(build_report()["path"])
