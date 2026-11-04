"""Backtest and forward-tracking charts (PNG, saved to reports/)."""
import logging
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.llm import tracker
from src.models.pipeline import COST_BPS, NEUTRAL_BAND, WEIGHT_WINDOW

log = logging.getLogger(__name__)


def _net(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    pos = np.where(p > 0.5 + NEUTRAL_BAND, 1.0, np.where(p < 0.5 - NEUTRAL_BAND, -1.0, 0.0))
    turn = 2.0 * np.abs(pos) if config.HSI_FLAT else np.abs(np.diff(pos, prepend=0.0))

    return pos * r - turn * COST_BPS / 1e4


def _save(fig, name: str) -> str:
    path = Path(config.CHART_DIR) / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def equity_curve(oos: pd.DataFrame) -> str:
    d = oos[["ens_weighted", "ens_equal", "target_ret"]].dropna()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for c, lbl in (("ens_weighted", "Weighted ensemble"), ("ens_equal", "Equal ensemble")):
        ax.plot(d.index, np.cumsum(_net(d[c].values, d["target_ret"].values)), label=lbl)
    ax.plot(d.index, np.cumsum(d["target_ret"].values), label="Buy & hold", color="grey", ls="--")
    ax.set_title(f"Walk-forward out-of-sample cumulative log return (cost {COST_BPS:g} bps)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Cumulative log return")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, "equity_curve.png")


def model_comparison(metrics: pd.DataFrame) -> str:
    m = metrics.drop(index="baseline_buyhold", errors="ignore")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(m.index, m["auc"].fillna(0.5), color="steelblue")
    axes[0].axhline(0.5, color="red", ls="--", lw=1, label="Random (0.5)")
    axes[0].set_title("Out-of-sample AUC")
    axes[0].legend()
    axes[1].bar(m.index, m["sharpe"].fillna(0.0), color="darkorange")
    if "baseline_buyhold" in metrics.index:
        axes[1].axhline(float(metrics.loc["baseline_buyhold", "sharpe"]),
                        color="grey", ls="--", lw=1, label="Buy & hold")
        axes[1].legend()
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_title(f"Net Sharpe (cost {COST_BPS:g} bps)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")
    return _save(fig, "model_comparison.png")


def rolling_accuracy(oos: pd.DataFrame) -> str:
    d = oos[["ens_weighted", "target_dir"]].dropna()
    hit = ((d["ens_weighted"] > 0.5) == (d["target_dir"] == 1)).astype(float)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(d.index, hit.rolling(WEIGHT_WINDOW).mean(), label=f"Ensemble hit rate ({WEIGHT_WINDOW}d)")
    ax.plot(d.index, d["target_dir"].rolling(WEIGHT_WINDOW).mean(),
            color="grey", ls="--", label="Share of up days")
    ax.axhline(0.5, color="red", lw=0.8)
    ax.set_title("Rolling out-of-sample direction accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, "rolling_accuracy.png")


def forward_chart(df: pd.DataFrame) -> str:
    st = df[df["actual_dir"].notna()].copy()
    if len(st) < 2:
        return ""
    st["date"] = pd.to_datetime(st["feature_date"])
    fig, ax = plt.subplots(figsize=(9, 4))
    for c, lbl in (("model_correct", "Model"), ("final_correct", "After LLM")):
        ax.plot(st["date"], st[c].expanding().mean(), marker="o", ms=3, label=lbl)
    ax.axhline(0.5, color="red", lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_title(f"Forward cumulative hit rate (n={len(st)}; noisy below {tracker.MIN_SAMPLE})")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, "forward_hit_rate.png")


def make_all() -> Dict[str, str]:
    """Build every chart that has data. Returns {name: path}; missing ones are omitted."""
    out: Dict[str, str] = {}
    oos_path, metrics_path = config.active_hsi_paths()
    if oos_path.exists():
        oos = pd.read_parquet(oos_path)

        if {"ens_weighted", "ens_equal"} <= set(oos.columns):
            out["equity"] = equity_curve(oos)
            out["rolling"] = rolling_accuracy(oos)
    else:
        log.warning("No OOS file: run the backtest first.")
    if metrics_path.exists():
        out["models"] = model_comparison(pd.read_csv(metrics_path, index_col=0))

    df = tracker.load_log()
    path = forward_chart(df) if not df.empty else ""
    if path:
        out["forward"] = path
    log.info("Charts: %s", ", ".join(out) or "none")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(make_all())
