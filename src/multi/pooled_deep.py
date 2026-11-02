"""B3-3c: pooled GRU walk-forward on the long table; adds p_gru to the OOS file."""
import argparse
import copy
import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src import config
from src.models.common import Preprocessor
from src.models.deep import DEVICE, RNNNet
from src.multi import features_long as fl
from src.multi.pooled import LABEL, date_splits

log = logging.getLogger(__name__)
L = config.SEQ_LEN


def _block_start(names: np.ndarray) -> np.ndarray:
    """For each row, the position of the first row of its stock block (rows sorted by name)."""
    new = np.r_[True, names[1:] != names[:-1]]
    return np.maximum.accumulate(np.where(new, np.arange(len(names)), 0))


def _windows(Z: np.ndarray, start: np.ndarray, pos: np.ndarray) -> torch.Tensor:
    """(B, L, F) windows ending at each position, never crossing into another stock."""
    idx = pos[:, None] + np.arange(-(L - 1), 1)[None, :]
    idx = np.maximum(idx, start[pos][:, None])
    return torch.from_numpy(Z[idx])


def _val_loss(net, Z, start, va, y, loss_fn) -> float:
    net.eval()
    tot = 0.0
    with torch.no_grad():
        for i in range(0, len(va), 1024):
            p = va[i:i + 1024]
            out = net(_windows(Z, start, p).to(DEVICE))
            tot += loss_fn(out, torch.from_numpy(y[p]).to(DEVICE)).item() * len(p)
    return tot / len(va)


def _fit(Z, start, tr, va, y, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    net = RNNNet(Z.shape[1], "gru").to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    best, wait = float("inf"), 0
    best_state = copy.deepcopy(net.state_dict())
    bs = config.MULTI_BATCH
    for _ in range(config.MULTI_EPOCHS):
        net.train()
        perm = tr[torch.randperm(len(tr), generator=gen).numpy()]
        for i in range(0, len(perm), bs):
            p = perm[i:i + bs]
            opt.zero_grad()
            out = net(_windows(Z, start, p).to(DEVICE))
            loss = loss_fn(out, torch.from_numpy(y[p]).to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        vl = _val_loss(net, Z, start, va, y, loss_fn)
        if vl < best - 1e-4:
            best, wait = vl, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            wait += 1
            if wait >= config.PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


def _predict(net, Z, start, pos) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(pos), 1024):
            xb = _windows(Z, start, pos[i:i + 1024]).to(DEVICE)
            out.append(torch.sigmoid(net(xb)).cpu().numpy())
    return np.concatenate(out)


def run_gru(df: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    fc = fl.long_feature_columns(df)
    g = df.sort_values("name", kind="stable")      # stock blocks, date order kept inside each
    X = g[fc].astype(float)
    y = g[LABEL].values.astype(np.float32)
    start = _block_start(g["name"].values)
    names, gidx = g["name"].values, g.index
    has = ~np.isnan(y)

    dates = gidx[has].unique().sort_values()
    splits = list(date_splits(dates, config.MULTI_WF_MIN_DATES,
                              config.MULTI_WF_STEP, config.MULTI_WF_MAX_DATES))
    if not splits:
        raise RuntimeError("Not enough dated labels for a single fold.")

    parts = []
    for k, (d0, d1, d2) in enumerate(splits, 1):
        trm = (gidx >= d0) & (gidx < d1) & has
        te = np.where((gidx >= d1) & (gidx <= d2))[0]
        tr_dates = gidx[trm].unique().sort_values()
        cut = tr_dates[int(len(tr_dates) * (1 - config.VAL_FRAC))]
        tr = np.where(trm & (gidx < cut))[0]
        va = np.where(trm & (gidx >= cut))[0]
        log.info("GRU fold %d/%d | train %d | val %d | test %d (%s -> %s)",
                 k, len(splits), len(tr), len(va), len(te), d1.date(), d2.date())
        try:
            pre = Preprocessor().fit(X.iloc[np.where(trm)[0]])
            Z = pre.transform(X)
            net = _fit(Z, start, tr, va, y, seed)
            p = _predict(net, Z, start, te)
        except Exception as exc:
            log.warning("GRU failed in fold %d: %s", k, exc)
            p = np.full(len(te), np.nan)
        parts.append(pd.DataFrame({"name": names[te], "p_gru": p}, index=gidx[te]))
    return pd.concat(parts)


def merge(oos: pd.DataFrame, gru: pd.DataFrame) -> pd.DataFrame:
    base = oos.drop(columns=["p_gru"], errors="ignore").rename_axis("date").reset_index()
    out = base.merge(gru.rename_axis("date").reset_index(),
                     on=["date", "name"], how="left").set_index("date")
    pcols = [c for c in out.columns if c.startswith("p_")]
    out["ens_equal"] = out[pcols].mean(axis=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Pooled GRU walk-forward")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    if not config.MULTI_OOS_PATH.exists():
        raise SystemExit("Run `python -m src.multi.pooled` first (same MULTI_TARGET).")
    oos = pd.read_parquet(config.MULTI_OOS_PATH)
    gru = run_gru(fl.load_long_features(), args.seed)
    out = merge(oos, gru)
    out.to_parquet(config.MULTI_OOS_PATH)
    log.info("p_gru coverage: %.1f%% of OOS rows", 100 * out["p_gru"].notna().mean())
    print(f"Updated {config.MULTI_OOS_PATH}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
