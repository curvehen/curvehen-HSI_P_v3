"""Sequence models (LSTM, GRU, Transformer). Output P(up) for the last step of a window.

Interface (used by pipeline._predict):
    fit(X_raw_df, y)                    -> self
    predict_proba(X_raw_df, positions)  -> P(up) for windows ending at `positions`
"""
import copy
import logging

import numpy as np
import pandas as pd
import torch
from torch import nn

from src import config
from src.models.common import Preprocessor, make_windows

log = logging.getLogger(__name__)

HIDDEN = getattr(config, "DEEP_HIDDEN", 32)
DROPOUT = getattr(config, "DEEP_DROPOUT", 0.3)
EPOCHS = getattr(config, "DEEP_EPOCHS", 40)
PATIENCE = getattr(config, "DEEP_PATIENCE", 6)
BATCH = getattr(config, "DEEP_BATCH", 64)
LR = getattr(config, "DEEP_LR", 1e-3)
WD = getattr(config, "DEEP_WEIGHT_DECAY", 1e-3)
DEVICE = torch.device("cpu")


class _RNNNet(nn.Module):
    def __init__(self, n_feat: int, cell: str):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_feat, HIDDEN, batch_first=True)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(self.drop(out[:, -1])).squeeze(-1)


class _TransformerNet(nn.Module):
    def __init__(self, n_feat: int, seq_len: int):
        super().__init__()
        d = HIDDEN
        self.proj = nn.Linear(n_feat, d)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=2 * d, dropout=DROPOUT,
            batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        h = self.enc(self.proj(x) + self.pos[:, : x.shape[1]])
        return self.head(self.drop(h[:, -1])).squeeze(-1)


class SeqModel:
    kind = "sequence"

    def __init__(self, name: str, seed: int = config.SEED):
        self.name = name
        self.seed = seed

    def _build(self, n_feat: int) -> nn.Module:
        if self.name == "transformer":
            return _TransformerNet(n_feat, config.SEQ_LEN)
        return _RNNNet(n_feat, self.name)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SeqModel":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        L = config.SEQ_LEN
        self.pre = Preprocessor().fit(X)
        Z = self.pre.transform(X)
        n = len(Z)
        c = max(1, int(n * (1 - config.VAL_FRAC)))
        Wtr = torch.from_numpy(make_windows(Z, np.arange(c), L))
        ytr = torch.from_numpy(np.asarray(y[:c], dtype=np.float32))
        if c < n:
            Wva = torch.from_numpy(make_windows(Z, np.arange(c, n), L))
            yva = torch.from_numpy(np.asarray(y[c:n], dtype=np.float32))
        else:
            Wva = yva = None

        self.net = self._build(Z.shape[1]).to(DEVICE)
        opt = torch.optim.Adam(self.net.parameters(), lr=LR, weight_decay=WD)
        lossf = nn.BCEWithLogitsLoss()
        best, best_state, bad = np.inf, None, 0
        for ep in range(EPOCHS):
            self.net.train()
            perm = torch.randperm(len(Wtr))
            for i in range(0, len(perm), BATCH):
                b = perm[i:i + BATCH]
                opt.zero_grad()
                loss = lossf(self.net(Wtr[b]), ytr[b])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            if Wva is None:
                continue
            self.net.eval()
            with torch.no_grad():
                v = float(lossf(self.net(Wva), yva))
            if v < best - 1e-5:
                best, best_state, bad = v, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict_proba(self, X: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
        Z = self.pre.transform(X)
        W = torch.from_numpy(make_windows(Z, np.asarray(positions), config.SEQ_LEN))
        self.net.eval()
        with torch.no_grad():
            return torch.sigmoid(self.net(W)).numpy().astype(float)


def get_sequence_models() -> list:
    return [SeqModel("lstm"), SeqModel("gru"), SeqModel("transformer")]
