"""Deep sequence models (PyTorch): LSTM, GRU, Transformer.

Interface (kind = "sequence"):
    fit(X_train, y_train)            X_train: raw DataFrame, contiguous rows in time order
    predict_proba(X, positions=None) X must include the lookback context before each
                                     position; positions are integer row positions in X
                                     (default = last row). Returns P(up).
Each window ends at its own row, so no future rows are used.
"""
import copy
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src import config
from src.models.common import Preprocessor, make_windows

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNNNet(nn.Module):
    def __init__(self, n_feat: int, cell: str, hidden: int = 48, dropout: float = 0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(), nn.Dropout(dropout))
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(hidden, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self.inp(x))
        return self.head(h[:, -1]).squeeze(-1)


class TransformerNet(nn.Module):
    def __init__(self, n_feat: int, seq_len: int, d_model: int = 32,
                 heads: int = 4, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.inp = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=d_model * 2, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(self.inp(x) + self.pos)
        return self.head(self.norm(h[:, -1])).squeeze(-1)


class SeqModel:
    name = "seq"
    kind = "sequence"

    def __init__(self, seed: int = config.SEED, seq_len: int = config.SEQ_LEN):
        self.seed = seed
        self.seq_len = seq_len

    def _build(self, n_feat: int) -> nn.Module:
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SeqModel":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.pre = Preprocessor().fit(X)
        Z = self.pre.transform(X)
        W = make_windows(Z, np.arange(len(Z)), self.seq_len)
        c = int(len(Z) * (1 - config.VAL_FRAC))  # chronological validation split

        dev = DEVICE
        xtr = torch.from_numpy(W[:c]).to(dev)
        ytr = torch.from_numpy(np.asarray(y[:c], dtype=np.float32)).to(dev)
        xva = torch.from_numpy(W[c:]).to(dev)
        yva = torch.from_numpy(np.asarray(y[c:], dtype=np.float32)).to(dev)

        self.net = self._build(Z.shape[1]).to(dev)
        opt = torch.optim.AdamW(
            self.net.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
        )
        loss_fn = nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(self.seed)

        best = float("inf")
        best_state = copy.deepcopy(self.net.state_dict())
        wait = 0
        bs = config.BATCH_SIZE
        for _ in range(config.EPOCHS):
            self.net.train()
            perm = torch.randperm(len(xtr), generator=gen)
            for i in range(0, len(perm), bs):
                idx = perm[i:i + bs].to(dev)
                opt.zero_grad()
                loss = loss_fn(self.net(xtr[idx]), ytr[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()

            self.net.eval()
            with torch.no_grad():
                vl = loss_fn(self.net(xva), yva).item()
            if vl < best - 1e-4:
                best, wait = vl, 0
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                wait += 1
                if wait >= config.PATIENCE:
                    break

        self.net.load_state_dict(best_state)
        self.net.eval()
        self.val_loss = best
        return self

    def predict_proba(self, X: pd.DataFrame, positions: Optional[np.ndarray] = None) -> np.ndarray:
        Z = self.pre.transform(X)
        if positions is None:
            positions = np.array([len(Z) - 1])
        W = make_windows(Z, np.asarray(positions), self.seq_len)
        out = []
        with torch.no_grad():
            for i in range(0, len(W), 256):
                xb = torch.from_numpy(W[i:i + 256]).to(DEVICE)
                out.append(torch.sigmoid(self.net(xb)).cpu().numpy())
        return np.concatenate(out)


class LSTMModel(SeqModel):
    name = "lstm"

    def _build(self, n_feat: int) -> nn.Module:
        return RNNNet(n_feat, "lstm")


class GRUModel(SeqModel):
    name = "gru"

    def _build(self, n_feat: int) -> nn.Module:
        return RNNNet(n_feat, "gru")


class TransformerModel(SeqModel):
    name = "transformer"

    def _build(self, n_feat: int) -> nn.Module:
        return TransformerNet(n_feat, self.seq_len)


def get_sequence_models() -> list:
    return [LSTMModel(), GRUModel(), TransformerModel()]
