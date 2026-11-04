"""Shared helpers: preprocessing, sequence windows, walk-forward splits."""
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd

from src import config
from src.features.builder import feature_columns


class Preprocessor:
    """Winsorise -> median impute -> standardise. Fitted on training rows only."""

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self.cols = list(X.columns)
        self.lo, self.hi = X.quantile(0.01), X.quantile(0.99)
        xc = X.clip(self.lo, self.hi, axis=1)
        self.med = xc.median()
        xf = xc.fillna(self.med)
        self.mu = xf.mean()
        self.sd = xf.std().replace(0, 1).fillna(1)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        xc = X[self.cols].clip(self.lo, self.hi, axis=1).fillna(self.med)
        z = ((xc - self.mu) / self.sd).clip(-5, 5).fillna(0)
        return z.values.astype(np.float32)


def make_windows(arr: np.ndarray, positions: np.ndarray, seq_len: int) -> np.ndarray:
    """Windows ending at each position (inclusive). Early rows are left-padded
    with the first row. Uses only rows <= position, so no look-ahead."""
    pad = np.vstack([np.repeat(arr[:1], seq_len - 1, axis=0), arr])
    return np.stack([pad[p:p + seq_len] for p in positions]).astype(np.float32)


def walk_forward_splits(
    n_labeled: int, min_train: int, step: int, max_train: Optional[int] = None
) -> Iterator[Tuple[int, int, int]]:
    """Yield (train_start, test_start, test_end). Train = [train_start, test_start),
    test = [test_start, test_end). Strictly forward in time."""
    start = min_train
    while start < n_labeled:
        end = min(start + step, n_labeled)
        tr0 = 0 if max_train is None else max(0, start - max_train)
        yield tr0, start, end
        start = end


def prepare(feats: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """Return (X incl. live row, y with NaN on live row, n_labeled).
    The label is the active HSI target (open-to-close or close-to-close)."""
    X = feats[feature_columns(feats)].astype(float)
    y = feats[config.HSI_DIR_COL].values.astype(float)
    n_lab = int(np.isfinite(y).sum())
    if n_lab == 0 or not np.isfinite(y[:n_lab]).all():
        raise ValueError("Labelled rows must form a contiguous prefix.")
    return X, y, n_lab
