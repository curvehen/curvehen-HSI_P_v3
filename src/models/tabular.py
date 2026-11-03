"""Tabular models. All take a raw DataFrame (NaN allowed) and output P(up)."""
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from src import config
from src.models.common import Preprocessor


def _cut(n: int) -> int:
    return int(n * (1 - config.VAL_FRAC))


class LogitModel:
    name, kind = "logit", "tabular"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LogitModel":
        self.pre = Preprocessor().fit(X)
        self.m = LogisticRegression(C=0.05, max_iter=500)
        self.m.fit(self.pre.transform(X), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.m.predict_proba(self.pre.transform(X))[:, 1]


class LGBMModel:
    name, kind = "lgbm", "tabular"

    def __init__(self, seed: int = config.SEED, overrides: dict = None):
        self.params = dict(
            n_estimators=400, learning_rate=0.03, num_leaves=15, max_depth=4,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.6, reg_lambda=5.0, random_state=seed,
            verbose=-1, n_jobs=-1,
        )
        self.params.update(overrides or {})


    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LGBMModel":
        c = _cut(len(X))
        self.cols = list(X.columns)
        self.m = lgb.LGBMClassifier(**self.params)
        self.m.fit(
            X.iloc[:c], y[:c], eval_set=[(X.iloc[c:], y[c:])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.m.predict_proba(X)[:, 1]

    def importance(self) -> pd.Series:
        s = pd.Series(self.m.booster_.feature_importance("gain"), index=self.cols)
        return (s / s.sum()).sort_values(ascending=False)


class XGBModel:
    name, kind = "xgb", "tabular"

    def __init__(self, seed: int = config.SEED, overrides: dict = None):
        self.params = dict(
            n_estimators=500, learning_rate=0.03, max_depth=3, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0,
            tree_method="hist", eval_metric="logloss",
            early_stopping_rounds=30, random_state=seed, n_jobs=-1,
        )
        self.params.update(overrides or {})

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "XGBModel":
        c = _cut(len(X))
        self.m = xgb.XGBClassifier(**self.params)
        self.m.fit(
            X.iloc[:c], y[:c], eval_set=[(X.iloc[c:], y[c:])], verbose=False
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.m.predict_proba(X)[:, 1]


def get_tabular_models(tuned: dict = None) -> list:
    """tuned: {"lgbm": {...}, "xgb": {...}} parameter overrides (optional)."""
    tuned = tuned or {}
    return [LogitModel(), LGBMModel(overrides=tuned.get("lgbm")),
            XGBModel(overrides=tuned.get("xgb"))]
