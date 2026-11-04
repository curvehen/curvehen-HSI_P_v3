"""Optuna hyperparameter search for LightGBM / XGBoost.

Everything happens INSIDE the training window handed to tune(). The window is
split into forward-moving inner folds; in each fold:
    fit rows   = [0, s) minus its last 15% (early-stopping set)
    score rows = [s, e)   <- never seen by the fit
No data after the training window is used, so walk-forward tests stay clean.
For pooled (date x stock) tables, pass `dates` so cuts snap to date boundaries.
"""
import logging
from typing import Optional, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score

from src import config

optuna.logging.set_verbosity(optuna.logging.WARNING)
log = logging.getLogger(__name__)

INNER_CUTS = (0.55, 0.70, 0.85, 1.0)   # three forward scoring segments
ES_FRAC = 0.15
MIN_ROWS = 400
FINAL_N_EST = 600
EPS = 1e-6

DEFAULTS = {
    "lgbm": dict(num_leaves=15, max_depth=4, learning_rate=0.03, min_child_samples=40,
                 subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0),
    "xgb": dict(max_depth=3, min_child_weight=5, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.6, reg_lambda=5.0, gamma=0.0),
}


def _space_lgbm(t: optuna.Trial) -> dict:
    return dict(
        num_leaves=t.suggest_int("num_leaves", 4, 31),
        max_depth=t.suggest_int("max_depth", 2, 6),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
        min_child_samples=t.suggest_int("min_child_samples", 20, 200),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 0.1, 50.0, log=True),
    )


def _space_xgb(t: optuna.Trial) -> dict:
    return dict(
        max_depth=t.suggest_int("max_depth", 2, 5),
        min_child_weight=t.suggest_int("min_child_weight", 1, 30),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 0.1, 50.0, log=True),
        gamma=t.suggest_float("gamma", 0.0, 5.0),
    )


SPACES = {"lgbm": _space_lgbm, "xgb": _space_xgb}


def _snap(pos: int, dates) -> int:
    """Move a row cut back to the first row of its date (dates must be sorted)."""
    if dates is None or pos <= 0 or pos >= len(dates):
        return pos
    return int(np.searchsorted(dates, dates[pos], side="left"))


def inner_folds(n: int, dates=None) -> list:
    cuts = [_snap(int(n * c), dates) for c in INNER_CUTS]
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def _fit_score(name: str, params: dict, X: pd.DataFrame, y: np.ndarray,
               s: int, e: int, seed: int, dates=None) -> float:
    c = _snap(int(s * (1 - ES_FRAC)), dates)
    Xtr, ytr, Xes, yes = X.iloc[:c], y[:c], X.iloc[c:s], y[c:s]
    Xte, yte = X.iloc[s:e], y[s:e]
    if name == "lgbm":
        m = lgb.LGBMClassifier(n_estimators=FINAL_N_EST, subsample_freq=1,
                               random_state=seed, verbose=-1, n_jobs=-1, **params)
        m.fit(Xtr, ytr, eval_set=[(Xes, yes)],
              callbacks=[lgb.early_stopping(30, verbose=False)])
    else:
        m = xgb.XGBClassifier(n_estimators=FINAL_N_EST, tree_method="hist",
                              eval_metric="logloss", early_stopping_rounds=30,
                              random_state=seed, n_jobs=-1, **params)
        m.fit(Xtr, ytr, eval_set=[(Xes, yes)], verbose=False)
    p = m.predict_proba(Xte)[:, 1]
    if config.TUNE_METRIC == "auc":
        return -float(roc_auc_score(yte, p))          # minimise -AUC
    return float(log_loss(yte, np.clip(p, EPS, 1 - EPS), labels=[0, 1]))


def tune(name: str, X: pd.DataFrame, y: np.ndarray,
         n_trials: Optional[int] = None, timeout: Optional[int] = None,
         seed: int = config.SEED, dates=None) -> Tuple[dict, dict]:
    """Return (param_overrides, info). Empty overrides = keep defaults.
    dates: sorted datetime64 array aligned with X rows (pooled tables only)."""
    if name not in SPACES:
        return {}, {"note": f"{name} is not tunable"}
    if len(X) < MIN_ROWS:
        return {}, {"note": f"only {len(X)} rows (< {MIN_ROWS}): defaults kept"}
    n_trials = n_trials or config.TUNE_TRIALS
    timeout = timeout or config.TUNE_TIMEOUT
    folds = [f for f in inner_folds(len(X), dates) if f[0] > 0 and f[1] > f[0]]
    if not folds:
        return {}, {"note": "no valid inner folds: defaults kept"}

    def objective(trial: optuna.Trial) -> float:
        params = SPACES[name](trial)
        return float(np.mean([_fit_score(name, params, X, y, s, e, seed, dates)
                              for s, e in folds]))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(DEFAULTS[name])       # trial 0 = current defaults
    study.optimize(objective, n_trials=n_trials, timeout=timeout, catch=(Exception,))

    done = [t for t in study.trials if t.value is not None]
    if not done:
        log.warning("Tuning %s: no trial completed, defaults kept", name)
        return {}, {"note": "no completed trial"}
    default_score = done[0].value if done[0].number == 0 else None
    best = study.best_trial
    params = dict(best.params)
    params["n_estimators"] = FINAL_N_EST
    info = {
        "metric": config.TUNE_METRIC, "n_trials": len(done), "n_rows": len(X),
        "default_score": None if default_score is None else round(default_score, 5),
        "best_score": round(best.value, 5),
        "best_is_default": best.number == 0,
    }
    log.info("Tuned %s: default %s -> best %.5f (%d trials)", name,
             info["default_score"], best.value, len(done))
    return params, info
