"""
tune.py
-------
Hyperparameter optimization for the two top candidate models
(LightGBM and XGBoost) using Optuna, optimizing directly for ROC-AUC
(the competition metric) rather than accuracy.

Compute note: tuning trials use a single 80/20 stratified split (not full
K-fold) to keep each trial fast in this single-CPU sandbox. The winning
hyperparameters are re-validated with proper Stratified K-Fold CV
afterwards (see evaluate.py / notebook) before being trusted.
"""

import sys
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, "/home/claude/EV-Purchase-Prediction/src")
from data_preprocessing import encode_target, load_raw_data
from feature_engineering import (
    ENGINEERED_BINARY,
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
    add_engineered_features,
)
from data_preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

import lightgbm as lgb
import xgboost as xgb

RANDOM_STATE = 42
ALL_NUMERIC = NUMERIC_FEATURES + ENGINEERED_NUMERIC + ENGINEERED_BINARY
ALL_CATEGORICAL = CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_split():
    train, test = load_raw_data(
        "/home/claude/EV-Purchase-Prediction/data/train.csv",
        "/home/claude/EV-Purchase-Prediction/data/test.csv",
    )
    train = encode_target(train)
    train = add_engineered_features(train)
    y = train["target"]
    X = train[ALL_NUMERIC + ALL_CATEGORICAL]
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    return Xtr, Xval, ytr, yval


def lgb_objective(trial, Xtr, Xval, ytr, yval):
    Xtr_c = Xtr.copy()
    Xval_c = Xval.copy()
    for c in ALL_CATEGORICAL:
        Xtr_c[c] = Xtr_c[c].astype("category")
        Xval_c[c] = Xval_c[c].astype("category")

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "random_state": RANDOM_STATE,
        "verbosity": -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(Xtr_c, ytr, categorical_feature=ALL_CATEGORICAL)
    pred = model.predict_proba(Xval_c)[:, 1]
    return roc_auc_score(yval, pred)


def xgb_objective(trial, Xtr, Xval, ytr, yval):
    Xtr_d = pd.get_dummies(Xtr, columns=ALL_CATEGORICAL)
    Xval_d = pd.get_dummies(Xval, columns=ALL_CATEGORICAL).reindex(columns=Xtr_d.columns, fill_value=0)

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "random_state": RANDOM_STATE,
        "eval_metric": "auc",
        "n_jobs": 1,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(Xtr_d, ytr)
    pred = model.predict_proba(Xval_d)[:, 1]
    return roc_auc_score(yval, pred)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lightgbm", "xgboost"], required=True)
    parser.add_argument("--n_trials", type=int, default=8)
    args = parser.parse_args()

    Xtr, Xval, ytr, yval = get_split()

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{args.model}_study",
        storage=f"sqlite:////home/claude/EV-Purchase-Prediction/outputs/optuna_{args.model}.db",
        load_if_exists=True,
    )

    t0 = time.time()
    if args.model == "lightgbm":
        study.optimize(lambda t: lgb_objective(t, Xtr, Xval, ytr, yval), n_trials=args.n_trials)
    else:
        study.optimize(lambda t: xgb_objective(t, Xtr, Xval, ytr, yval), n_trials=args.n_trials)
    elapsed = time.time() - t0

    print(f"Completed {args.n_trials} trials in {elapsed:.1f}s. Total trials so far: {len(study.trials)}")
    print("Best AUC so far:", study.best_value)
    print("Best params:", study.best_params)
