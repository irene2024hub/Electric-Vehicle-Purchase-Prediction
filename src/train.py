"""
train.py
--------
Cross-validated comparison of a baseline model and four stronger models,
all evaluated on the competition metric (ROC-AUC).

Compute note: this project was developed in a single-CPU sandbox, so the
model comparison uses 3-fold (rather than 5-fold) Stratified K-Fold CV to
keep total run time reasonable. The final chosen model is re-validated with
5-fold CV before being used for the Kaggle submission (see notebook).
"""

import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, "/home/claude/EV-Purchase-Prediction/src")
from data_preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    encode_target,
    load_raw_data,
)
from feature_engineering import (
    ENGINEERED_BINARY,
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
    add_engineered_features,
)

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

RANDOM_STATE = 42
N_SPLITS = 3  # see compute note above

ALL_NUMERIC = NUMERIC_FEATURES + ENGINEERED_NUMERIC + ENGINEERED_BINARY
ALL_CATEGORICAL = CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL


def build_sklearn_pipeline(estimator):
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), ALL_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ALL_CATEGORICAL),
        ]
    )
    return Pipeline([("pre", pre), ("clf", estimator)])


def cv_sklearn(X, y, estimator_fn, name):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    oof = np.zeros(len(y))
    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        pipe = build_sklearn_pipeline(estimator_fn())
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = pipe.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = pred
        auc = roc_auc_score(y.iloc[val_idx], pred)
        aucs.append(auc)
        print(f"  [{name}] fold {fold}: AUC={auc:.5f}")
    elapsed = time.time() - t0
    return {
        "model": name,
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "fold_aucs": aucs,
        "seconds": elapsed,
    }, oof


def cv_lightgbm(X, y):
    Xc = X.copy()
    for c in ALL_CATEGORICAL:
        Xc[c] = Xc[c].astype("category")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    oof = np.zeros(len(y))
    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(skf.split(Xc, y)):
        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
        model.fit(Xc.iloc[tr_idx], y.iloc[tr_idx], categorical_feature=ALL_CATEGORICAL)
        pred = model.predict_proba(Xc.iloc[val_idx])[:, 1]
        oof[val_idx] = pred
        auc = roc_auc_score(y.iloc[val_idx], pred)
        aucs.append(auc)
        print(f"  [LightGBM] fold {fold}: AUC={auc:.5f}")
    elapsed = time.time() - t0
    return {
        "model": "LightGBM",
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "fold_aucs": aucs,
        "seconds": elapsed,
    }, oof


def cv_xgboost(X, y):
    Xd = pd.get_dummies(X, columns=ALL_CATEGORICAL)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    oof = np.zeros(len(y))
    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(skf.split(Xd, y)):
        model = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_STATE,
            eval_metric="auc",
            n_jobs=1,
        )
        model.fit(Xd.iloc[tr_idx], y.iloc[tr_idx])
        pred = model.predict_proba(Xd.iloc[val_idx])[:, 1]
        oof[val_idx] = pred
        auc = roc_auc_score(y.iloc[val_idx], pred)
        aucs.append(auc)
        print(f"  [XGBoost] fold {fold}: AUC={auc:.5f}")
    elapsed = time.time() - t0
    return {
        "model": "XGBoost",
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "fold_aucs": aucs,
        "seconds": elapsed,
    }, oof


def cv_catboost(X, y):
    cat_idx = [X.columns.get_loc(c) for c in ALL_CATEGORICAL]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    oof = np.zeros(len(y))
    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        model = CatBoostClassifier(
            iterations=150,
            learning_rate=0.08,
            depth=6,
            random_state=RANDOM_STATE,
            verbose=False,
            thread_count=1,
        )
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_idx)
        pred = model.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = pred
        auc = roc_auc_score(y.iloc[val_idx], pred)
        aucs.append(auc)
        print(f"  [CatBoost] fold {fold}: AUC={auc:.5f}")
    elapsed = time.time() - t0
    return {
        "model": "CatBoost",
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "fold_aucs": aucs,
        "seconds": elapsed,
    }, oof


OUTPUTS_DIR = "/home/claude/EV-Purchase-Prediction/outputs"


def get_data():
    train, test = load_raw_data(
        "/home/claude/EV-Purchase-Prediction/data/train.csv",
        "/home/claude/EV-Purchase-Prediction/data/test.csv",
    )
    train = encode_target(train)
    train = add_engineered_features(train)
    y = train["target"]
    X = train[ALL_NUMERIC + ALL_CATEGORICAL]
    return X, y


def run_one(model_name: str):
    """Run CV for a single model and persist its result + OOF predictions.

    Designed to be invoked once per model (rather than looping over all
    five in one process) so each run comfortably fits the sandbox's
    per-command time limit.
    """
    X, y = get_data()

    if model_name == "LogisticRegression":
        res, oof = cv_sklearn(X, y, lambda: LogisticRegression(max_iter=1000), "LogisticRegression")
    elif model_name == "RandomForest":
        res, oof = cv_sklearn(
            X, y,
            lambda: RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=1, random_state=RANDOM_STATE),
            "RandomForest",
        )
    elif model_name == "LightGBM":
        res, oof = cv_lightgbm(X, y)
    elif model_name == "XGBoost":
        res, oof = cv_xgboost(X, y)
    elif model_name == "CatBoost":
        res, oof = cv_catboost(X, y)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    with open(f"{OUTPUTS_DIR}/result_{model_name}.json", "w") as f:
        json.dump(res, f, indent=2)
    np.save(f"{OUTPUTS_DIR}/oof_{model_name}.npy", oof)
    np.save(f"{OUTPUTS_DIR}/y_true.npy", y.values)
    print(f"\nSaved result_{model_name}.json and oof_{model_name}.npy")
    print(json.dumps({k: v for k, v in res.items() if k != "fold_aucs"}, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                         choices=["LogisticRegression", "RandomForest", "LightGBM", "XGBoost", "CatBoost"])
    args = parser.parse_args()
    run_one(args.model)
