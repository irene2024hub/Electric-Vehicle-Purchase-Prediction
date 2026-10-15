"""
generate_tuned_oof.py
----------------------
Generates out-of-fold (OOF) predictions for the tuned LightGBM and tuned
XGBoost models using 3-fold Stratified CV. These OOF predictions are used
for (a) an honest CV estimate of the tuned models and (b) determining
ensemble blend weights without touching the test set.
"""
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "/home/claude/EV-Purchase-Prediction/src")
from data_preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, encode_target, load_raw_data
from feature_engineering import ENGINEERED_BINARY, ENGINEERED_CATEGORICAL, ENGINEERED_NUMERIC, add_engineered_features
import lightgbm as lgb
import xgboost as xgb

RANDOM_STATE = 42
ALL_NUMERIC = NUMERIC_FEATURES + ENGINEERED_NUMERIC + ENGINEERED_BINARY
ALL_CATEGORICAL = CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL
OUT = "/home/claude/EV-Purchase-Prediction/outputs"

TUNED_LGB_PARAMS = dict(
    n_estimators=502, learning_rate=0.07342346439278777, num_leaves=17, max_depth=5,
    min_child_samples=117, subsample=0.635437443130527, colsample_bytree=0.9276348809969169,
    reg_alpha=0.01885288373806066, reg_lambda=0.4541720470948581, random_state=RANDOM_STATE, verbosity=-1,
)
TUNED_XGB_PARAMS = dict(
    n_estimators=515, learning_rate=0.09473195324161723, max_depth=4, min_child_weight=11,
    subsample=0.8416918882638424, colsample_bytree=0.8704272083934104,
    reg_alpha=1.5796040714216826, reg_lambda=0.07464750535175321,
    random_state=RANDOM_STATE, eval_metric="auc", n_jobs=1,
)


def get_data():
    train, test = load_raw_data(f"{OUT}/../data/train.csv", f"{OUT}/../data/test.csv")
    train = encode_target(train)
    train = add_engineered_features(train)
    y = train["target"]
    X = train[ALL_NUMERIC + ALL_CATEGORICAL]
    return X, y


def run(model_name):
    X, y = get_data()
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    t0 = time.time()

    if model_name == "lightgbm_tuned":
        Xc = X.copy()
        for c in ALL_CATEGORICAL:
            Xc[c] = Xc[c].astype("category")
        for tr_idx, val_idx in skf.split(Xc, y):
            model = lgb.LGBMClassifier(**TUNED_LGB_PARAMS)
            model.fit(Xc.iloc[tr_idx], y.iloc[tr_idx], categorical_feature=ALL_CATEGORICAL)
            oof[val_idx] = model.predict_proba(Xc.iloc[val_idx])[:, 1]
    elif model_name == "xgboost_tuned":
        Xd = pd.get_dummies(X, columns=ALL_CATEGORICAL)
        for tr_idx, val_idx in skf.split(Xd, y):
            model = xgb.XGBClassifier(**TUNED_XGB_PARAMS)
            model.fit(Xd.iloc[tr_idx], y.iloc[tr_idx])
            oof[val_idx] = model.predict_proba(Xd.iloc[val_idx])[:, 1]
    else:
        raise ValueError(model_name)

    auc = roc_auc_score(y, oof)
    print(f"{model_name}: OOF AUC={auc:.5f} ({time.time()-t0:.0f}s)")
    np.save(f"{OUT}/oof_{model_name}.npy", oof)
    np.save(f"{OUT}/y_true.npy", y.values)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["lightgbm_tuned", "xgboost_tuned"], required=True)
    args = p.parse_args()
    run(args.model)
