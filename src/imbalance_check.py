"""
imbalance_check.py
-------------------
Quantifies class imbalance in the target and tests whether explicitly
weighting the minority class improves cross-validated ROC-AUC for the
tuned LightGBM model. Class weighting is tested (not assumed) because
ROC-AUC is a ranking metric and does not automatically benefit from
reweighting the way accuracy-style metrics might.
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

RANDOM_STATE = 42
ALL_NUMERIC = NUMERIC_FEATURES + ENGINEERED_NUMERIC + ENGINEERED_BINARY
ALL_CATEGORICAL = CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL

TUNED_LGB_PARAMS = dict(
    n_estimators=502, learning_rate=0.07342346439278777, num_leaves=17, max_depth=5,
    min_child_samples=117, subsample=0.635437443130527, colsample_bytree=0.9276348809969169,
    reg_alpha=0.01885288373806066, reg_lambda=0.4541720470948581, random_state=RANDOM_STATE, verbosity=-1,
)

train, test = load_raw_data(
    "/home/claude/EV-Purchase-Prediction/data/train.csv",
    "/home/claude/EV-Purchase-Prediction/data/test.csv",
)
train = encode_target(train)
train = add_engineered_features(train)
y = train["target"]
X = train[ALL_NUMERIC + ALL_CATEGORICAL].copy()
for c in ALL_CATEGORICAL:
    X[c] = X[c].astype("category")

n_pos = int(y.sum())
n_neg = int((1 - y).sum())
print(f"Positive (Will_Buy_EV=Yes): {n_pos} ({n_pos/len(y):.2%})")
print(f"Negative (Will_Buy_EV=No):  {n_neg} ({n_neg/len(y):.2%})")
print(f"Imbalance ratio (neg:pos):  {n_neg/n_pos:.2f} : 1")

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

for weight_mode in ["none", "balanced"]:
    aucs = []
    t0 = time.time()
    for tr_idx, val_idx in skf.split(X, y):
        params = dict(TUNED_LGB_PARAMS)
        if weight_mode == "balanced":
            params["class_weight"] = "balanced"
        model = lgb.LGBMClassifier(**params)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx], categorical_feature=ALL_CATEGORICAL)
        pred = model.predict_proba(X.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[val_idx], pred))
    print(f"class_weight={weight_mode}: CV AUC mean={np.mean(aucs):.5f} std={np.std(aucs):.5f} ({time.time()-t0:.0f}s)")
