"""
predict.py
----------
Trains the final blended model (tuned LightGBM + tuned XGBoost, weighted
50/50 based on OOF-optimized blend weights) on the FULL training set and
produces the Kaggle submission file.

Model artifacts are also saved to models/ for reproducibility and for the
Streamlit app to load.
"""
import sys
import time
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/EV-Purchase-Prediction/src")
from data_preprocessing import CATEGORICAL_FEATURES, ID_COL, NUMERIC_FEATURES, TARGET_COL, encode_target, load_raw_data
from feature_engineering import ENGINEERED_BINARY, ENGINEERED_CATEGORICAL, ENGINEERED_NUMERIC, add_engineered_features
import lightgbm as lgb
import xgboost as xgb

RANDOM_STATE = 42
ALL_NUMERIC = NUMERIC_FEATURES + ENGINEERED_NUMERIC + ENGINEERED_BINARY
ALL_CATEGORICAL = CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL
DATA_DIR = "/home/claude/EV-Purchase-Prediction/data"
OUT = "/home/claude/EV-Purchase-Prediction/outputs"
MODELS_DIR = "/home/claude/EV-Purchase-Prediction/models"

LGB_WEIGHT, XGB_WEIGHT = 0.48, 0.52  # from OOF blend-weight search

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


def main():
    t0 = time.time()
    train, test = load_raw_data(f"{DATA_DIR}/train.csv", f"{DATA_DIR}/test.csv")
    train = encode_target(train)
    train = add_engineered_features(train)
    test_fe = add_engineered_features(test)

    y = train["target"]
    X = train[ALL_NUMERIC + ALL_CATEGORICAL]
    X_test = test_fe[ALL_NUMERIC + ALL_CATEGORICAL]
    test_ids = test_fe[ID_COL]

    # --- LightGBM (native categorical handling) ---
    X_lgb = X.copy()
    Xtest_lgb = X_test.copy()
    for c in ALL_CATEGORICAL:
        X_lgb[c] = X_lgb[c].astype("category")
        Xtest_lgb[c] = pd.Categorical(Xtest_lgb[c], categories=X_lgb[c].cat.categories)

    lgb_model = lgb.LGBMClassifier(**TUNED_LGB_PARAMS)
    lgb_model.fit(X_lgb, y, categorical_feature=ALL_CATEGORICAL)
    lgb_pred = lgb_model.predict_proba(Xtest_lgb)[:, 1]
    print(f"LightGBM trained on full data ({time.time()-t0:.0f}s elapsed)")

    # --- XGBoost (one-hot encoded; align columns) ---
    X_xgb = pd.get_dummies(X, columns=ALL_CATEGORICAL)
    Xtest_xgb = pd.get_dummies(X_test, columns=ALL_CATEGORICAL).reindex(columns=X_xgb.columns, fill_value=0)

    xgb_model = xgb.XGBClassifier(**TUNED_XGB_PARAMS)
    xgb_model.fit(X_xgb, y)
    xgb_pred = xgb_model.predict_proba(Xtest_xgb)[:, 1]
    print(f"XGBoost trained on full data ({time.time()-t0:.0f}s elapsed)")

    # --- Blend ---
    blend_pred = LGB_WEIGHT * lgb_pred + XGB_WEIGHT * xgb_pred

    # --- Save models ---
    with open(f"{MODELS_DIR}/lightgbm_final.pkl", "wb") as f:
        pickle.dump(lgb_model, f)
    with open(f"{MODELS_DIR}/xgboost_final.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    with open(f"{MODELS_DIR}/xgb_columns.pkl", "wb") as f:
        pickle.dump(list(X_xgb.columns), f)
    with open(f"{MODELS_DIR}/lgb_categories.pkl", "wb") as f:
        pickle.dump({c: list(X_lgb[c].cat.categories) for c in ALL_CATEGORICAL}, f)

    # --- Submission validation ---
    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: blend_pred})
    assert submission.shape[0] == len(test), "Row count mismatch with test set"
    assert submission[TARGET_COL].isnull().sum() == 0, "Missing predictions found"
    assert submission[TARGET_COL].between(0, 1).all(), "Predictions out of valid probability range"
    assert list(submission.columns) == [ID_COL, TARGET_COL], "Column names/order do not match requirement"
    assert submission[ID_COL].is_unique, "Duplicate ids in submission"

    submission.to_csv(f"{OUT}/submission.csv", index=False)
    print("\nSubmission validated and saved.")
    print(submission.head())
    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
