"""
evaluate.py
-----------
Model explainability, calibration check, and error analysis for the final
chosen model (tuned LightGBM, used here as the representative model for
SHAP since it is fastest to explain; the deployed prediction still uses
the LightGBM+XGBoost blend).
"""
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

sys.path.insert(0, "/home/claude/EV-Purchase-Prediction/src")
from data_preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, encode_target, load_raw_data
from feature_engineering import ENGINEERED_BINARY, ENGINEERED_CATEGORICAL, ENGINEERED_NUMERIC, add_engineered_features
import lightgbm as lgb

RANDOM_STATE = 42
ALL_NUMERIC = NUMERIC_FEATURES + ENGINEERED_NUMERIC + ENGINEERED_BINARY
ALL_CATEGORICAL = CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL
OUT = "/home/claude/EV-Purchase-Prediction/outputs"

TUNED_LGB_PARAMS = dict(
    n_estimators=502, learning_rate=0.07342346439278777, num_leaves=17, max_depth=5,
    min_child_samples=117, subsample=0.635437443130527, colsample_bytree=0.9276348809969169,
    reg_alpha=0.01885288373806066, reg_lambda=0.4541720470948581, random_state=RANDOM_STATE, verbosity=-1,
)

train, test = load_raw_data(f"{OUT}/../data/train.csv", f"{OUT}/../data/test.csv")
train_raw = train.copy()
train = encode_target(train)
train = add_engineered_features(train)
y = train["target"]
X = train[ALL_NUMERIC + ALL_CATEGORICAL].copy()
for c in ALL_CATEGORICAL:
    X[c] = X[c].astype("category")

Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
raw_val_idx = Xval.index

model = lgb.LGBMClassifier(**TUNED_LGB_PARAMS)
model.fit(Xtr, ytr, categorical_feature=ALL_CATEGORICAL)
pred = model.predict_proba(Xval)[:, 1]
auc = roc_auc_score(yval, pred)
print("Hold-out AUC (tuned LightGBM):", auc)

# ---- Native feature importance -----------------------------------------
importance = pd.DataFrame({
    "feature": X.columns,
    "importance_gain": model.booster_.feature_importance(importance_type="gain"),
}).sort_values("importance_gain", ascending=False)
importance["importance_gain_pct"] = (importance["importance_gain"] / importance["importance_gain"].sum() * 100).round(2)
importance.to_csv(f"{OUT}/feature_importance.csv", index=False)
print("\n=== Feature importance (gain) ===")
print(importance.to_string(index=False))

# ---- ROC curve data -------------------------------------------------------
fpr, tpr, _ = roc_curve(yval, pred)
np.savez(f"{OUT}/roc_curve.npz", fpr=fpr, tpr=tpr, auc=auc)

# ---- Calibration check ----------------------------------------------------
prob_true, prob_pred = calibration_curve(yval, pred, n_bins=10, strategy="quantile")
np.savez(f"{OUT}/calibration.npz", prob_true=prob_true, prob_pred=prob_pred)
print("\n=== Calibration (quantile bins): predicted vs observed rate ===")
for pt, pp in zip(prob_true, prob_pred):
    print(f"  predicted~{pp:.3f}  observed={pt:.3f}")

# ---- Error analysis --------------------------------------------------------
val_df = train_raw.loc[raw_val_idx].copy()
val_df["y_true"] = yval.values
val_df["y_pred_proba"] = pred
val_df["abs_error"] = np.abs(val_df["y_true"] - val_df["y_pred_proba"])

high_conf_correct = val_df[((val_df.y_pred_proba > 0.8) & (val_df.y_true == 1)) | ((val_df.y_pred_proba < 0.2) & (val_df.y_true == 0))]
high_conf_wrong = val_df[((val_df.y_pred_proba > 0.8) & (val_df.y_true == 0)) | ((val_df.y_pred_proba < 0.2) & (val_df.y_true == 1))]
near_boundary = val_df[(val_df.y_pred_proba > 0.4) & (val_df.y_pred_proba < 0.6)]

print(f"\nHigh-confidence correct: {len(high_conf_correct)} ({len(high_conf_correct)/len(val_df):.1%})")
print(f"High-confidence WRONG:   {len(high_conf_wrong)} ({len(high_conf_wrong)/len(val_df):.1%})")
print(f"Near decision boundary (0.4-0.6): {len(near_boundary)} ({len(near_boundary)/len(val_df):.1%})")

print("\n=== Segment-level AUC (by City_Type) ===")
for seg, sub in val_df.groupby("City_Type"):
    if sub["y_true"].nunique() > 1:
        print(f"  {seg}: AUC={roc_auc_score(sub['y_true'], sub['y_pred_proba']):.4f}  n={len(sub)}")

print("\n=== Segment-level AUC (by Gender) [fairness check] ===")
for seg, sub in val_df.groupby("Gender"):
    if sub["y_true"].nunique() > 1:
        print(f"  {seg}: AUC={roc_auc_score(sub['y_true'], sub['y_pred_proba']):.4f}  n={len(sub)}  positive_rate={sub['y_true'].mean():.3f}")

val_df.to_csv(f"{OUT}/holdout_predictions.csv", index=False)
print("\nSaved feature_importance.csv, roc_curve.npz, calibration.npz, holdout_predictions.csv")
