"""make_model_plots.py -- model comparison, ROC, calibration, feature importance, SHAP plots."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

OUT = "/home/claude/EV-Purchase-Prediction/outputs"
FIG = f"{OUT}/figures"
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

# 1. Model comparison bar chart
rows = []
import json
for name in ["LogisticRegression", "RandomForest", "LightGBM", "XGBoost", "CatBoost"]:
    with open(f"{OUT}/result_{name}.json") as f:
        r = json.load(f)
        rows.append(r)
comp = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False)
comp.to_csv(f"{OUT}/model_comparison.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(comp["model"], comp["cv_auc_mean"], xerr=comp["cv_auc_std"], color="#4c72b0", capsize=4)
ax.set_xlim(0.9, 0.95)
ax.set_xlabel("Cross-Validated ROC-AUC")
ax.set_title("Model Comparison (3-fold Stratified CV)")
for i, (v, s) in enumerate(zip(comp["cv_auc_mean"], comp["cv_auc_std"])):
    ax.text(v + 0.001, i, f"{v:.4f}", va="center")
plt.tight_layout()
plt.savefig(f"{FIG}/model_comparison.png")
plt.close()

# 2. ROC curve
roc = np.load(f"{OUT}/roc_curve.npz")
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(roc["fpr"], roc["tpr"], color="#c44e52", lw=2, label=f"Tuned LightGBM (AUC={float(roc['auc']):.4f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve (hold-out validation set)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG}/roc_curve.png")
plt.close()

# 3. Calibration curve
cal = np.load(f"{OUT}/calibration.npz")
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(cal["prob_pred"], cal["prob_true"], marker="o", color="#4c72b0", label="Tuned LightGBM")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed frequency")
ax.set_title("Calibration Curve")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG}/calibration_curve.png")
plt.close()

# 4. Feature importance (gain)
imp = pd.read_csv(f"{OUT}/feature_importance.csv").head(12)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(imp["feature"][::-1], imp["importance_gain_pct"][::-1], color="#55a868")
ax.set_xlabel("Importance (% of total gain)")
ax.set_title("Top 12 Features by LightGBM Gain Importance")
plt.tight_layout()
plt.savefig(f"{FIG}/feature_importance.png")
plt.close()

# 5. SHAP summary plot
sv = np.load(f"{OUT}/shap_values.npy")
sample = pd.read_csv(f"{OUT}/shap_sample.csv")
plt.figure()
shap.summary_plot(sv, sample, show=False, max_display=12)
plt.tight_layout()
plt.savefig(f"{FIG}/shap_summary.png", bbox_inches="tight")
plt.close()

print("Saved model plots to", FIG)
print(comp[["model", "cv_auc_mean", "cv_auc_std"]])
