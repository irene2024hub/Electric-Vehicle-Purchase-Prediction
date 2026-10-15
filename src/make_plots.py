"""make_plots.py -- generates all figures used in the notebook/README/app."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, "/home/claude/EV-Purchase-Prediction/src")
from data_preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, encode_target, load_raw_data

OUT = "/home/claude/EV-Purchase-Prediction/outputs/figures"
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.dpi"] = 110

train, test = load_raw_data(
    "/home/claude/EV-Purchase-Prediction/data/train.csv",
    "/home/claude/EV-Purchase-Prediction/data/test.csv",
)
train = encode_target(train)

# 1. Target distribution
fig, ax = plt.subplots(figsize=(5, 4))
train["Will_Buy_EV"].value_counts().plot(kind="bar", color=["#4c72b0", "#dd8452"], ax=ax)
ax.set_title("Target Distribution: Will_Buy_EV")
ax.set_xlabel("")
ax.set_ylabel("Count")
for i, v in enumerate(train["Will_Buy_EV"].value_counts()):
    ax.text(i, v + 5000, f"{v:,}\n({v/len(train):.1%})", ha="center")
plt.tight_layout()
plt.savefig(f"{OUT}/target_distribution.png")
plt.close()

# 2. Numeric feature distributions
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.flat, NUMERIC_FEATURES):
    sns.histplot(train[col], kde=True, ax=ax, color="#4c72b0")
    ax.set_title(col)
axes.flat[-1].axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/numeric_distributions.png")
plt.close()

# 3. Categorical bar charts
cat_cols = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.flat, cat_cols):
    train[col].value_counts().plot(kind="bar", ax=ax, color="#55a868")
    ax.set_title(col)
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(f"{OUT}/categorical_distributions.png")
plt.close()

# 4. Target rate by key categorical drivers
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, col in zip(axes, ["Subsidy_Available", "Range_Anxiety_Level", "Environmental_Concern_Level"]):
    rates = train.groupby(col)["target"].mean().sort_values(ascending=False)
    rates.plot(kind="bar", ax=ax, color="#c44e52")
    ax.set_title(f"EV purchase rate by {col}")
    ax.set_ylabel("Observed Will_Buy_EV rate")
    ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(f"{OUT}/target_rate_by_driver.png")
plt.close()

# 5. Correlation heatmap (numeric features + target)
num_plus_target = NUMERIC_FEATURES + ["target"]
corr = train[num_plus_target].corr()
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Heatmap (numeric features + target)")
plt.tight_layout()
plt.savefig(f"{OUT}/correlation_heatmap.png")
plt.close()

print("Saved EDA figures to", OUT)
