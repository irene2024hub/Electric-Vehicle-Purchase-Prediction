"""
feature_engineering.py
-----------------------
Creates engineered features on top of the raw EV purchase columns.

Every feature here is built from information already present in a single
row (no aggregation across rows, no use of the target), so these transforms
are safe to apply identically to train and test without leakage.
"""

import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Age group -----------------------------------------------------
    # Why: raw age is continuous; buying behaviour often clusters by life
    # stage rather than changing smoothly year over year. A tree model can
    # find these splits itself, but a bucketed version helps linear models
    # and makes the EDA / business story easier to read.
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["Under_25", "26_35", "36_45", "46_55", "56_65", "Over_65"],
    ).astype(str)

    # --- Income band -----------------------------------------------------
    # Why: affordability is rarely linear in raw income; banding captures
    # "can this person plausibly afford an EV" style thresholds.
    # NOTE: uses FIXED bin edges (computed once from the training set) rather
    # than qcut()'d fresh on whatever dataframe is passed in. Recomputing
    # quantiles separately on train vs. test vs. a single inference row would
    # (a) give test rows different bin boundaries than train rows saw during
    # fitting, and (b) crash outright on a single-row dataframe (no
    # quantiles can be computed from one value). Fixed edges make the
    # feature well-defined and consistent everywhere it's used.
    INCOME_BAND_EDGES = [-np.inf, 62671, 78915, 91619, 107798, np.inf]
    df["Income_Band"] = pd.cut(
        df["Annual_Income_USD"],
        bins=INCOME_BAND_EDGES,
        labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"],
    ).astype(str)

    # --- Total charging access ------------------------------------------
    # Why: home + work charging station counts both proxy the same
    # underlying idea (ease of charging); summing gives one combined
    # "charging accessibility" signal.
    df["Total_Charging_Access"] = (
        df["Charging_Stations_Near_Home"] + df["Charging_Stations_Near_Work"]
    )

    # --- Commute cost proxy -----------------------------------------------
    # Why: EVs pay off most for people who drive a lot; commute distance is
    # a direct proxy for how much fuel/energy cost a switch would save.
    df["Commute_Intensity"] = pd.cut(
        df["Daily_Commute_km"],
        bins=[-0.1, 15, 35, 60, 1000],
        labels=["Short", "Moderate", "Long", "Very_Long"],
    ).astype(str)

    # --- Affordability x subsidy interaction -----------------------------
    # Why: a subsidy likely matters most to buyers who are otherwise
    # borderline on affordability; an interaction term lets simpler models
    # (logistic regression) capture that combined effect directly.
    # Fixed threshold (train median), for the same reason as Income_Band above.
    TRAIN_MEDIAN_INCOME = 84880.0
    df["High_Income"] = (df["Annual_Income_USD"] >= TRAIN_MEDIAN_INCOME).astype(int)
    df["Subsidy_Yes"] = (df["Subsidy_Available"] == "Yes").astype(int)
    df["Subsidy_x_MidLowIncome"] = df["Subsidy_Yes"] * (1 - df["High_Income"])

    # --- Cars per household proxy -----------------------------------------
    # Why: households with more cars may be more likely to add/replace one
    # with an EV rather than depend on a single vehicle.
    df["Multi_Car_Household"] = (df["Number_of_Cars_Owned"] >= 2).astype(int)

    return df


ENGINEERED_NUMERIC = ["Total_Charging_Access"]
ENGINEERED_BINARY = ["High_Income", "Subsidy_Yes", "Subsidy_x_MidLowIncome", "Multi_Car_Household"]
ENGINEERED_CATEGORICAL = ["Age_Group", "Income_Band", "Commute_Intensity"]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/EV-Purchase-Prediction/src")
    from data_preprocessing import load_raw_data, encode_target

    train, test = load_raw_data(
        "/home/claude/EV-Purchase-Prediction/data/train.csv",
        "/home/claude/EV-Purchase-Prediction/data/test.csv",
    )
    train = encode_target(train)
    train_fe = add_engineered_features(train)
    print(train_fe[["Age", "Age_Group", "Income_Band", "Total_Charging_Access",
                     "Commute_Intensity", "Subsidy_x_MidLowIncome"]].head())
    print(train_fe.shape)
