"""
data_preprocessing.py
----------------------
Loads the raw EV purchase prediction data and performs light, leakage-safe
cleaning: type fixes, target encoding, and column typing.

This module intentionally does NOT do any statistical fitting (e.g. scaling,
imputation) -- that belongs inside the modelling pipeline (see train.py) so
that every transform is fit only on training folds and never on validation
or test data.
"""

import pandas as pd

ID_COL = "id"
TARGET_COL = "Will_Buy_EV"

NUMERIC_FEATURES = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]


def load_raw_data(train_path: str, test_path: str):
    """Load the raw train/test CSVs exactly as provided by the competition."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def encode_target(train: pd.DataFrame) -> pd.DataFrame:
    """Convert the Yes/No target into a 0/1 integer column.

    'Will_Buy_EV' is stored as text ('Yes'/'No') in the raw file. Scikit-learn
    and the gradient boosting libraries need a numeric target, so we map
    Yes -> 1, No -> 0. This mapping is fixed and does not depend on the data,
    so it introduces no leakage.
    """
    train = train.copy()
    mapping = {"Yes": 1, "No": 0}
    unexpected = set(train[TARGET_COL].unique()) - set(mapping.keys())
    if unexpected:
        raise ValueError(f"Unexpected target values found: {unexpected}")
    train["target"] = train[TARGET_COL].map(mapping).astype(int)
    return train


def basic_data_audit(df: pd.DataFrame, name: str = "dataset") -> pd.DataFrame:
    """Return a tidy audit table: dtype, missing %, unique count per column."""
    audit = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_count": df.isnull().sum(),
            "missing_pct": (df.isnull().sum() / len(df) * 100).round(3),
            "n_unique": df.nunique(),
        }
    )
    audit.index.name = "column"
    audit.attrs["dataset_name"] = name
    return audit


if __name__ == "__main__":
    train, test = load_raw_data(
        "/home/claude/EV-Purchase-Prediction/data/train.csv",
        "/home/claude/EV-Purchase-Prediction/data/test.csv",
    )
    train = encode_target(train)
    print(basic_data_audit(train, "train"))
    print(train["target"].value_counts(normalize=True))
