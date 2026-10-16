"""
Streamlit dashboard for the EV Purchase Prediction project.

Run with: streamlit run app/app.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from src.feature_engineering import add_engineered_features
st.set_page_config(page_title="EV Purchase Prediction", layout="wide")

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"


@st.cache_resource
def load_artifacts():
    with open(MODELS_DIR / "lightgbm_final.pkl", "rb") as f:
        lgb_model = pickle.load(f)
    with open(MODELS_DIR / "xgboost_final.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open(MODELS_DIR / "xgb_columns.pkl", "rb") as f:
        xgb_columns = pickle.load(f)
    with open(MODELS_DIR / "lgb_categories.pkl", "rb") as f:
        lgb_categories = pickle.load(f)
    return lgb_model, xgb_model, xgb_columns, lgb_categories


LGB_WEIGHT, XGB_WEIGHT = 0.48, 0.52

st.title("🔋 Predicting Electric Vehicle Purchases")
st.caption(
    "A portfolio ML case study — predicts the probability a customer will purchase an EV, "
    "based on a Kaggle-style competition dataset. Not a validated real-world model."
)

tab_overview, tab_predict, tab_insights, tab_business = st.tabs(
    ["Overview", "EV Purchase Prediction", "Model Insights", "Business Insights"]
)

# ---------------------------------------------------------------- Overview
with tab_overview:
    col1, col2, col3 = st.columns(3)
    col1.metric("Training rows", "668,665")
    col2.metric("Test rows", "286,571")
    col3.metric("Target: Yes rate", "17.5%")

    st.subheader("Project objective")
    st.write(
        "Predict the probability that a customer will purchase an Electric Vehicle "
        "(`Will_Buy_EV`), evaluated by ROC-AUC. The final model is a 48/52 blend of a "
        "tuned LightGBM and a tuned XGBoost classifier."
    )

    st.subheader("Model performance")
    st.metric("Out-of-fold ROC-AUC (blended model)", "0.9419")
    try:
        st.image(str(OUTPUTS_DIR / "figures" / "model_comparison.png"), caption="Model comparison (3-fold CV)")
    except Exception:
        st.info("Run make_model_plots.py to generate this figure.")

# ---------------------------------------------------------- Customer input
with tab_predict:
    st.subheader("Customer Explorer")
    st.write("Enter a hypothetical customer's characteristics to see the predicted EV purchase probability.")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 18, 75, 40)
        income = st.number_input("Annual Income (USD)", 10000, 250000, 80000, step=1000)
        commute = st.slider("Daily Commute (km)", 0.0, 120.0, 30.0)
        cars_owned = st.slider("Number of Cars Owned", 0, 5, 1)
    with c2:
        charge_home = st.slider("Charging Stations Near Home", 0, 15, 5)
        charge_work = st.slider("Charging Stations Near Work", 0, 20, 7)
        env_concern = st.select_slider("Environmental Concern Level (1-5)", options=[1, 2, 3, 4, 5], value=3)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with c3:
        city_type = st.selectbox("City Type", ["Urban", "Suburban", "Rural"])
        car_type = st.selectbox("Current Car Type", ["Sedan", "SUV", "Hatchback", "Truck"])
        home_charging = st.selectbox("Home Charging Possible", ["Yes", "No"])
        subsidy = st.selectbox("Subsidy Available", ["Yes", "No"])
        range_anxiety = st.selectbox("Range Anxiety Level", ["Low", "Medium", "High"])

    if st.button("Predict EV Purchase Probability", type="primary"):
        row = pd.DataFrame([{
            "Age": age, "Annual_Income_USD": income, "Daily_Commute_km": commute,
            "Number_of_Cars_Owned": cars_owned, "Charging_Stations_Near_Home": charge_home,
            "Charging_Stations_Near_Work": charge_work, "Environmental_Concern_Level": env_concern,
            "Gender": gender, "City_Type": city_type, "Current_Car_Type": car_type,
            "Home_Charging_Possible": home_charging, "Subsidy_Available": subsidy,
            "Range_Anxiety_Level": range_anxiety,
        }])
        row_fe = add_engineered_features(row)

        try:
            lgb_model, xgb_model, xgb_columns, lgb_categories = load_artifacts()
            cat_cols = list(lgb_categories.keys())

            row_lgb = row_fe.copy()
            for c in cat_cols:
                row_lgb[c] = pd.Categorical(row_lgb[c], categories=lgb_categories[c])
            lgb_features = [c for c in lgb_model.feature_name_]
            lgb_pred = lgb_model.predict_proba(row_lgb[lgb_features])[:, 1][0]

            row_xgb = pd.get_dummies(row_fe, columns=cat_cols).reindex(columns=xgb_columns, fill_value=0)
            xgb_pred = xgb_model.predict_proba(row_xgb)[:, 1][0]

            proba = LGB_WEIGHT * lgb_pred + XGB_WEIGHT * xgb_pred

            st.metric("Predicted probability of EV purchase", f"{proba:.1%}")
            if proba >= 0.5:
                st.success("This customer profile shows a higher predicted likelihood of purchasing an EV.")
            elif proba >= 0.2:
                st.info("This customer profile shows a moderate predicted likelihood of purchasing an EV.")
            else:
                st.warning("This customer profile shows a lower predicted likelihood of purchasing an EV.")
            st.caption(
                "This is a model-based estimate from a competition dataset, not a guarantee of "
                "individual behavior."
            )
        except FileNotFoundError:
            st.error("Model artifacts not found. Run `python src/predict.py` first to train and save the models.")

# --------------------------------------------------------------- Insights
with tab_insights:
    st.subheader("Feature Importance")
    try:
        st.image(str(OUTPUTS_DIR / "figures" / "feature_importance.png"))
        st.image(str(OUTPUTS_DIR / "figures" / "shap_summary.png"), caption="SHAP summary (sample of 2,000 hold-out rows)")
    except Exception:
        st.info("Run make_model_plots.py to generate these figures.")

    st.subheader("ROC Curve")
    try:
        st.image(str(OUTPUTS_DIR / "figures" / "roc_curve.png"))
    except Exception:
        pass

    st.subheader("Calibration")
    try:
        st.image(str(OUTPUTS_DIR / "figures" / "calibration_curve.png"))
    except Exception:
        pass

    st.subheader("Model Comparison")
    try:
        comp = pd.read_csv(OUTPUTS_DIR / "model_comparison.csv")
        st.dataframe(comp, use_container_width=True)
    except Exception:
        pass

# --------------------------------------------------------------- Business
with tab_business:
    st.subheader("Key drivers of predicted EV purchase")
    st.markdown(
        """
        - **Subsidy availability** is the strongest single driver: observed purchase rate is
          ~27.5% when a subsidy is available vs. ~0.6% when it isn't.
        - **Environmental concern level** shows a near-linear relationship with purchase rate,
          from ~0.6% (lowest concern) to ~51.8% (highest concern).
        - **Range anxiety** is strongly negatively associated with purchase likelihood.
        - Charging-station counts, vehicle type, and demographics contribute far less to the
          model's predictions than these three factors.
        """
    )
    st.subheader("Suggested business applications")
    st.markdown(
        """
        - **Marketing**: prioritize outreach toward higher predicted-probability segments
        - **Sales**: score and rank inbound leads
        - **Financing**: surface likely candidates for EV loan products
        - **Sustainability planning**: understand which population segments may need more
          support (subsidies, charging access, range-anxiety education) to adopt EVs

        These are model-based suggestions from a competition dataset and would need
        validation against real outcomes before any operational use.
        """
    )
