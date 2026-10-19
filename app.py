"""
Professional Streamlit dashboard for the EV Purchase Prediction project.
Run with: python -m streamlit run app.py
"""
import pickle
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "models").exists() else HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from src.feature_engineering import add_engineered_features

st.set_page_config(page_title="ELECTRIC VEHICLE PURCHASE PREDICTION", page_icon="🔋", layout="wide")

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"
REQUIRED_MODEL_FILES = [
    "lightgbm_final.pkl", "xgboost_final.pkl", "xgb_columns.pkl", "lgb_categories.pkl",
]
LGB_WEIGHT, XGB_WEIGHT = 0.48, 0.52

NAVY, TEAL, GREEN, SLATE = "#0B1F3A", "#0E9F9A", "#22C55E", "#64748B"

# ------------------------------------------------------------------ Styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, #EAF2FB 0%, #F6F9FC 45%, #FFFFFF 100%);
    }}
    .block-container {{ padding-top: 1.5rem; max-width: 1300px; }}

    .hero {{
        background: linear-gradient(120deg, {NAVY} 0%, #12426B 55%, {TEAL} 100%);
        padding: 2rem 2.2rem; border-radius: 18px; color: white;
        box-shadow: 0 10px 30px rgba(11,31,58,.25); margin-bottom: 1.2rem;
    }}
    .hero h1 {{ color: white; margin: 0; font-size: 2.2rem; }}
    .hero p {{ color: #CFE3F5; margin: .4rem 0 0 0; font-size: .98rem; }}

    div[data-testid="stMetric"] {{
        background: white; border-radius: 14px; padding: 1rem 1.2rem;
        border-top: 4px solid {TEAL};
        box-shadow: 0 4px 14px rgba(15,23,42,.08);
    }}
    div[data-testid="stMetricLabel"] {{ color: {SLATE}; }}
    div[data-testid="stMetricValue"] {{ color: {NAVY}; font-weight: 700; }}

    .card {{
        background: white; border-radius: 14px; padding: 1.3rem 1.5rem;
        box-shadow: 0 4px 14px rgba(15,23,42,.08); margin-bottom: 1rem;
    }}
    .card h3 {{ color: {NAVY}; margin-top: 0; }}

    button[data-baseweb="tab"] {{ font-weight: 600; font-size: 1rem; }}
    .stButton > button {{
        background: linear-gradient(90deg, {TEAL}, {GREEN}); color: white;
        border: none; border-radius: 10px; padding: .6rem 1.6rem; font-weight: 600;
    }}
    .stButton > button:hover {{ filter: brightness(1.08); color: white; }}
    h2, h3 {{ color: {NAVY}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🔋 Predicting Electric Vehicle Purchases</h1>
        <p>A portfolio ML case study: estimating the probability that a customer will buy an EV,
        using a competition dataset. Not a validated real-world model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ Helpers
def missing_model_files():
    return [f for f in REQUIRED_MODEL_FILES if not (MODELS_DIR / f).exists()]


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


def show_figure(filename, caption=None):
    """Show a saved PNG if it exists; never raise an error."""
    try:
        path = FIG_DIR / filename
        if path.exists():
            st.image(str(path), caption=caption, use_container_width=True)
            return True
    except Exception:
        pass
    return False


def gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 44, "color": NAVY}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": TEAL},
            "steps": [
                {"range": [0, 20], "color": "#FDE2E2"},
                {"range": [20, 50], "color": "#FEF3C7"},
                {"range": [50, 100], "color": "#D1FAE5"},
            ],
        },
    ))
    fig.update_layout(height=290, margin=dict(l=20, r=20, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def importance_chart(names, values, title, color, top_n=12):
    df = pd.DataFrame({"Feature": list(names), "Importance": list(values)})
    total = df["Importance"].sum()
    if total > 0:
        df["Importance"] = df["Importance"] / total * 100
    df = df.sort_values("Importance", ascending=False).head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=df["Importance"], y=df["Feature"], orientation="h",
        marker_color=color,
        text=[f"{v:.1f}%" for v in df["Importance"]], textposition="outside",
    ))
    fig.update_layout(
        title=title, height=460, plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=40, t=50, b=20),
        xaxis_title="Relative importance (%)",
    )
    return fig


tab_overview, tab_predict, tab_insights, tab_business = st.tabs(
    ["📊 Overview", "🎯 EV Purchase Prediction", "🔍 Model Insights", "💼 Business Insights"]
)

# ----------------------------------------------------------------- Overview
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training rows", "668,665")
    c2.metric("Test rows", "286,571")
    c3.metric("Target: Yes rate", "17.5%")
    c4.metric("OOF ROC-AUC (blend)", "0.9419")

    st.write("")
    left, right = st.columns([1, 1.4])
    with left:
        st.markdown(
            """
            <div class="card">
            <h3>Project objective</h3>
            Predict the probability that a customer will purchase an Electric Vehicle
            (<code>Will_Buy_EV</code>), evaluated by ROC-AUC.<br><br>
            The final model is a <b>48/52 blend</b> of a tuned <b>LightGBM</b> and a tuned
            <b>XGBoost</b> classifier.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("<div class='card'><h3>Model performance</h3>", unsafe_allow_html=True)
        if not show_figure("model_comparison.png", "Model comparison (3-fold CV)"):
            st.info("Model comparison figure not found. Run make_model_plots.py to generate it.")
        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------- Prediction
with tab_predict:
    st.markdown("### Customer Explorer")
    st.write("Enter a hypothetical customer's characteristics to see the predicted EV purchase probability.")

    form_col, result_col = st.columns([1.6, 1])

    with form_col:
        a, b = st.columns(2)
        with a:
            age = st.slider("Age", 18, 75, 40)
            income = st.number_input("Annual Income (USD)", 10000, 250000, 80000, step=1000)
            commute = st.slider("Daily Commute (km)", 0.0, 120.0, 30.0)
            cars_owned = st.slider("Number of Cars Owned", 0, 5, 1)
            charge_home = st.slider("Charging Stations Near Home", 0, 15, 5)
            charge_work = st.slider("Charging Stations Near Work", 0, 20, 7)
            env_concern = st.select_slider("Environmental Concern Level (1-5)",
                                           options=[1, 2, 3, 4, 5], value=3)
        with b:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            city_type = st.selectbox("City Type", ["Urban", "Suburban", "Rural"])
            car_type = st.selectbox("Current Car Type", ["Sedan", "SUV", "Hatchback", "Truck"])
            home_charging = st.selectbox("Home Charging Possible", ["Yes", "No"])
            subsidy = st.selectbox("Subsidy Available", ["Yes", "No"])
            range_anxiety = st.selectbox("Range Anxiety Level", ["Low", "Medium", "High"])
        predict_clicked = st.button("Predict EV Purchase Probability")

    with result_col:
        st.markdown("#### Prediction result")
        if predict_clicked:
            missing = missing_model_files()
            if missing:
                st.error("Model files missing from the `models/` folder:\n\n"
                         + "\n".join(f"- `{m}`" for m in missing))
            else:
                row = pd.DataFrame([{
                    "Age": age,
                    "Annual_Income_USD": income,
                    "Daily_Commute_km": commute,
                    "Number_of_Cars_Owned": cars_owned,
                    "Charging_Stations_Near_Home": charge_home,
                    "Charging_Stations_Near_Work": charge_work,
                    "Environmental_Concern_Level": env_concern,
                    "Gender": gender,
                    "City_Type": city_type,
                    "Current_Car_Type": car_type,
                    "Home_Charging_Possible": home_charging,
                    "Subsidy_Available": subsidy,
                    "Range_Anxiety_Level": range_anxiety,
                }])
                try:
                    row_fe = add_engineered_features(row)
                    lgb_model, xgb_model, xgb_columns, lgb_categories = load_artifacts()
                    cat_cols = list(lgb_categories.keys())

                    row_lgb = row_fe.copy()
                    for c in cat_cols:
                        row_lgb[c] = pd.Categorical(row_lgb[c], categories=lgb_categories[c])
                    lgb_pred = lgb_model.predict_proba(
                        row_lgb[list(lgb_model.feature_name_)])[:, 1][0]

                    row_xgb = (pd.get_dummies(row_fe, columns=cat_cols)
                               .reindex(columns=xgb_columns, fill_value=0))
                    xgb_pred = xgb_model.predict_proba(row_xgb)[:, 1][0]

                    proba = LGB_WEIGHT * lgb_pred + XGB_WEIGHT * xgb_pred

                    st.plotly_chart(gauge(proba), use_container_width=True)
                    if proba >= 0.5:
                        st.success("Higher predicted likelihood of purchasing an EV.")
                    elif proba >= 0.2:
                        st.info("Moderate predicted likelihood of purchasing an EV.")
                    else:
                        st.warning("Lower predicted likelihood of purchasing an EV.")
                    st.caption("Model-based estimate from a competition dataset, "
                               "not a guarantee of individual behaviour.")
                except Exception as e:
                    st.error(f"Prediction failed: {type(e).__name__}: {e}")
        else:
            st.info("Fill in the customer profile and click **Predict**.")

# ----------------------------------------------------------------- Insights
with tab_insights:
    st.markdown("### Feature Importance")
    try:
        lgb_m, xgb_m, xgb_cols, _lgb_cats = load_artifacts()
        ic1, ic2 = st.columns(2)
        with ic1:
            st.plotly_chart(
                importance_chart(lgb_m.feature_name_, lgb_m.feature_importances_,
                                 "LightGBM", TEAL),
                use_container_width=True)
        with ic2:
            st.plotly_chart(
                importance_chart(xgb_cols, xgb_m.feature_importances_,
                                 "XGBoost", NAVY),
                use_container_width=True)
    except Exception as e:
        st.info(f"Live feature importance is unavailable ({type(e).__name__}). "
                "Saved figures are shown below where available.")

    st.markdown("### Saved Figures")
    saved_figures = [
        ("feature_importance.png", None),
        ("shap_summary.png", "SHAP summary (sample of 2,000 hold-out rows)"),
        ("roc_curve.png", "ROC curve"),
        ("calibration_curve.png", "Calibration curve"),
    ]
    shown_any = False
    for fname, cap in saved_figures:
        if show_figure(fname, cap):
            shown_any = True
    if not shown_any:
        st.info("No saved figures found. Run make_model_plots.py to generate them.")

    st.markdown("### Model Comparison")
    try:
        comp = pd.read_csv(OUTPUTS_DIR / "model_comparison.csv")
        st.dataframe(comp, use_container_width=True)
    except Exception:
        st.info("model_comparison.csv not found in the outputs folder.")

# ----------------------------------------------------------------- Business
with tab_business:
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown(
            """
            <div class="card">
            <h3>Key drivers of predicted EV purchase</h3>
            <ul>
            <li><b>Subsidy availability</b> is the strongest single driver: observed purchase rate
            is ~27.5% with a subsidy vs. ~0.6% without.</li>
            <li><b>Environmental concern</b> rises near-linearly, from ~0.6% (lowest concern)
            to ~51.8% (highest).</li>
            <li><b>Range anxiety</b> is strongly negatively associated with purchase likelihood.</li>
            <li>Charging-station counts, vehicle type and demographics contribute far less.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        try:
            sub_fig = go.Figure(go.Bar(
                x=["Subsidy available", "No subsidy"], y=[27.5, 0.6],
                marker_color=[TEAL, "#CBD5E1"], text=["27.5%", "0.6%"], textposition="outside",
            ))
            sub_fig.update_layout(title="Observed purchase rate by subsidy", height=330,
                                  yaxis_title="Purchase rate (%)", plot_bgcolor="white",
                                  paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=20))
            st.plotly_chart(sub_fig, use_container_width=True)
        except Exception:
            pass

    st.markdown(
        """
        <div class="card">
        <h3>Suggested business applications</h3>
        <ul>
        <li><b>Marketing:</b> prioritise outreach toward higher predicted-probability segments</li>
        <li><b>Sales:</b> score and rank inbound leads</li>
        <li><b>Financing:</b> surface likely candidates for EV loan products</li>
        <li><b>Sustainability planning:</b> identify segments needing support (subsidies,
        charging access, range-anxiety education)</li>
        </ul>
        <i>These are model-based suggestions from a competition dataset and would need
        validation against real outcomes before operational use.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------- Credit footer
st.markdown(
    f"""
    <div style="
        background: linear-gradient(120deg, {NAVY} 0%, #12426B 55%, {TEAL} 100%);
        border-radius: 14px; padding: 1.2rem 1.6rem; margin-top: 1.5rem;
        text-align: center; box-shadow: 0 6px 18px rgba(11,31,58,.2);">
        <div style="color:#CFE3F5; font-size:.8rem; letter-spacing:.12em;
                    text-transform:uppercase;">Model built by</div>
        <div style="color:white; font-size:1.35rem; font-weight:700; margin:.2rem 0;">
            Irene Ayakazi
        </div>
        <div style="color:#CFE3F5; font-size:.95rem;">
            Data Scientist &nbsp;•&nbsp; Machine Learning Engineer &nbsp;•&nbsp; Generative AI Learning
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)