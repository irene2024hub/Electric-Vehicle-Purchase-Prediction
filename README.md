# Electric Vehicle Purchase Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Models](https://img.shields.io/badge/Models-LightGBM%20%7C%20XGBoost%20%7C%20CatBoost-orange)
![Metric](https://img.shields.io/badge/Metric-ROC--AUC-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

An end-to-end machine learning project that predicts the likelihood of a customer indicating they would purchase an electric vehicle (EV). The project covers exploratory data analysis, feature engineering, model comparison, hyperparameter tuning, ensembling, explainability, calibration, error analysis, fairness evaluation, and an interactive Streamlit dashboard.

**Final blended out-of-fold ROC-AUC: 0.9419**

The final model combines LightGBM and XGBoost.

> **Important:** This is a Kaggle-style competition dataset, not verified real-world purchase data. The results demonstrate a machine learning workflow and should not be interpreted as proof of actual consumer behaviour or causal relationships.

---

## Interactive Dashboard

The Streamlit dashboard allows users to explore model performance, understand important features, and generate predictions for hypothetical customer profiles.

### 1. Dashboard Overview

![Dashboard Overview](outputs/figures/dashboard_overview.png)

**Insight:** Five models were compared using stratified cross-validation. XGBoost (0.9415) and LightGBM (0.9414) achieved the strongest baseline ROC-AUC scores, ahead of CatBoost, Logistic Regression and Random Forest. Blending the two tuned models gives the final out-of-fold ROC-AUC of 0.9419.

### 2. Customer Explorer: Live Prediction

![EV Purchase Prediction](outputs/figures/dashboard_prediction.png)

**Insight:** Users enter customer characteristics and receive an estimated probability of EV purchase. In the example shown, the predicted probability is 5.5%, a lower-likelihood result. This is a model estimate, not a guarantee of individual behaviour.

### 3. Model Insights

![Model Insights](outputs/figures/dashboard_insights.png)

**Insight:** Feature importance differs by model. XGBoost relies heavily on subsidy-related features, while LightGBM ranks annual income, daily commute and age highest on split counts. Charging-station counts and demographics contribute far less.

### 4. Business Insights

![Business Insights](outputs/figures/dashboard_business.png)

**Insight:** Subsidy availability is the strongest association with purchase willingness (about 27.5% with a subsidy versus 0.6% without). Environmental concern rises from about 0.6% to 51.8%, and range anxiety is strongly linked to lower purchase likelihood. This is consistent with upfront cost being a key barrier (see the economic case below).

---

## Business Problem

Can machine learning identify customers who are more likely to indicate that they would purchase an EV based on their demographic, financial, vehicle, environmental, and charging-access characteristics?

Potential applications include:

- **EV manufacturers and dealerships:** Prioritise customer outreach.
- **Marketing teams:** Explore customer segments for targeted campaigns.
- **EV financing providers:** Understand potential interest in EV financing.
- **Policy and sustainability teams:** Investigate patterns relevant to EV adoption.

The model supports exploratory analysis and should not be used alone for consequential decisions or infrastructure investment.

## Why EV Adoption Matters: The Economic Case

Understanding who is likely to buy an EV matters because EV ownership can offer real economic advantages for households and fleets. The main ones are:

- **Lower running costs:** Electricity is typically cheaper per kilometre than petrol or diesel, so drivers with long daily commutes tend to benefit most.
- **Lower maintenance costs:** EVs have fewer moving parts than combustion-engine vehicles (no oil changes, fewer fluids and no exhaust system), which generally means less routine servicing.
- **Government incentives:** Purchase subsidies, tax credits and reduced registration or road fees can lower the effective upfront price.
- **Protection from fuel price swings:** Owners rely less on volatile fuel markets, especially where they can charge at home.
- **Total cost of ownership:** A higher purchase price is often offset over time by lower fuel and maintenance spending, depending on usage and local prices.

**Trade-offs:** Upfront prices can be higher, and charging access (including installing a home charger) affects both convenience and cost. These are the same barriers reflected in this dataset.

### How the dataset reflects the economics

The patterns in this project fit the economic picture, although they are observed associations and not proof of cause:

- **Subsidy availability** shows the largest gap in the data: the purchase-willingness rate is **27.5% with a subsidy versus 0.6% without**, which suggests that lowering the upfront cost matters a great deal.
- **Annual income** ranks among the important model features, consistent with affordability affecting willingness to buy.
- **Daily commute distance** and **charging access** relate to how much a driver could save on running costs and how practical charging is.
- **Range anxiety** captures a practical concern that can offset the financial benefits.

### Business implications

- **EV manufacturers and dealerships:** Highlight total cost of ownership, not only the sticker price, when targeting high-commute or high-probability customers.
- **Financing providers:** Design loan products that spread the upfront cost, since subsidies and price appear to be strong barriers.
- **Policy and sustainability teams:** Use the subsidy finding to explore where incentives or charging access could widen adoption.

> *Economic advantages vary by country, vehicle model, electricity and fuel prices, and available incentives. Evaluate local figures before drawing conclusions for a specific market.*

---

## Dataset

## Handling Class Imbalance

The dataset is imbalanced: about **17.5% of customers are "Yes"** and **82.5% are "No"**.

![Target Distribution](outputs/figures/target_distribution.png)

- **Why it matters:** Predicting "No" for every customer would be about 82.5% accurate while finding no buyers at all, so accuracy is misleading.
- **What was done:** ROC-AUC was used as the metric because it measures how well buyers are ranked above non-buyers, and stratified cross-validation kept the 17.5% / 82.5% split in every fold.
- **Decision:** Class weighting did not improve ROC-AUC (0.9417 versus 0.9418), so the original class distribution was kept.

- `train.csv`: 668,665 rows and 15 columns, including the target.
- `test.csv`: 286,571 rows and 14 feature columns, without target labels.
- The source data was checked for missing values and duplicate rows or IDs.
- Features include age, annual income, daily commute distance, vehicle ownership, charging-station access, environmental concern, gender, city type, home charging, subsidy availability, and range anxiety.

### Target Variable and Class Imbalance

The target variable is `Will_Buy_EV`:

- `Yes = 1`: Indicates willingness to purchase an EV.
- `No = 0`: Indicates unwillingness to purchase an EV.

The target distribution is approximately 17.5% Yes and 82.5% No.

Class weighting was tested but did not improve ROC-AUC:

| Approach | ROC-AUC |
|---|---:|
| Without class weighting | 0.9418 |
| With class weighting | 0.9417 |

The final approach retained the original class distribution.

ROC-AUC measures how well a model ranks positive cases above negative cases across different classification thresholds.

## Exploratory Data Analysis

Selected observed purchase-willingness rates:

| Feature | Observed rate |
|---|---:|
| Subsidy available | 27.5% |
| No subsidy available | 0.6% |
| Environmental concern level 5 | 51.8% |
| Environmental concern level 1 | 0.6% |
| Low range anxiety | 18.9% |
| High range anxiety | 0.1% |

These figures represent associations in the dataset, not causal effects.

Additional charts (target distribution, numeric and categorical distributions, purchase rate by driver, and the correlation heatmap) are saved in `outputs/figures/`.

## Feature Engineering

New features were created from existing row-level information without using the target variable:

- `Age_Group`: Groups customers by age.
- `Income_Band`: Groups customers by income.
- `Total_Charging_Access`: Combines home and work charging-station counts.
- `Commute_Intensity`: Groups daily commute distance.
- `Subsidy_x_MidLowIncome`: Tests the interaction between subsidy availability and below-median income.
- `Multi_Car_Household`: Identifies households owning two or more cars.

## Model Development and Validation

Stratified 3-fold cross-validation was used to preserve the class distribution in each fold and manage computational costs.

### Baseline Model Comparison

| Model | Mean CV ROC-AUC | Standard deviation |
|---|---:|---:|
| XGBoost | 0.9415 | 0.0006 |
| LightGBM | 0.9414 | 0.0006 |
| CatBoost | 0.9404 | 0.0007 |
| Logistic Regression | 0.9383 | 0.0007 |
| Random Forest | 0.9349 | 0.0005 |

Optuna was used to tune LightGBM and XGBoost, optimising ROC-AUC. Each tuned model achieved a reported validation AUC of approximately 0.9417.

### Ensembling and Final Performance

The final blend combined:

- 48% LightGBM predictions
- 52% XGBoost predictions

The blend achieved a reported out-of-fold ROC-AUC of **0.9419**, a small improvement over the individual models.

![ROC Model Comparison Curves](outputs/figures/roc_model_comparison_curves.png)

A ROC-AUC of 0.9419 indicates strong ranking performance on the evaluation data. It does not mean that 94.19% of predictions are correct. 

- **Strong separation from chance:** All models sit far above the random-guess diagonal (AUC 0.50), with AUC scores between 0.9349 and 0.9418, so they rank likely EV buyers well above non-buyers.

- **Tuning gave a small but consistent gain:** Tuned LightGBM and tuned XGBoost both reach an AUC of 0.9418, ahead of their untuned versions (0.9414 and 0.9415) and of CatBoost (0.9404).

- **Random Forest lags slightly:** Its curve dips below the others between roughly 0.15 and 0.35 false positive rate (AUC 0.9349), while the rest of the models are tightly clustered, which suggests the signal in the data is strong and easy to learn.

- **Strong separation from chance:** All models sit far above the random-guess diagonal (AUC 0.50), with AUC scores between 0.9349 and 0.9418, so they rank likely EV buyers well above non-buyers.
- **Tuning gave a small but consistent gain:** Tuned LightGBM and tuned XGBoost both reach an AUC of 0.9418, ahead of their untuned versions (0.9414 and 0.9415) and of CatBoost (0.9404).
- **Random Forest lags slightly:** Its curve dips below the others between roughly 0.15 and 0.35 false positive rate (AUC 0.9349), while the rest of the models are tightly clustered, which suggests the signal in the data is strong and easy to learn.

## Model Explainability

The analyses (feature importance and SHAP) highlighted subsidy-related features, environmental concern, annual income, and range anxiety. Feature importance describes model behaviour and associations; it does not establish causation. The SHAP summary and feature importance figures are saved in `outputs/figures/`.

## Model Calibration

![Model Calibration ](outputs/figures/calibration_curve.png)

Predicted probabilities from tuned LightGBM were compared with observed frequencies across 10 quantile bins on a hold-out set. Calibration was reported as close to the ideal diagonal, so a separate isotonic or Platt calibration step was not applied. outputs/figures/calibration_curve.png`.

## Error Analysis and Fairness

The reported hold-out analysis found:

- 74.8% of predictions were high-confidence and correct.
- 2.5% were high-confidence but incorrect.
- 6.1% fell within the uncertain probability range of 0.40–0.60.

### Performance by City Type

- Urban ROC-AUC: 0.945
- Rural ROC-AUC: 0.937

The reason for this difference has not been established and requires further investigation.

### Performance by Gender

- Female ROC-AUC: 0.9418
- Male ROC-AUC: 0.9419
- Other ROC-AUC: 0.9294

The Other group had a substantially smaller sample size, approximately 1,034 customers in the reported analysis. Its performance estimate is therefore less certain. Further subgroup evaluation is necessary before consequential use.

## Limitations

- The dataset is a single snapshot and does not confirm actual EV purchases or track behaviour over time.
- Three-fold cross-validation was used to reduce computing time.
- CatBoost was trained with fewer iterations and received less extensive tuning.
- Fairness estimates for the smaller gender group are less certain.
- Observed relationships do not establish causal effects.
- Real-world deployment would require validation on representative data and ongoing performance monitoring.

## Deployment

The Streamlit application is `app.py` in the project root. It allows users to enter hypothetical customer details, view estimated EV purchase probabilities, and explore model performance and explainability information.

## Project Structure

```text
EV-Purchase-Prediction/
├── data/
├── notebooks/
│   └── EV_Purchase_Prediction.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── tune.py
│   ├── generate_tuned_oof.py
│   ├── imbalance_check.py
│   ├── evaluate.py
│   ├── make_plots.py
│   ├── make_model_plots.py
│   └── predict.py
├── models/
├── outputs/
│   ├── submission.csv
│   ├── feature_importance.csv
│   ├── model_comparison.csv
│   └── figures/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Technologies Used

Python 3.12, Pandas, NumPy, scikit-learn, LightGBM, XGBoost, CatBoost, SHAP, Optuna, Matplotlib, Seaborn, Plotly, and Streamlit.

## How to Run

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

> **Important:** The saved models were trained with **XGBoost 3.4.0, LightGBM 4.7.0 and scikit-learn 1.9.0**. Use these exact versions (pinned in `requirements.txt`) to avoid model-loading errors such as `XGBoostError: input stream corrupted`.

```bash
python -m streamlit run app.py
```

### Training Workflow


```bash
# Compare baseline models
python src/train.py --model LightGBM

# Tune hyperparameters
python src/tune.py --model lightgbm --n_trials 20

# Generate out-of-fold predictions
python src/generate_tuned_oof.py --model lightgbm_tuned

# Run evaluation and analysis
python src/evaluate.py

# Train the final blend and generate submission.csv
python src/predict.py
```

## Future Improvements

- **Generative AI chatbot (in progress):** Build a chatbot app on top of this project so users can ask questions about EV purchase insights in plain language.
- Evaluate with 5-fold cross-validation and tune CatBoost more extensively.
- Test predictions against real-world EV purchase outcomes.
- Explore additional market and geographic data.
- Expand fairness evaluation using larger, representative samples.

## Author

**Irene Ayakazi**: Data Scientist and Machine Learning Engineer | Generative AI Learning

[LinkedIn](https://www.linkedin.com/in/irene-ayakazi)