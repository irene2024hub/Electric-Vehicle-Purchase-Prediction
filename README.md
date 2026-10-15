# Predicting Electric Vehicle Purchases

![Python](https://img.shields.io/badge/Python-3.12-blue)
![ML](https://img.shields.io/badge/Models-LightGBM%20%7C%20XGBoost%20%7C%20CatBoost-orange)
![Metric](https://img.shields.io/badge/Metric-ROC--AUC-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## Overview

An end-to-end, portfolio-grade machine learning case study that predicts
the probability a customer will purchase an Electric Vehicle (EV), built
around a Kaggle-style binary classification competition. The project
covers the full workflow: business framing, data auditing, EDA, feature
engineering, model comparison, hyperparameter tuning, explainability,
calibration, error analysis, fairness review, ensembling, and a deployed
Streamlit dashboard.

**Final cross-validated performance: ROC-AUC ≈ 0.9419** (blended LightGBM +
XGBoost, out-of-fold estimate).

## Business Problem

> Can machine learning identify customers who are more likely to purchase
> an electric vehicle, based on their demographic, financial, vehicle,
> environmental, and infrastructure-access characteristics?

## Who Can Benefit from This Project?

This project helps predict which people are more likely to buy an electric vehicle (EV). It can help different groups make better decisions.

- **EV Manufacturers and Car Dealers:** Identify people who may be interested in buying an electric car and focus their sales efforts on those customers.

- **EV Loan and Financing Companies:** Understand which customers may be interested in getting a loan to buy an electric vehicle.

- **Charging Station Providers:** Identify areas where more people may want to use electric cars and plan where to build charging stations.

- **Marketing Teams:** Find groups of people who are more likely to buy electric vehicles and create advertisements that match their needs and interests.

- **Government and Environmental Planners:** Understand which groups of people may need support, such as financial assistance or more charging stations, to make electric vehicles easier to buy and use.

### Why Is This Project Important?

By ranking people according to how likely they are to buy an electric vehicle, this project can help businesses plan their sales activities, improve marketing, and make better decisions about where to invest. It can also help governments identify ways to encourage more people to switch to cleaner transportation.

**In simple terms:** This project uses machine learning to help answer the question, *"Who is most likely to buy an electric vehicle, and how can we use this information to make better decisions?"*

Importantly, this is a **Kaggle-style competition dataset**, not verified
real-world purchase records. The findings below describe patterns *in this
dataset* — they are a demonstration of a modelling workflow, not a
validated claim about real consumer behavior. Any production use would
need revalidation on real, representative data.

## Dataset

- `train.csv`: 668,665 rows, 15 columns (14 features + target)
- `test.csv`: 286,571 rows, 14 columns (no target)
- No missing values, no duplicate rows or ids in either file
- Feature set: `Age`, `Annual_Income_USD`, `Daily_Commute_km`,
  `Number_of_Cars_Owned`, `Charging_Stations_Near_Home`,
  `Charging_Stations_Near_Work`, `Environmental_Concern_Level`, `Gender`,
  `City_Type`, `Current_Car_Type`, `Home_Charging_Possible`,
  `Subsidy_Available`, `Range_Anxiety_Level`

## Target Variable

**`Will_Buy_EV`** is the target variable. It tells us whether a person is willing to buy an electric vehicle (EV).

- **`Yes` = 1:** The person is willing to buy an electric vehicle.
- **`No` = 0:** The person is not willing to buy an electric vehicle.

### Class Imbalance

The dataset is **imbalanced** because the two groups do not have an equal number of people:

- **17.5% Yes:** People willing to buy an EV.
- **82.5% No:** People not willing to buy an EV.

This means that for every person who is willing to buy an EV, there are about 4.7 people who are not. This difference can make it harder for a model to identify people who are likely to buy an EV.

### Why Did We Not Use Class Weighting?

Class weighting is a technique that gives more importance to the smaller group during model training. We tested it to see whether it would improve our model's performance.

However, the cross-validation results showed almost no improvement in ROC-AUC:

- **Without class weighting:** 0.9418 ROC-AUC
- **With class weighting:** 0.9417 ROC-AUC

Since the model performed slightly better without class weighting, we decided to use the original class distribution for our final models.

**In simple terms:** Our goal is to rank people correctly according to how likely they are to buy an EV. Since class weighting did not improve the model's ability to rank people, we kept the simpler approach without class weighting.

We used **ROC-AUC** to compare the models because it measures how well a model separates people who are likely to buy an EV from those who are not, across different decision thresholds.

The competition scores **predicted probabilities** with ROC-AUC, so the
submission is not thresholded at 0.5 — raw probabilities are preserved.

## Exploratory Data Analysis

![Target distribution](outputs/figures/target_distribution.png)
![Numeric distributions](outputs/figures/numeric_distributions.png)
![Categorical distributions](outputs/figures/categorical_distributions.png)

Three features show a strong, near-monotonic relationship with purchase
likelihood:

| Driver | Observed Will_Buy_EV rate |
|---|---|
| `Subsidy_Available` = Yes | 27.5% |
| `Subsidy_Available` = No | 0.6% |
| `Environmental_Concern_Level` = 5 (highest) | 51.8% |
| `Environmental_Concern_Level` = 1 (lowest) | 0.6% |
| `Range_Anxiety_Level` = Low | 18.9% |
| `Range_Anxiety_Level` = High | 0.1% |

![Target rate by driver](outputs/figures/target_rate_by_driver.png)
![Correlation heatmap](outputs/figures/correlation_heatmap.png)

These are strong *associations*, not causal claims — the data does not
support concluding, for example, that a subsidy alone would cause a
customer to buy an EV.

## Feature Engineering

New features were built only from information already present in a row
(no cross-row aggregation, no target usage), so they carry no leakage risk:

- `Age_Group`, `Income_Band` — bucketed versions of continuous fields, useful for linear models and readability
- `Total_Charging_Access` — sum of home + work charging station counts
- `Commute_Intensity` — bucketed daily commute distance
- `Subsidy_x_MidLowIncome` — interaction between subsidy availability and below-median income, testing whether subsidies matter more for less-affluent buyers
- `Multi_Car_Household` — flag for owning 2+ cars

## Validation Strategy

**Stratified K-Fold Cross-Validation** was used throughout, preserving the
17.5%/82.5% class balance in every fold — appropriate since this is a
single-snapshot binary classification problem with no evidence of repeated
customers, time ordering, or grouped structure (checked: `id` is a simple
unique row index in both files, and there are no timestamp or grouping
columns).

## Computing Resources

In simple terms: Because computing resources were limited, we used 3-fold cross-validation to save time. We then checked the model using both cross-validation and a separate test set to get a more reliable picture of its performance.

## Models Tested

| Model | CV ROC-AUC Mean | CV ROC-AUC Std |
|---|---:|---:|
| XGBoost (default params) | 0.9415 | 0.0006 |
| LightGBM (default params) | 0.9414 | 0.0006 |
| CatBoost (default params, reduced iterations for compute) | 0.9404 | 0.0007 |
| Logistic Regression (baseline) | 0.9383 | 0.0007 |
| Random Forest | 0.9349 | 0.0005 |

![Model comparison](outputs/figures/model_comparison.png)

## Hyperparameter Optimization

Optuna (Bayesian/TPE search) was used to tune LightGBM and XGBoost,
optimizing directly for **ROC-AUC** on a held-out validation split (not
accuracy). After 18–20 trials each:

- **Tuned LightGBM**: AUC = 0.9417 (`n_estimators=502, learning_rate=0.073, num_leaves=17, max_depth=5, ...`)
- **Tuned XGBoost**: AUC = 0.9417 (`n_estimators=515, learning_rate=0.095, max_depth=4, min_child_weight=11, ...`)

## Model Ensembling

I combined predictions from LightGBM and XGBoost to improve the model's ability to predict which customers are likely to buy an electric vehicle (EV). I tested different combinations of the models to find the best balance, rather than choosing the weights randomly.

The best-performing combination used 48% of LightGBM’s predictions and 52% of XGBoost’s predictions, with these weights selected through testing to achieve the best overall prediction performance.

The combined model achieved an **OOF ROC-AUC of 0.9419**, a small improvement over either model used alone (0.9418).

## ROC-AUC

ROC-AUC measures how well a model distinguishes between customers who are likely to buy an EV and those who are not.

A score of **0.9419** means the model has about a **94% chance of ranking a randomly selected EV buyer as more likely to purchase than a randomly selected non-buyer**.

From a business perspective, this can help EV companies identify potential customers more effectively, target their marketing campaigns, and allocate advertising budgets more efficiently.

![ROC curve](outputs/figures/roc_curve.png)

## Model Calibration

Predicted probabilities from the tuned LightGBM were checked against
observed frequencies in 10 quantile bins on a hold-out set. Calibration is
already close to the ideal diagonal (e.g. predicted ≈0.545 vs. observed
=0.535 in the largest-gap bin), so no separate isotonic/Platt calibration
step was applied — it would not meaningfully improve this model.

![Calibration curve](outputs/figures/calibration_curve.png)

## Explainability

![Feature importance](outputs/figures/feature_importance.png)
![SHAP summary](outputs/figures/shap_summary.png)

Both native gain importance and SHAP agree on the top drivers:
`Subsidy_Available`/`Subsidy_Yes`, `Environmental_Concern_Level`,
`Annual_Income_USD`, and `Range_Anxiety_Level`. The model relies heavily
on these four signals — this means the model's predictions are, in this
dataset, largely a story about subsidy access, environmental attitude, and
income, rather than about vehicle type, charging-station counts, or
demographics like gender.

## Error Analysis

The model was evaluated on a separate hold-out dataset to assess how well it predicts customers' likelihood of buying an electric vehicle (EV).

- **74.8% high-confidence and correct:** The model made confident predictions that matched the actual customer outcomes.
- **2.5% high-confidence but incorrect:** The model was confident in its predictions but got the actual outcomes wrong.
- **6.1% uncertain predictions:** The model assigned probabilities between 40% and 60%, indicating uncertainty about whether these customers would buy an EV.

### Performance by City Type

The model achieved a ROC-AUC of **0.945 in Urban areas** and **0.937 in Rural areas**. This indicates that the model was slightly better at distinguishing potential buyers from non-buyers in urban areas.

One possible explanation is that differences in charging infrastructure and access to charging stations may influence EV purchasing decisions in rural areas. However, further analysis would be needed to confirm this.

Overall, the results show that the model performs well, but some predictions remain uncertain or incorrect and may require further investigation.

## Responsible AI and Fairness

The model's performance was almost the same for female and male customers. The AUC score was 0.9418 for females and 0.9419 for males. For customers who selected "Other" as their gender, the score was 0.9294. This group had a much smaller sample size (about 1,034 customers), so more data is needed before drawing conclusions about any performance difference.

`Subsidy_Available` and `Environmental_Concern_Level` are not demographic characteristics. However, they may be related to protected groups, so this should be checked before using the model for important decisions, such as loan eligibility.

The `Gender` and `City_Type` columns were included in the analysis to help identify possible differences in model performance across groups.

## Business Impact

The model could help businesses in several ways:

- **Marketing:** Identify customers who are more likely to buy an electric vehicle (EV).
- **Sales:** Rank potential customers based on their predicted interest.
- **Financing:** Identify potential customers who may be interested in EV loan products.
- **Dealership promotions:** Help businesses target customers with suitable offers.

However, the model should not be used alone to decide where to build charging stations because the dataset does not contain enough geographical and market information.

The model also cannot prove that a particular action, such as offering a subsidy, will cause someone to buy an EV. It only identifies patterns in the available data.

## Customer Segmentation

The model predicts the probability that a customer will buy an electric vehicle. Based on these predictions, customers can be grouped into three categories:

- **High interest:** Customers with a high predicted probability of buying an EV.
- **Medium interest:** Customers with a moderate predicted probability.
- **Low interest:** Customers with a low predicted probability.

Businesses can use these groups to plan different marketing strategies. However, these categories are based on the model's predictions and do not guarantee what an individual customer will actually do.

## Limitations

- **Limited data over time:** The dataset represents a single snapshot and does not track customers over time or confirm their actual purchases.
- **CatBoost model:** CatBoost was tested with fewer training iterations than LightGBM and XGBoost because of limited computing resources. It was also not extensively tuned to find its best settings.
- **Cross-validation:** Three folds were used instead of five to reduce computing time during model comparison.
- **Small gender group:** The "Other" gender group had approximately 1,000 customers in the hold-out dataset. Therefore, the fairness results for this group are less certain and should be checked with more data.

## Deployment

A Streamlit dashboard (`app/app.py`) allows users to enter example customer details and view the model's predicted probability of buying an electric vehicle.

The dashboard also displays model performance results and explainability information to help users understand how the model makes its predictions.

## Project Structure

```
EV-Purchase-Prediction/
│
├── data/                     # train.csv, test.csv, sample_submission.csv
├── notebooks/
│   └── EV_Purchase_Prediction.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py               # CV model comparison
│   ├── tune.py                 # Optuna hyperparameter search
│   ├── generate_tuned_oof.py   # OOF predictions for blending
│   ├── imbalance_check.py
│   ├── evaluate.py             # explainability, calibration, error analysis
│   ├── make_plots.py / make_model_plots.py
│   └── predict.py              # trains final blend, writes submission.csv
├── models/                    # saved model artifacts (.pkl)
├── outputs/
│   ├── submission.csv
│   ├── feature_importance.csv
│   ├── model_comparison.csv
│   └── figures/
├── app/
│   └── app.py                  # Streamlit dashboard
├── requirements.txt
├── README.md
└── .gitignore
```

## Technologies Used

Python | Pandas | NumPy | Scikit-learn | XGBoost | LightGBM | CatBoost |
SHAP | Optuna | Feature Engineering | Stratified Cross-Validation |
ROC-AUC | Model Explainability | Streamlit | Matplotlib / Seaborn

## How to Run

```bash
pip install -r requirements.txt

# 1. Compare baseline models (run once per model)
python src/train.py --model LightGBM

# 2. Tune hyperparameters
python src/tune.py --model lightgbm --n_trials 20

# 3. Generate OOF predictions for blending
python src/generate_tuned_oof.py --model lightgbm_tuned

# 4. Explainability, calibration, error analysis
python src/evaluate.py

# 5. Train final blended model and write submission.csv
python src/predict.py

# 6. Launch the dashboard
streamlit run app/app.py
```

## Future Improvements

- **Improve model evaluation:** Use 5-fold cross-validation and fine-tune CatBoost on a computer with more processing power to get more reliable results.
- **Improve categorical data handling:** Use better encoding methods if more categories or new categorical features are added to the dataset.
- **Add more external data:** Include dealership locations and local charging station information to improve insights for EV market and infrastructure planning.
- **Test with real-world data:** Compare the model's predictions with actual EV purchases to check how well it performs before using it in real business decisions.