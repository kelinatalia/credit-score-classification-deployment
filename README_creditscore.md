# Credit Score Classification with Model Deployment

**Individual Project**

## Overview
Built an end-to-end machine learning pipeline to classify a customer's credit score into three categories (Poor, Standard, Good) based on financial and behavioral data. The project covers the full lifecycle, from data cleaning and feature engineering to model comparison, tuning, and deployment (local pipeline, AWS cloud pipeline, and a Streamlit web app).

## What I Did
- Cleaned messy real-world data, including fixing datatype mismatches, handling invalid values (negative ages, unrealistic entries), and imputing missing values.
- Analyzed feature distributions and correlations to understand what drives credit score outcomes, such as outstanding debt, interest rate, and payment delays.
- Handled class imbalance in the target variable using class weighting.
- Compared multiple models (Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM), then tuned the top two performers with RandomizedSearchCV.
- Selected Random Forest as the final model based on macro F1-score and recall on the minority "Poor" class, which matters most for risk-sensitive predictions.
- Exported the full preprocessing and model pipeline, then deployed it through a local pipeline, an AWS cloud pipeline, and an interactive Streamlit app.
- Tracked experiments using MLflow.

## Results
| Model | Accuracy | Macro F1 | Poor Class Recall |
|---|---|---|---|
| Random Forest (tuned) | 0.7290 | 0.7114 | 0.7414 |
| XGBoost (tuned) | 0.7286 | 0.7058 | 0.7185 |

Random Forest was selected for deployment due to its stronger macro F1-score and better recall on the "Poor" credit class, which is the most critical to catch correctly in a real-world lending context.

## Tech Stack
Python, scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn, SHAP, MLflow, Streamlit, AWS

## What I Learned
Gained hands-on experience in handling real-world messy data, comparing and tuning multiple model architectures, choosing the right evaluation metric for an imbalanced, risk-sensitive classification problem, and deploying a full machine learning pipeline across local and cloud environments.
