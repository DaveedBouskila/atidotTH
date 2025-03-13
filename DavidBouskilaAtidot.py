# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tACDrvZC_GGgZE4jyD8Ir7BPAhPP_m6n
"""

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('churn_data (4).csv')

df

df.groupby('plan_type').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

df.groupby('churn').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

# Check for missing values
print(df.isnull().sum())

#we see a pretty welll distributed plan type so imputing missing row with mode
df['plan_type'].fillna('Basic', inplace=True)
# One-hot encoding for plan_type
df = pd.get_dummies(df, columns=['plan_type'], drop_first=True)

# Convert date columns to datetime format
df['date'] = pd.to_datetime(df['date'])
df['issuing_date'] = pd.to_datetime(df['issuing_date'])

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Replace this with your actual transaction amount data
transaction_amounts = df['transaction_amount'].dropna()  # Drop NaNs for plotting

# Compute mean and median
mean_value = transaction_amounts.mean()
median_value = transaction_amounts.median()

# Plot distribution
plt.figure(figsize=(10, 5))
sns.histplot(transaction_amounts, bins=50, kde=True, color="lightgray")

# Add mean and median lines
plt.axvline(mean_value, color='blue', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')

plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.title("Transaction Amount Distribution")
plt.legend()
plt.show()

# we see mean and median being almost eual so using mean to impute missing rows
df['transaction_amount'].fillna(df['transaction_amount'].mean(), inplace=True)

# Recency feature
df['days_since_last_transaction'] = df.groupby('customer_id')['date'].diff().dt.days

# Rolling transaction amount features
df['rolling_mean_3'] = df.groupby('customer_id')['transaction_amount'].transform(lambda x: x.rolling(3, min_periods=1).mean())
df['rolling_std_3'] = df.groupby('customer_id')['transaction_amount'].transform(lambda x: x.rolling(3, min_periods=1).std())

# Customer tenure
df['customer_tenure'] = (df['date'] - df['issuing_date']).dt.days

# Check for missing values
print(df.isnull().sum())

df['days_since_last_transaction'].fillna(df['days_since_last_transaction'].mean(), inplace=True)
df['rolling_std_3'].fillna(df['rolling_std_3'].mean(), inplace=True)

# Define train-test split
train_df = df[df['date'] < '2023-11-01']
test_df = df[(df['date'] >= '2023-11-01') & (df['date'] <= '2024-01-01')]

# Drop unnecessary columns
features = ['transaction_amount', 'days_since_last_transaction', 'rolling_mean_3', 'rolling_std_3', 'customer_tenure'] + [col for col in df.columns if 'plan_type' in col]
X_train, y_train = train_df[features], train_df['churn']
X_test, y_test = test_df[features], test_df['churn']

print(X_train.shape)
print(X_test.shape)

#with this highly engineered data we are choosing xgboost for efficiency and strength in imbalanced classification
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# Train XGBoost model
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    scale_pos_weight=6,  # Increase weight for churners
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print(max(y_pred))

metrics = {
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred)
}

# Save metrics to JSON
import json
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print(metrics)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Get probability scores
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Compute precision-recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)

# Plot curve
plt.figure(figsize=(8,5))
plt.plot(thresholds, precisions[:-1], label="Precision")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.title("Precision-Recall Tradeoff")
plt.show()

from imblearn.over_sampling import SMOTE

# Apply SMOTE to balance classes (increase churn samples)
smote = SMOTE(sampling_strategy=0.8, random_state=42)  # Adjust ratio as needed
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Retrain model
model.fit(X_train_balanced, y_train_balanced)

# Predict again
y_pred = model.predict(X_test)
print("Max prediction after SMOTE:", max(y_pred))
metrics = {
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred)
}

# Save metrics to JSON
import json
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print(metrics)

print(X_train.dtypes)

# Convert all boolean columns to integers (SHAP might not support bool directly)
X_train = X_train.astype(float)
X_test = X_test.astype(float)

import shap

# Initialize SHAP explainer
explainer = shap.Explainer(model, X_train)

# Compute SHAP values
shap_values = explainer(X_test)

# Plot feature importance
shap.summary_plot(shap_values, X_test)

# we will add some new features now that shap has taight us about our feature strength

# Normalize by customer's tenure to capture inactivity risk
df["normalized_inactivity"] = df["days_since_last_transaction"] / (df["customer_tenure"] + 1)  # Avoid div by 0
df["normalized_inactivity"].fillna(1, inplace=True)  # If NaN (new customer), assume max risk

# Compute spending change percentage over last 3 months
df["spending_change"] = df.groupby("customer_id")["rolling_mean_3"].pct_change()
df["spending_change"].fillna(0, inplace=True)  # If NaN, assume no change

df["standard_plan_risk"] = df["plan_type_Standard"] * df["normalized_inactivity"]

# Define train-test split
train_df = df[df['date'] < '2023-11-01']
test_df = df[(df['date'] >= '2023-11-01') & (df['date'] <= '2024-01-01')]

# Drop unnecessary columns
features = ['transaction_amount', 'days_since_last_transaction', 'rolling_mean_3', 'rolling_std_3', 'customer_tenure'] + [col for col in df.columns if 'plan_type' in col]
X_train, y_train = train_df[features], train_df['churn']
X_test, y_test = test_df[features], test_df['churn']

# Updated Feature Set
features = [
    "transaction_amount", "days_since_last_transaction",
    "rolling_mean_3", "rolling_std_3", "customer_tenure",
    "plan_type_Premium", "plan_type_Standard",
    "normalized_inactivity", "spending_change", "standard_plan_risk"
]

X_train, y_train = train_df[features], train_df["churn"]
X_test, y_test = test_df[features], test_df["churn"]

# Retrain Model
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
metrics = {
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred)
}
print(metrics)

from imblearn.over_sampling import SMOTE

# Apply SMOTE to balance classes (increase churn samples)
smote = SMOTE(sampling_strategy=0.8, random_state=42)  # Adjust ratio as needed
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Retrain model
model.fit(X_train_balanced, y_train_balanced)

# Predict again
y_pred = model.predict(X_test)
print("Max prediction after SMOTE:", max(y_pred))
metrics = {
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred)
}

# Save metrics to JSON
import json
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print(metrics)

predicted = model.predict(train_df[features])

import joblib

# Save model
joblib.dump(model, "churn_model.pkl")

# Save predictions
test_df["churn_prediction"] = y_pred
test_df.to_csv("churn_data_with_predictions.csv", index=False)

