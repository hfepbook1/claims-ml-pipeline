#!/usr/bin/env python
"""
train_models.py

This script integrates EDA, preprocessing, model pipeline building,
training, evaluation, and saving for an end-to-end ML project on a synthetic
healthcare claims dataset.
"""

import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

# --------------------------------------------------
# 1. EDA and Preprocessing
# --------------------------------------------------

print("Loading synthetic data from 'data/synthetic_claims.csv'...")
df = pd.read_csv('data/synthetic_claims.csv')

# Split features and targets
X = df.drop(columns=['claim_cost', 'is_fraud', 'readmit_30d'])
y_cost = df['claim_cost']
y_fraud = df['is_fraud']
y_readmit = df['readmit_30d']

# Train-test split (stratify on fraud to maintain class ratio)
print("Splitting data into train and test sets...")
X_train, X_test, y_cost_train, y_cost_test, y_fraud_train, y_fraud_test, y_readmit_train, y_readmit_test = \
    train_test_split(X, y_cost, y_fraud, y_readmit, test_size=0.2, random_state=42,
                     stratify=y_fraud)

# Define numeric and categorical columns
numeric_cols = ['age', 'chronic_condition_count', 'num_visits', 'num_er_visits', 'num_inpatient_stays']
categorical_cols = ['gender', 'region', 'provider_type', 'primary_diagnosis']

# Impute missing values
print("Imputing missing values...")
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

# Build a ColumnTransformer for encoding and scaling
print("Fitting the preprocessor...")
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = StandardScaler()

preprocessor = ColumnTransformer(transformers=[
    ('num', scaler, numeric_cols),
    ('cat', encoder, categorical_cols)
])

# Fit the preprocessor on training data and transform both train and test sets
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Save the fitted preprocessor to disk
joblib.dump(preprocessor, 'models/preprocessor.pkl')
print("Preprocessor saved to 'models/preprocessor.pkl'.")

# Handle class imbalance for classification targets using SMOTE (optional for training)
# Note: Here we prepare balanced datasets for potential use if needed.
smote = SMOTE(random_state=42)
X_train_fraud, y_fraud_train_bal = smote.fit_resample(X_train_proc, y_fraud_train)
X_train_readmit, y_readmit_train_bal = smote.fit_resample(X_train_proc, y_readmit_train)

# Save preprocessed data for reuse or inspection (optional)
print("Saving preprocessed train and test data...")
train_data = pd.DataFrame(X_train_proc)
train_data['claim_cost'] = y_cost_train.values
train_data['is_fraud'] = y_fraud_train.values
train_data['readmit_30d'] = y_readmit_train.values

test_data = pd.DataFrame(X_test_proc)
test_data['claim_cost'] = y_cost_test.values
test_data['is_fraud'] = y_fraud_test.values
test_data['readmit_30d'] = y_readmit_test.values

train_data.to_csv('data/train_preprocessed.csv', index=False)
test_data.to_csv('data/test_preprocessed.csv', index=False)
print("Preprocessed data saved to 'data/train_preprocessed.csv' and 'data/test_preprocessed.csv'.")

# --------------------------------------------------
# 2. Build Model Pipelines, Train, Evaluate, and Save Models
# --------------------------------------------------

# Load the preprocessor from disk (simulate how your API will load it)
print("Loading preprocessor from disk...")
preprocessor_loaded = joblib.load('models/preprocessor.pkl')

# Define models
regressor = XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
classifier = XGBClassifier(n_estimators=100, max_depth=5,
                           use_label_encoder=False, eval_metric='logloss', random_state=42)

# Build pipelines (each pipeline takes raw input, preprocesses it, then applies the model)
cost_pipeline = Pipeline([
    ('preprocessor', preprocessor_loaded),
    ('xgb_reg', regressor)
])
fraud_pipeline = Pipeline([
    ('preprocessor', preprocessor_loaded),
    ('xgb_clf', classifier)
])
readmit_pipeline = Pipeline([
    ('preprocessor', preprocessor_loaded),
    ('xgb_clf', XGBClassifier(n_estimators=100, max_depth=5,
                              use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# Train the pipelines using the original (unbalanced) training data
print("Training claim cost regression model...")
cost_pipeline.fit(X_train, y_cost_train)
print("Training fraud detection model...")
fraud_pipeline.fit(X_train, y_fraud_train)
print("Training readmission prediction model...")
readmit_pipeline.fit(X_train, y_readmit_train)

# Evaluate on test data
print("\nEvaluating models on test data...")
y_cost_pred = cost_pipeline.predict(X_test)
y_fraud_pred = fraud_pipeline.predict(X_test)
y_readmit_pred = readmit_pipeline.predict(X_test)

rmse_cost = mean_squared_error(y_cost_test, y_cost_pred, squared=False)
fraud_acc = accuracy_score(y_fraud_test, y_fraud_pred)
readmit_acc = accuracy_score(y_readmit_test, y_readmit_pred)

print("Test RMSE (Claim Cost):", rmse_cost)
print("Test Fraud Accuracy:", fraud_acc)
print("Test Readmission Accuracy:", readmit_acc)
print("Fraud Confusion Matrix:\n", confusion_matrix(y_fraud_test, y_fraud_pred))
print("Readmission Confusion Matrix:\n", confusion_matrix(y_readmit_test, y_readmit_pred))

# Save the trained model pipelines to disk for later use (e.g., in your FastAPI backend)
joblib.dump(cost_pipeline, 'models/model_cost.pkl')
joblib.dump(fraud_pipeline, 'models/model_fraud.pkl')
joblib.dump(readmit_pipeline, 'models/model_readmit.pkl')
print("Trained model pipelines saved to the 'models/' directory.")
