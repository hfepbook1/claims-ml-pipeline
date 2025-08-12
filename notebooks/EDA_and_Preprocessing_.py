# Databricks notebook source
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

# Load synthetic data
df = pd.read_csv('data/synthetic_claims.csv')

# Split features and targets
X = df.drop(columns=['claim_cost', 'is_fraud', 'readmit_30d'])
y_cost = df['claim_cost']
y_fraud = df['is_fraud']
y_readmit = df['readmit_30d']

# Train-test split (stratify on binary targets to maintain class ratio in splits)
X_train, X_test, y_cost_train, y_cost_test, y_fraud_train, y_fraud_test, y_readmit_train, y_readmit_test = \
    train_test_split(X, y_cost, y_fraud, y_readmit, test_size=0.2, random_state=42,
                     stratify=y_fraud)  # stratify on fraud for balanced split

# Define numeric and categorical columns
numeric_cols = ['age', 'chronic_condition_count', 'num_visits', 'num_er_visits', 'num_inpatient_stays']
categorical_cols = ['gender', 'region', 'provider_type', 'primary_diagnosis']

# Impute missing values
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

# Encoding and scaling: fit on train, apply to train and test
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = StandardScaler()

# We use ColumnTransformer to apply transformations
preprocessor = ColumnTransformer(transformers=[
    ('num', scaler, numeric_cols),
    ('cat', encoder, categorical_cols)
])

# Fit the preprocessor on the training data and transform both sets
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# At this point, X_train_proc and X_test_proc are numpy arrays ready for modeling.
# Handle class imbalance for classification targets using SMOTE
smote = SMOTE(random_state=42)
X_train_fraud, y_fraud_train_bal = smote.fit_resample(X_train_proc, y_fraud_train)
X_train_readmit, y_readmit_train_bal = smote.fit_resample(X_train_proc, y_readmit_train)

# Save preprocessed data for reuse (we can convert back to DataFrame if needed for inspection)
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


# COMMAND ----------

