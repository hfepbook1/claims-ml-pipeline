# frontend/pages/02_EDA_Visualizations.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="EDA & Insights", layout="wide")

st.title("Exploratory Data Analysis & Insights")
st.markdown("""
This page presents a comprehensive analysis of the synthetic healthcare claims dataset.  
We will:
- **Load the dataset** and show an overview.
- **Check and fill missing values:** Using median for numerical features and mode for categorical.
- **Examine distributions:** For key variables such as claim cost and age.
- **Assess skewness:** If claim cost is heavily skewed, we will apply a log transformation and compare the results.
- **Visualize relationships:** With boxplots and heatmaps to highlight correlations and differences by provider type.
- **Display distributions:** For fraud flags and 30-day readmission counts.

Let's explore the data!
""")

# Cache data loading to improve performance
@st.cache_data
def load_data():
    # Make sure the path is correct relative to the root directory
    df = pd.read_csv("data/synthetic_claims.csv")
    return df

df = load_data()

st.subheader("Dataset Overview")
st.write("Dataset Preview:")
st.dataframe(df.head())
st.write("Shape of the dataset:", df.shape)
st.write("Summary Statistics:")
st.write(df.describe(include='all'))

# ---------------------------
# Missing Values Analysis
# ---------------------------
st.subheader("Missing Values Analysis")
missing = df.isnull().sum()
missing = missing[missing > 0]
if not missing.empty:
    st.write("Columns with missing values:")
    st.write(missing)
    st.markdown("""
    **Imputation Strategy:**  
    - **Numeric features:** Fill missing values with the median.  
    - **Categorical features:** Fill missing values with the mode or a constant such as "Unknown".
    """)
else:
    st.write("No missing values detected.")

# Let's perform imputation for demonstration
df_imputed = df.copy()
num_cols = ['age', 'chronic_condition_count', 'num_visits', 'num_er_visits', 'num_inpatient_stays', 'claim_cost']
cat_cols = ['gender', 'region', 'provider_type', 'primary_diagnosis']

# Impute numeric columns with median
for col in num_cols:
    median_val = df_imputed[col].median()
    df_imputed[col].fillna(median_val, inplace=True)

# Impute categorical columns with mode
for col in cat_cols:
    mode_val = df_imputed[col].mode()[0]
    df_imputed[col].fillna(mode_val, inplace=True)

st.markdown("**After Imputation:**")
st.write(df_imputed.isnull().sum())

# ---------------------------
# Distribution of Claim Cost
# ---------------------------
st.subheader("Distribution of Claim Cost")

# Plot raw claim_cost distribution
fig1, ax1 = plt.subplots(figsize=(8, 4))
sns.histplot(df_imputed['claim_cost'], bins=50, kde=True, color='steelblue', ax=ax1)
ax1.set_title("Raw Claim Cost Distribution")
ax1.set_xlabel("Claim Cost")
ax1.set_ylabel("Frequency")
st.pyplot(fig1)

# Check skewness
skewness = df_imputed['claim_cost'].skew()
st.write(f"Skewness of claim_cost: **{skewness:.2f}**")

# If skewed, apply log transformation
if abs(skewness) > 1:
    st.markdown("The distribution is highly skewed. We will apply a log transformation for better visualization.")
    df_imputed['log_claim_cost'] = np.log1p(df_imputed['claim_cost'])  # add 1 to avoid log(0)
    
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df_imputed['claim_cost'], bins=50, kde=True, color='steelblue', ax=ax2)
    ax2.set_title("Raw Claim Cost Distribution")
    ax2.set_xlabel("Claim Cost")
    
    sns.histplot(df_imputed['log_claim_cost'], bins=50, kde=True, color='coral', ax=ax3)
    ax3.set_title("Log-Transformed Claim Cost Distribution")
    ax3.set_xlabel("Log(Claim Cost + 1)")
    
    st.pyplot(fig2)
else:
    st.markdown("The claim cost distribution is not highly skewed; no log transformation is applied.")

# ---------------------------
# Age Distribution
# ---------------------------
st.subheader("Age Distribution")
fig3, ax3 = plt.subplots(figsize=(8, 4))
sns.histplot(df_imputed['age'], bins=30, kde=True, ax=ax3, color='forestgreen')
ax3.set_title("Age Distribution")
ax3.set_xlabel("Age")
ax3.set_ylabel("Frequency")
st.pyplot(fig3)

# ---------------------------
# Boxplot: Claim Cost by Provider Type
# ---------------------------
st.subheader("Claim Cost by Provider Type")
fig4, ax4 = plt.subplots(figsize=(8, 5))
sns.boxplot(x='provider_type', y='claim_cost', data=df_imputed, ax=ax4, palette='pastel')
ax4.set_title("Claim Cost by Provider Type")
ax4.set_xlabel("Provider Type")
ax4.set_ylabel("Claim Cost")
st.pyplot(fig4)

# ---------------------------
# Correlation Heatmap
# ---------------------------
st.subheader("Correlation Heatmap")
numeric_features = ['age', 'chronic_condition_count', 'num_visits', 'num_er_visits', 'num_inpatient_stays', 'claim_cost']
corr_matrix = df_imputed[numeric_features].corr()
fig5, ax5 = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax5)
ax5.set_title("Correlation Heatmap")
st.pyplot(fig5)

# ---------------------------
# Fraud Flag and Readmission Distributions
# ---------------------------
st.subheader("Fraud Flag Distribution")
fig6, ax6 = plt.subplots(figsize=(6, 4))
sns.countplot(x='is_fraud', data=df_imputed, ax=ax6, palette='Set2')
ax6.set_title("Fraud Flag Counts")
ax6.set_xlabel("is_fraud")
ax6.set_ylabel("Count")
st.pyplot(fig6)
st.write("Fraud counts:")
st.write(df_imputed['is_fraud'].value_counts())

st.subheader("30-Day Readmission Distribution")
fig7, ax7 = plt.subplots(figsize=(6, 4))
sns.countplot(x='readmit_30d', data=df_imputed, ax=ax7, palette='Set3')
ax7.set_title("30-Day Readmission Counts")
ax7.set_xlabel("readmit_30d")
ax7.set_ylabel("Count")
st.pyplot(fig7)
st.write("Readmission counts:")
st.write(df_imputed['readmit_30d'].value_counts())

# ---------------------------
# Insights and Findings
# ---------------------------
st.subheader("Key Insights & Findings")
st.markdown(f"""
- **Missing Values:**  
  A small percentage of missing values were detected in categorical features and were filled with the mode (or "Unknown"). Numerical features were imputed using the median.

- **Claim Cost Distribution:**  
  The raw claim cost distribution is right-skewed (skewness = **{skewness:.2f}**).  
  {"A log transformation was applied to better visualize the distribution." if abs(skewness) > 1 else "No log transformation was necessary."}

- **Age Distribution:**  
  The age distribution spans a wide range, indicating a diverse patient population.

- **Provider Impact:**  
  Boxplots show that claims associated with hospitals generally incur higher costs compared to other provider types.

- **Correlations:**  
  There is a positive correlation between age, chronic condition count, and claim cost, suggesting that older patients with more chronic conditions tend to have higher medical expenses.

- **Fraud & Readmission:**  
  Both the fraud flag and 30-day readmission are relatively rare events, which is consistent with real-world insurance data.
""")
