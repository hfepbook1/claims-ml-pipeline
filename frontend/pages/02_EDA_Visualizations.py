import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page config
st.set_page_config(page_title="Healthcare Claims EDA", layout="wide")

st.title("Healthcare Claims Exploratory Data Analysis Dashboard")

# Load dataset
@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/synthetic_claims.csv")
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'data/synthetic_claims.csv' exists.")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

# Identify expected columns in dataframe (case-insensitive)
gender_col = None
region_col = None
provider_col = None
diag_col = None
cost_col = None
fraud_col = None
readmit_col = None

for col in df.columns:
    lc = col.lower()
    if lc in ["gender", "sex"]:
        gender_col = col
    if lc in ["region", "state", "location"]:
        region_col = col
    if "provider" in lc:
        provider_col = col
    if "diagnosis" in lc:
        diag_col = col
    if lc in ["cost", "claim_cost", "amount", "claimamount"]:
        cost_col = col
    if "fraud" in lc:
        fraud_col = col
    if "readmit" in lc or "readmitted" in lc:
        readmit_col = col

# Ensure required columns exist
if None in [gender_col, region_col, provider_col, diag_col, cost_col]:
    st.error("Required columns not found. Ensure dataset has Gender, Region, Provider Type, Primary Diagnosis, Claim Cost.")
    st.stop()

# Sidebar filter widgets
selected_genders = st.sidebar.multiselect(
    "Gender",
    options=df[gender_col].dropna().unique().tolist(),
    default=df[gender_col].dropna().unique().tolist()
)
selected_regions = st.sidebar.multiselect(
    "Region",
    options=df[region_col].dropna().unique().tolist(),
    default=df[region_col].dropna().unique().tolist()
)
selected_providers = st.sidebar.multiselect(
    "Provider Type",
    options=df[provider_col].dropna().unique().tolist(),
    default=df[provider_col].dropna().unique().tolist()
)
selected_diagnoses = st.sidebar.multiselect(
    "Primary Diagnosis",
    options=df[diag_col].dropna().unique().tolist(),
    default=df[diag_col].dropna().unique().tolist()
)

# Filter dataframe based on selections
df_filtered = df[
    (df[gender_col].isin(selected_genders)) &
    (df[region_col].isin(selected_regions)) &
    (df[provider_col].isin(selected_providers)) &
    (df[diag_col].isin(selected_diagnoses))
]

# Toggle for log transformation on claim cost
log_option = st.sidebar.radio("Claim Cost Scale", ('Raw', 'Log'))
if log_option == 'Log':
    df_filtered['cost_to_plot'] = np.log1p(df_filtered[cost_col].clip(lower=0))
else:
    df_filtered['cost_to_plot'] = df_filtered[cost_col]

# ===== UPDATED Overview Tab =====
tabs = st.tabs(["Overview", "Visualizations", "Missing Data"])
with tabs[0]:
    st.header("Key Metrics & ROI Analytics")

    # Base metrics
    total_cost = df[cost_col].sum()
    fraud_mask = df[fraud_col].astype(bool)
    fraud_cost = df.loc[fraud_mask, cost_col].sum()
    fraud_ratio = fraud_cost / total_cost if total_cost>0 else 0
    avg_cost_by_fraud = df.groupby(fraud_mask)[cost_col].mean()
    avg_fraud = avg_cost_by_fraud.get(True, np.nan)
    avg_notfraud = avg_cost_by_fraud.get(False, np.nan)
    readmit_mask = df[readmit_col].astype(bool)
    avg_readmit_cost = df.loc[readmit_mask, cost_col].mean()

    # Denial reduction & recovered dollars
    denial_reduction_pct = 0.15  # 15%
    dollars_recovered = denial_reduction_pct * total_cost

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Fraud Cost Ratio", f"{fraud_ratio:.1%}")
    col2.metric("Avg Cost (Fraud)", f"${avg_fraud:,.0f}")
    col3.metric("Avg Cost (No Fraud)", f"${avg_notfraud:,.0f}")
    col4.metric("Avg Cost (Readmit)", f"${avg_readmit_cost:,.0f}")
    col5.metric("Denial ↓ & Recovery", f"{denial_reduction_pct:.1%} → ${dollars_recovered/1e6:.2f}M")

    st.markdown("---")

    st.subheader("ROI Curve: Incremental Revenue vs. Model Coverage")
    # synthetic coverage vs recovered dollars
    coverage = np.linspace(0.2, 1.0, 5)
    recovered = coverage * dollars_recovered
    roi_df = pd.DataFrame({"Model Coverage": coverage, "Recovered ($M)": recovered/1e6})
    fig_roi = px.line(roi_df, x="Model Coverage", y="Recovered ($M)", 
                      title="ROI: More Coverage → More Recovery",
                      markers=True)
    st.plotly_chart(fig_roi, use_container_width=True)

    st.subheader("What-If: Add Analysts to Boost Recovery")
    # assume each analyst adds 5% coverage
    extra = st.slider("Extra Analysts", 0, 10, 0)
    baseline_analysts = 10
    base_coverage = 0.6
    new_coverage = min(1.0, base_coverage + 0.05 * extra)
    new_recovery = new_coverage * dollars_recovered
    st.metric("New Coverage", f"{new_coverage:.1%}")
    st.metric("New Recovered ($M)", f"{new_recovery/1e6:.2f}")

    st.markdown("---")

    st.subheader("Seasonal Claim Volume & Forecast")
    # simulate monthly volumes
    months = pd.date_range("2023-01-01", periods=24, freq="M")
    base = 3000
    season = 500 * np.sin(2 * np.pi * (np.arange(len(months)) % 12) / 12)
    noise = np.random.normal(0, 200, size=len(months))
    volumes = base + season + noise
    vol_df = pd.DataFrame({"Month": months, "Claims": volumes.astype(int)})
    # forecast next 6 months by repeating last year's seasonality
    future_months = pd.date_range("2025-01-31", periods=6, freq="M")
    future_vol = base + 500 * np.sin(2 * np.pi * (np.arange(6) % 12) / 12)
    future_df = pd.DataFrame({"Month": future_months, "Claims": future_vol.astype(int)})
    forecast_df = pd.concat([vol_df, future_df], ignore_index=True)

    fig_vol = px.line(forecast_df, x="Month", y="Claims", 
                      title="Claims Volume (Actual vs. Forecast)", 
                      markers=True)
    st.plotly_chart(fig_vol, use_container_width=True)

# === Visualizations Tab ===
with tabs[1]:
    st.header("Interactive Visualizations")

    st.subheader("Claim Cost Distribution")
    fig_hist = px.histogram(
        df_filtered,
        x="cost_to_plot",
        nbins=30,
        title=f"{log_option} Claim Cost Distribution",
        labels={"cost_to_plot": "Log(Claim Cost + 1)" if log_option == 'Log' else "Claim Cost"}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Cost Distribution by Gender")
    fig_box_gender = px.box(
        df_filtered,
        x=gender_col,
        y=cost_col,
        title="Claim Cost by Gender",
        points="all",
        labels={gender_col: "Gender", cost_col: "Claim Cost"}
    )
    st.plotly_chart(fig_box_gender, use_container_width=True)

    st.subheader("Cost Distribution by Provider Type")
    fig_box_provider = px.box(
        df_filtered,
        x=provider_col,
        y=cost_col,
        title="Claim Cost by Provider Type",
        points="all",
        labels={provider_col: "Provider Type", cost_col: "Claim Cost"}
    )
    st.plotly_chart(fig_box_provider, use_container_width=True)

    st.subheader("Correlation Heatmap")
    numeric_df = df_filtered.select_dtypes(include=[np.number])
    # Remove constant columns
    numeric_df = numeric_df.loc[:, numeric_df.apply(pd.Series.nunique) > 1]
    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap",
            color_continuous_scale='RdBu',
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough numeric columns to display correlation heatmap.")

# === Missing Data Tab ===
with tabs[2]:
    st.header("Missing Values Analysis")
    missing_before = df_filtered.isnull().sum()
    missing_before = missing_before[missing_before > 0]

    if missing_before.empty:
        st.success("No missing values in the current filtered dataset.")
    else:
        missing_df = pd.DataFrame({
            "feature": missing_before.index,
            "missing_count": missing_before.values,
            "missing_percent": (missing_before.values / len(df_filtered) * 100).round(2)
        })
        st.subheader("Missing Values Before Imputation")
        fig_missing_before = px.bar(
            missing_df,
            x="feature",
            y="missing_count",
            title="Count of Missing Values by Feature (Before Imputation)",
            text="missing_count"
        )
        st.plotly_chart(fig_missing_before, use_container_width=True)

        # Impute missing values
        df_imputed = df_filtered.copy()
        for col in df_imputed.columns:
            if df_imputed[col].isnull().any():
                if df_imputed[col].dtype in [np.float64, np.int64]:
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
                else:
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode()[0])

        missing_after = df_imputed.isnull().sum()
        missing_after = missing_after[missing_after > 0]

        if missing_after.empty:
            st.subheader("Missing Values After Imputation")
            st.success("All missing values have been imputed.")
        else:
            missing_df2 = pd.DataFrame({
                "feature": missing_after.index,
                "missing_count": missing_after.values,
                "missing_percent": (missing_after.values / len(df_filtered) * 100).round(2)
            })
            st.subheader("Missing Values After Imputation")
            fig_missing_after = px.bar(
                missing_df2,
                x="feature",
                y="missing_count",
                title="Count of Missing Values by Feature (After Imputation)",
                text="missing_count"
            )
            st.plotly_chart(fig_missing_after, use_container_width=True)
