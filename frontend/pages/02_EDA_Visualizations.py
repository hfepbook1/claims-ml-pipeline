# frontend/pages/02_EDA_Visualizations.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Healthcare Claims EDA", layout="wide")

st.title("Healthcare Claims Exploratory Data Analysis")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/synthetic_claims.csv", parse_dates=["claim_date"])
    return df

df = load_data()
if df.empty:
    st.error("The dataset is empty or not found.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Filter Claims")
gender_opts = df["gender"].dropna().unique().tolist()
region_opts = df["region"].dropna().unique().tolist()
provider_opts = df["provider_type"].dropna().unique().tolist()
diag_opts = df["primary_diagnosis"].dropna().unique().tolist()

selected_genders = st.sidebar.multiselect("Gender", gender_opts, default=gender_opts)
selected_regions = st.sidebar.multiselect("Region", region_opts, default=region_opts)
selected_providers = st.sidebar.multiselect("Provider Type", provider_opts, default=provider_opts)
selected_diags = st.sidebar.multiselect("Primary Diagnosis", diag_opts, default=diag_opts)

df_filtered = df[
    df["gender"].isin(selected_genders) &
    df["region"].isin(selected_regions) &
    df["provider_type"].isin(selected_providers) &
    df["primary_diagnosis"].isin(selected_diags)
].copy()
if df_filtered.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# Ensure month column
df_filtered["claim_date"] = pd.to_datetime(df_filtered["claim_date"])
df_filtered["month"] = df_filtered["claim_date"].dt.to_period("M").dt.to_timestamp()

# --- Compute Monthly Aggregates and Forecast ---
monthly = df_filtered.groupby("month").agg(
    volume=("claim_cost", "count"),
    total_cost=("claim_cost", "sum")
).reset_index()

# Forecast next 3 months with last-3-mo average
if len(monthly) >= 3:
    avg_vol = monthly["volume"].tail(3).mean()
    avg_cost = monthly["total_cost"].tail(3).mean()
else:
    avg_vol = monthly["volume"].mean()
    avg_cost = monthly["total_cost"].mean()
future_months = pd.date_range(
    monthly["month"].max() + pd.offsets.MonthBegin(1),
    periods=3,
    freq="M"
)
forecast = pd.DataFrame({
    "month": future_months,
    "volume": avg_vol,
    "total_cost": avg_cost
})
monthly_full = pd.concat([monthly.rename(columns={"volume":"volume","total_cost":"total_cost"}), forecast], ignore_index=True)

# --- KPIs ---
# Fraud cost avoided
fraud_cost = df_filtered.loc[df_filtered["is_fraud"] == 1, "claim_cost"].sum(skipna=True)
baseline_recall = 0.5
fraud_lift = 0.60
avoided_fraud = fraud_lift * (fraud_cost / baseline_recall) if baseline_recall else 0.0

# Readmission savings
readmit_cases = df_filtered["readmit_30d"].sum()
avg_readmit_cost = df_filtered.loc[df_filtered["readmit_30d"] == 1, "claim_cost"].mean(skipna=True)
if np.isnan(avg_readmit_cost):
    avg_readmit_cost = 0.0
readmit_drop = 0.15
saved_readmit = readmit_cases * avg_readmit_cost * readmit_drop

# High-cost ratio
threshold = df_filtered["claim_cost"].quantile(0.90)
high_cost_ratio = (df_filtered["claim_cost"] > threshold).mean()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Overview", "Visualizations", "Missing Data"])

with tab1:
    st.header("Overview & Business Impact")

    # Time-series chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=monthly_full["month"], y=monthly_full["volume"],
        name="Claim Volume", mode="lines+markers"
    ))
    fig1.add_trace(go.Scatter(
        x=monthly_full["month"], y=monthly_full["total_cost"],
        name="Total Cost", mode="lines+markers", yaxis="y2"
    ))
    fig1.update_layout(
        title="Monthly Claim Volume & Total Cost (with Forecast)",
        xaxis_title="Month",
        yaxis=dict(title="Volume"),
        yaxis2=dict(title="Cost (USD)", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown(
        "Above: we roll up monthly claim counts and total cost, then append a 3-month average forecast. "
        "Use this to **plan staffing**, **reserve funds**, and **allocate resources** ahead of seasonal peaks."
    )

    # KPI metrics
    k1, k2, k3 = st.columns(3)
    k1.metric("Fraud Cost Avoided", f"${avoided_fraud:,.0f}", delta="60% uplift")
    k2.metric("Readmission Savings", f"${saved_readmit:,.0f}", delta="15% reduction")
    k3.metric("High-Cost Claims Ratio", f"{high_cost_ratio:.1%}", delta="Top 10% threshold")
    st.markdown("""
    - **Fraud ROI:** A 60% lift in detection translates into real dollars saved by preventing high-cost fraud cases.
    - **Readmission Impact:** Reducing 30-day readmissions by 15% saves on average per-patient costs.
    - **Triage Workflow:** Flag high-cost claims (>90th percentile) for senior review; low-cost ones can be expedited.
    """)

with tab2:
    st.header("Interactive Charts")

    st.subheader("Claim Cost Distribution")
    fig2 = px.histogram(
        df_filtered, x="claim_cost", nbins=40,
        title="Claim Cost Distribution",
        labels={"claim_cost":"Claim Cost (USD)"}
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("Most claims cluster at lower costs with a long tail; informs **reserve buffers**.")

    st.subheader("Cost by Provider Type")
    fig3 = px.box(
        df_filtered, x="provider_type", y="claim_cost", points="all",
        title="Claim Cost by Provider Type",
        labels={"provider_type":"Provider Type","claim_cost":"Cost (USD)"}
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("Hospitals tend to have higher median and more variable costs.")

    st.subheader("Feature Correlations")
    num_cols = ["age","chronic_condition_count","claim_cost","is_fraud","readmit_30d"]
    corr = df_filtered[num_cols].corr()
    fig4 = px.imshow(
        corr, text_auto=True, aspect="auto",
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("Strong positive correlation between chronic conditions, age, and claim cost.")

with tab3:
    st.header("Missing Data & Imputation")

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values detected.")
    else:
        miss_df = missing.reset_index()
        miss_df.columns = ["Feature","Count"]
        fig5 = px.bar(
            miss_df, x="Feature", y="Count",
            title="Missing Values Before Imputation"
        )
        st.plotly_chart(fig5, use_container_width=True)

        # Impute
        df_imp = df.copy()
        for col in df_imp.select_dtypes(include=[np.number]):
            df_imp[col].fillna(df_imp[col].median(), inplace=True)
        for col in df_imp.select_dtypes(include=["object","datetime"]):
            if df_imp[col].isna().any():
                df_imp[col].fillna(df_imp[col].mode()[0], inplace=True)

        missing_after = df_imp.isna().sum()
        missing_after = missing_after[missing_after > 0]
        if missing_after.empty:
            st.success("All missing values have been imputed.")
        else:
            miss_after_df = missing_after.reset_index()
            miss_after_df.columns = ["Feature","Count"]
            fig6 = px.bar(
                miss_after_df, x="Feature", y="Count",
                title="Missing Values After Imputation"
            )
            st.plotly_chart(fig6, use_container_width=True)
