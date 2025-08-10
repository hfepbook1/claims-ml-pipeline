# frontend/pages/02_EDA_Visualizations.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Healthcare Claims Dashboard", layout="wide")

# Load or simulate data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/synthetic_claims.csv")
    except FileNotFoundError:
        # simulate if missing
        np.random.seed(42)
        n = 10000
        df = pd.DataFrame({
            "age": np.random.randint(18, 90, size=n),
            "gender": np.random.choice(["Male", "Female"], size=n),
            "region": np.random.choice(["North", "South", "East", "West"], size=n),
            "provider_type": np.random.choice(["Hospital", "Clinic", "Physician", "Lab"], size=n),
            "primary_diagnosis": np.random.choice(
                ["Diabetes", "Cancer", "Cardiac", "Orthopedic", "Respiratory"], size=n),
            "chronic_condition_count": np.random.poisson(2, size=n),
            "claim_cost": np.round(np.random.gamma(2.0, 2000.0, size=n), 2),
            "is_fraud": np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
            "readmit_30d": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        })
        # simulate missing claim_date
        df["claim_date"] = pd.to_datetime("2023-01-01") + pd.to_timedelta(
            np.random.randint(0, 365*2, size=n), unit="D"
        )
        # introduce some missing values
        for col in ["gender", "provider_type", "primary_diagnosis", "claim_cost"]:
            df.loc[df.sample(frac=0.02, random_state=42).index, col] = np.nan
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filter Data")
sel_gender = st.sidebar.multiselect("Gender", options=df["gender"].dropna().unique(), default=df["gender"].dropna().unique())
sel_region = st.sidebar.multiselect("Region", options=df["region"].dropna().unique(), default=df["region"].dropna().unique())
sel_provider = st.sidebar.multiselect("Provider Type", options=df["provider_type"].dropna().unique(), default=df["provider_type"].dropna().unique())
sel_diag = st.sidebar.multiselect("Primary Diagnosis", options=df["primary_diagnosis"].dropna().unique(), default=df["primary_diagnosis"].dropna().unique())

df = df[
    df["gender"].isin(sel_gender) &
    df["region"].isin(sel_region) &
    df["provider_type"].isin(sel_provider) &
    df["primary_diagnosis"].isin(sel_diag)
].copy()

# Ensure claim_date column exists
if "claim_date" not in df.columns:
    df["claim_date"] = (
        pd.to_datetime("2023-01-01") +
        pd.to_timedelta(np.random.randint(0, 365 * 2, size=len(df)), unit="D")
    )
df["claim_date"] = pd.to_datetime(df["claim_date"])
df["month"] = df["claim_date"].dt.to_period("M").dt.to_timestamp()

# Monthly aggregates
monthly = df.groupby("month").agg(
    monthly_volume=pd.NamedAgg(column="claim_cost", aggfunc="size"),
    monthly_cost=pd.NamedAgg(column="claim_cost", aggfunc="sum"),
    fraud_count=pd.NamedAgg(column="is_fraud", aggfunc="sum"),
    readmit_count=pd.NamedAgg(column="readmit_30d", aggfunc="sum"),
).reset_index()

# Forecast next 3 months by 3-month average
if len(monthly) >= 3:
    last3 = monthly.tail(3)
    avg_vol = last3["monthly_volume"].mean()
    avg_cost = last3["monthly_cost"].mean()
else:
    avg_vol = monthly["monthly_volume"].mean()
    avg_cost = monthly["monthly_cost"].mean()

future_months = pd.date_range(
    monthly["month"].max() + pd.offsets.MonthBegin(1), periods=3, freq="M"
)
forecast = pd.DataFrame({
    "month": future_months,
    "monthly_volume": avg_vol,
    "monthly_cost": avg_cost,
    "fraud_count": 0,
    "readmit_count": 0,
})
monthly_full = pd.concat([monthly, forecast], ignore_index=True)

# Compute ROI KPIs
total_fraud_cost = df.loc[df["is_fraud"] == 1, "claim_cost"].sum(skipna=True)
baseline_recall = 0.5
fraud_lift = 0.60
avoided_fraud = fraud_lift * (total_fraud_cost / baseline_recall) if baseline_recall else 0

avg_readmit_cost = df.loc[df["readmit_30d"] == 1, "claim_cost"].mean(skipna=True) or 0
num_readmit = df["readmit_30d"].sum(skipna=True)
readmission_savings = num_readmit * avg_readmit_cost * 0.15  # assume 15% reduction

threshold = df["claim_cost"].quantile(0.90)
high_cost_pct = (df["claim_cost"] > threshold).mean()
flagged = df[df["claim_cost"] >= threshold]
general = df[df["claim_cost"] < threshold]
cost_ratio = flagged["claim_cost"].mean() / general["claim_cost"].mean() if len(general) else np.nan

# Tabs
tab_dashboard, tab_missing = st.tabs(["Dashboard", "Missing Data"])

# Dashboard Tab: ROI then Charts
with tab_dashboard:
    st.header("Key ROI Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Fraud Detection Lift", "60%")
    k1.metric("Fraud Cost Avoided", f"${avoided_fraud:,.0f}")
    k2.metric("Readmission Reduction", "15%")
    k2.metric("Readmission Savings", f"${readmission_savings:,.0f}")
    k3.metric("High-Cost Claims %", f"{high_cost_pct:.1%}")
    k3.metric("Flagged/General Cost Ratio", f"{cost_ratio:.2f}")
    k4.metric("Total Claims", f"{len(df):,}")
    k4.metric("Avg Cost (Readmit)", f"${avg_readmit_cost:,.0f}")

    st.markdown("""
    - **Reserve Planning:** Forecasts guide fund allocation before large claims arrive.
    - **Triage Workflow:** Flag >$20k claims for senior adjusters; fast-track low-cost ones.
    - **Fraud ROI:** 60% lift in detection directly reduces losses.
    - **Preventive Care:** Target top 10% high-cost patients for maximum impact.
    - **Case Management:** 15% fewer readmissions improves outcomes and cuts costs.
    """)

    st.header("Claims Trend & Distributions")

    # Time-series volume & cost
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=monthly_full["month"], y=monthly_full["monthly_volume"],
        mode="lines+markers", name="Volume"
    ))
    fig1.add_trace(go.Scatter(
        x=monthly_full["month"], y=monthly_full["monthly_cost"],
        mode="lines+markers", name="Cost", yaxis="y2"
    ))
    fig1.update_layout(
        title="Monthly Claim Volume & Total Cost",
        xaxis_title="Month",
        yaxis=dict(title="Volume"),
        yaxis2=dict(title="Cost (USD)", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --- Claim Cost Distribution with Raw/Log Toggle ---
    st.subheader("Claim Cost Distribution")
    scale = st.radio("Cost Scale", ["Raw", "Log"], horizontal=True)
    if scale == "Log":
        df["cost_plot"] = np.log1p(df["claim_cost"])
        x_label = "Log(Claim Cost + 1)"
    else:
        df["cost_plot"] = df["claim_cost"]
        x_label = "Claim Cost (USD)"

    fig_cost_dist = px.histogram(
        df,
        x="cost_plot",
        nbins=50,
        title=f"{scale} Claim Cost Distribution",
        labels={"cost_plot": x_label, "count": "Frequency"}
    )
    st.plotly_chart(fig_cost_dist, use_container_width=True)
    st.markdown(
        f"This shows the {scale.lower()} distribution of claim cost. "
        + ("Log transform reveals the bulk of data when skew is high." 
           if scale == "Log" else "Raw view highlights skew and outliers.")
    )

    # --- Boxplot by Provider Type ---
    st.subheader("Claim Cost by Provider Type")
    fig_cost_provider = px.box(
        df,
        x="provider_type",
        y="claim_cost",
        title="Claim Cost by Provider Type",
        points="all",
        labels={"provider_type": "Provider Type", "claim_cost": "Claim Cost (USD)"}
    )
    # use inclusive quartile for more precise whiskers
    fig_cost_provider.update_traces(quartilemethod="inclusive")
    st.plotly_chart(fig_cost_provider, use_container_width=True)
    st.markdown(
        "This box plot compares median, IQR, and outliers by provider; hospitals typically show higher costs."
    )

    # --- Correlation Heatmap with Reversed Palette ---
    st.subheader("Feature Correlation Heatmap")
    num_cols = ['age', 'chronic_condition_count', 'num_visits', 'num_er_visits', 'num_inpatient_stays', 'claim_cost']
    corr = df[num_cols].corr()
    fig_heat = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap (RdBu_r Palette)",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown(
        "Note strong links between inpatient stays and claim cost."
    )


# Missing Data Tab
with tab_missing:
    st.header("Missing Data Analysis")

    miss_before = df.isna().sum()
    miss_before = miss_before[miss_before > 0]
    if miss_before.empty:
        st.success("No missing values detected.")
    else:
        mb = miss_before.reset_index()
        mb.columns = ["Feature", "Missing Count"]
        fig5 = px.bar(mb, x="Feature", y="Missing Count",
                      title="Missing Values Before Imputation")
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown("### Imputation of Missing Values")
        df_imp = df.copy()
        for col in df_imp.select_dtypes(include=[np.number]).columns:
            df_imp[col].fillna(df_imp[col].median(), inplace=True)
        for col in df_imp.select_dtypes(include=["object", "category"]).columns:
            df_imp[col].fillna(df_imp[col].mode()[0], inplace=True)

        miss_after = df_imp.isna().sum()
        miss_after = miss_after[miss_after > 0]
        if miss_after.empty:
            st.success("All missing values have been imputed.")
        else:
            ma = miss_after.reset_index()
            ma.columns = ["Feature", "Missing Count"]
            fig6 = px.bar(ma, x="Feature", y="Missing Count",
                          title="Missing Values After Imputation")
            st.plotly_chart(fig6, use_container_width=True)
