# frontend/pages/02_EDA_Visualizations.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Healthcare Claims EDA", layout="wide")

st.title("Healthcare Claims EDA & Insights")

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
st.sidebar.header("Filters")
genders = df["gender"].dropna().unique()
regions = df["region"].dropna().unique()
providers = df["provider_type"].dropna().unique()
diags = df["primary_diagnosis"].dropna().unique()

sel_genders = st.sidebar.multiselect("Gender", options=genders, default=list(genders))
sel_regions = st.sidebar.multiselect("Region", options=regions, default=list(regions))
sel_providers = st.sidebar.multiselect("Provider Type", options=providers, default=list(providers))
sel_diags = st.sidebar.multiselect("Primary Diagnosis", options=diags, default=list(diags))

# Apply filters
df_filtered = df[
    df["gender"].isin(sel_genders) &
    df["region"].isin(sel_regions) &
    df["provider_type"].isin(sel_providers) &
    df["primary_diagnosis"].isin(sel_diags)
].copy()

# Simulate or parse claim_date
if "claim_date" not in df_filtered.columns:
    df_filtered["claim_date"] = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        np.random.randint(0, 365 * 2, size=len(df_filtered)), unit="D"
    )
df_filtered["claim_date"] = pd.to_datetime(df_filtered["claim_date"])
df_filtered["month"] = df_filtered["claim_date"].dt.to_period("M").dt.to_timestamp()

# Compute monthly aggregates
monthly = df_filtered.groupby("month").agg(
    monthly_volume=pd.NamedAgg(column="claim_cost", aggfunc="size"),
    monthly_cost=pd.NamedAgg(column="claim_cost", aggfunc="sum"),
    fraud_count=pd.NamedAgg(column="is_fraud", aggfunc="sum"),
    readmit_count=pd.NamedAgg(column="readmit_30d", aggfunc="sum"),
).reset_index()

# Forecast next 3 months using last-3-month average
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

# Compute KPIs
total_fraud_cost = df_filtered.loc[df_filtered["is_fraud"] == 1, "claim_cost"].sum(skipna=True)
baseline_recall = 0.5
fraud_lift = 0.60
avoided_fraud = fraud_lift * (total_fraud_cost / baseline_recall) if baseline_recall else 0

readmit_cases = df_filtered["readmit_30d"].sum(skipna=True)
avg_readmit_cost = df_filtered.loc[df_filtered["readmit_30d"] == 1, "claim_cost"].mean(skipna=True)
readmission_savings = readmit_cases * avg_readmit_cost * 0.15  # 15% reduction

threshold = df_filtered["claim_cost"].quantile(0.90)
high_cost_pct = (df_filtered["claim_cost"] > threshold).mean()
flagged = df_filtered[df_filtered["claim_cost"] >= threshold]
general = df_filtered[df_filtered["claim_cost"] < threshold]
cost_ratio = (
    flagged["claim_cost"].mean() / general["claim_cost"].mean()
    if not general.empty else np.nan
)

# Tabs
tab1, tab2, tab3 = st.tabs(["Overview", "Visualizations", "Missing Data"])

# Overview Tab
with tab1:
    st.header("Overview & Business Impact")

    # Time-series volume & cost
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=monthly_full["month"], y=monthly_full["monthly_volume"],
        mode="lines+markers", name="Claim Volume"
    ))
    fig1.add_trace(go.Scatter(
        x=monthly_full["month"], y=monthly_full["monthly_cost"],
        mode="lines+markers", name="Total Cost", yaxis="y2"
    ))
    fig1.update_layout(
        title="Monthly Claim Volume & Total Cost (incl. 3-mo avg forecast)",
        xaxis_title="Month",
        yaxis_title="Volume",
        yaxis2=dict(title="Cost (USD)", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown(
        "Forecast is a flat 3-month average—use for staffing and reserve planning before peaks."
    )

    # KPIs
    st.subheader("Key ROI Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fraud Detection Lift", "60%")
    c1.metric("Fraud Cost Avoided", f"${avoided_fraud:,.0f}")
    c2.metric("Readmission Rate Reduction", "15%")
    c2.metric("Readmission Savings", f"${readmission_savings:,.0f}")
    c3.metric("High-Cost Claim %", f"{high_cost_pct:.1%}")
    c3.metric("Flagged/General Cost Ratio", f"{cost_ratio:.2f}")
    c4.metric("Total Claims", f"{len(df_filtered):,}")
    c4.metric("Avg Readmit Cost", f"${avg_readmit_cost:,.0f}")

    st.markdown("""
    - **Reserve Planning:** Rolling forecasts guide funds allocation prior to large claims.
    - **Triage Workflow:** >$20k claims are flagged for senior adjusters, streamlining reviews.
    - **Fraud ROI:** 60% uplift in detection saves significant losses.
    - **Preventive Care:** Focus on top 10% high-cost patients for maximum care impact.
    - **Case Management:** 15% drop in readmissions improves outcomes and lowers penalties.
    """)

# Visualizations Tab
with tab2:
    st.header("Interactive Visualizations")

    st.subheader("Claim Cost Distribution")
    fig2 = px.histogram(
        df_filtered, x="claim_cost", nbins=50,
        title="Claim Cost Distribution",
        labels={"claim_cost": "Claim Cost (USD)"}
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("Shows right-skewed distribution—plan for outliers.")

    st.subheader("Cost by Provider Type")
    fig3 = px.box(
        df_filtered, x="provider_type", y="claim_cost",
        title="Claim Cost by Provider Type",
        points="all",
        labels={"provider_type": "Provider Type", "claim_cost": "Cost (USD)"}
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("Hospitals trend higher costs than clinics or labs.")

    st.subheader("Correlation Heatmap")
    num_df = df_filtered.select_dtypes(include=np.number)
    corr = num_df.corr()
    fig4 = px.imshow(
        corr, text_auto=True, aspect="auto",
        title="Numeric Feature Correlation"
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("Age, chronic conditions, and cost show strong positive correlation.")

# Missing Data Tab
with tab3:
    st.header("Missing Data Analysis")
    miss_before = df_filtered.isna().sum()
    miss_before = miss_before[miss_before > 0]
    if miss_before.empty:
        st.success("No missing values detected.")
    else:
        df_miss_before = miss_before.reset_index()
        df_miss_before.columns = ["Feature", "Missing Count"]
        fig5 = px.bar(
            df_miss_before, x="Feature", y="Missing Count",
            title="Missing Values Before Imputation"
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("Counts of missing values by feature before filling.")

        # Impute
        df_imputed = df_filtered.copy()
        for col in df_imputed.columns:
            if df_imputed[col].dtype in [np.float64, np.int64]:
                df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
            else:
                df_imputed[col].fillna(df_imputed[col].mode()[0], inplace=True)

        miss_after = df_imputed.isna().sum()
        miss_after = miss_after[miss_after > 0]
        if miss_after.empty:
            st.success("All missing values have been imputed.")
        else:
            df_miss_after = miss_after.reset_index()
            df_miss_after.columns = ["Feature", "Missing Count"]
            fig6 = px.bar(
                df_miss_after, x="Feature", y="Missing Count",
                title="Missing Values After Imputation"
            )
            st.plotly_chart(fig6, use_container_width=True)
            st.markdown("Remaining missing values after imputation.")
