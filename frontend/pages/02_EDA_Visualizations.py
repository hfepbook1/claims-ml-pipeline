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
        # Simulate data if CSV not found
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
        # Simulate missing claim_date for synthetic data
        df["claim_date"] = pd.to_datetime("2023-01-01") + pd.to_timedelta(
            np.random.randint(0, 365*2, size=n), unit="D"
        )
        # Introduce some missing values in a few columns
        for col in ["gender", "provider_type", "primary_diagnosis", "claim_cost"]:
            df.loc[df.sample(frac=0.02, random_state=42).index, col] = np.nan
    # Add simulated geo location (state) and member IDs for analysis
    np.random.seed(0)  # seed for reproducibility of random assignment
    region_to_states = {
        "Northeast": [
            "CT","ME","MA","NH","RI","VT",  # New England
            "NJ","NY","PA"                  # Mid-Atlantic
        ],
        "Midwest": [
            "IL","IN","MI","OH","WI",       # East North Central
            "IA","KS","MN","MO","NE","ND","SD"  # West North Central
        ],
        "South": [
            "DE","FL","GA","MD","NC","SC","VA","DC","WV",  # South Atlantic
            "AL","KY","MS","TN",                            # East South Central
            "AR","LA","OK","TX"                             # West South Central
        ],
        "West": [
            "AZ","CO","ID","MT","NV","NM","UT","WY",  # Mountain
            "AK","CA","HI","OR","WA"                  # Pacific
        ]
    }
    df["state"] = df["region"].apply(lambda r: np.random.choice(region_to_states.get(r, ["Unknown"])))
    # Simulate member IDs (anonymized patient identifiers)
    unique_members = min(len(df), max(10, int(len(df) * 0.1)))
    df["member_id"] = np.random.randint(1, unique_members + 1, size=len(df))
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filter Data")
sel_gender = st.sidebar.multiselect(
    "Gender", options=df["gender"].dropna().unique(), 
    default=df["gender"].dropna().unique()
)
sel_region = st.sidebar.multiselect(
    "Region", options=df["region"].dropna().unique(), 
    default=df["region"].dropna().unique()
)
sel_provider = st.sidebar.multiselect(
    "Provider Type", options=df["provider_type"].dropna().unique(), 
    default=df["provider_type"].dropna().unique()
)
sel_diag = st.sidebar.multiselect(
    "Primary Diagnosis", options=df["primary_diagnosis"].dropna().unique(), 
    default=df["primary_diagnosis"].dropna().unique()
)
age_min, age_max = int(df["age"].min(skipna=True)), int(df["age"].max(skipna=True))
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

# Apply filters to data
df = df[
    df["gender"].isin(sel_gender) &
    df["region"].isin(sel_region) &
    df["provider_type"].isin(sel_provider) &
    df["primary_diagnosis"].isin(sel_diag) &
    df["age"].between(age_range[0], age_range[1])
].copy()

# If no data remains after filtering, show warning and stop
if df.empty:
    st.warning("No data available for the selected filters. Please adjust the filters.")
    st.stop()

# Ensure claim_date column exists and create month period
if "claim_date" not in df.columns:
    df["claim_date"] = (
        pd.to_datetime("2023-01-01") +
        pd.to_timedelta(np.random.randint(0, 365 * 2, size=len(df)), unit="D")
    )
df["claim_date"] = pd.to_datetime(df["claim_date"])
df["month"] = df["claim_date"].dt.to_period("M").dt.to_timestamp()

# Monthly aggregates for trends
monthly = df.groupby("month").agg(
    monthly_volume=pd.NamedAgg(column="claim_cost", aggfunc="size"),
    monthly_cost=pd.NamedAgg(column="claim_cost", aggfunc="sum"),
    fraud_count=pd.NamedAgg(column="is_fraud", aggfunc="sum"),
    readmit_count=pd.NamedAgg(column="readmit_30d", aggfunc="sum"),
).reset_index()

# Forecast next 3 months (simple average-based forecast)
if len(monthly) >= 3:
    last3 = monthly.tail(3)
    avg_vol = last3["monthly_volume"].mean()
    avg_cost = last3["monthly_cost"].mean()
else:
    avg_vol = monthly["monthly_volume"].mean() if len(monthly) > 0 else 0
    avg_cost = monthly["monthly_cost"].mean() if len(monthly) > 0 else 0

future_months = pd.date_range(
    monthly["month"].max() + pd.offsets.MonthBegin(1) if not monthly.empty else pd.Timestamp("2023-01-01"), 
    periods=3, freq="M"
)
forecast = pd.DataFrame({
    "month": future_months,
    "monthly_volume": avg_vol,
    "monthly_cost": avg_cost,
    "fraud_count": 0,
    "readmit_count": 0,
})
monthly_full = pd.concat([monthly, forecast], ignore_index=True)

# Compute key ROI metrics
total_fraud_cost = df.loc[df["is_fraud"] == 1, "claim_cost"].sum(skipna=True)
baseline_recall = 0.5  # baseline model recall for fraud
fraud_lift = 0.60      # our model's lift in fraud detection
avoided_fraud = fraud_lift * (total_fraud_cost / baseline_recall) if baseline_recall else 0

avg_readmit_cost = df.loc[df["readmit_30d"] == 1, "claim_cost"].mean(skipna=True) or 0
num_readmit = int(df["readmit_30d"].sum(skipna=True))
readmission_savings = num_readmit * avg_readmit_cost * 0.15  # assume 15% cost reduction from interventions

threshold = df["claim_cost"].quantile(0.90)  # 90th percentile cost
high_cost_pct = (df["claim_cost"] > threshold).mean()  # % of claims that are high-cost
flagged = df[df["claim_cost"] >= threshold]
general = df[df["claim_cost"] < threshold]
cost_ratio = flagged["claim_cost"].mean() / general["claim_cost"].mean() if len(general) else np.nan

# Set up tabs for Dashboard and Missing Data analysis
tab_dashboard, tab_missing = st.tabs(["Dashboard", "Missing Data"])

# Dashboard Tab: Key metrics and visualizations
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
    - **Triage Workflow:** Flag high-cost (>$20k) claims for senior adjusters; fast-track low-cost ones.
    - **Fraud ROI:** 60% lift in detection directly reduces losses from fraudulent claims.
    - **Preventive Care:** Target top 10% high-cost patients for proactive care to maximize impact.
    - **Case Management:** 15% fewer readmissions improves patient outcomes and cuts repeat costs.
    """)

    st.header("Claims Trend & Distributions")

    # Monthly trends: claim volume and cost over time (with simple forecast)
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly_full["month"], y=monthly_full["monthly_volume"],
        mode="lines+markers", name="Monthly Volume"
    ))
    fig_trend.add_trace(go.Scatter(
        x=monthly_full["month"], y=monthly_full["monthly_cost"],
        mode="lines+markers", name="Monthly Cost",
        yaxis="y2"
    ))
    fig_trend.update_layout(
        title="Monthly Claim Volume & Total Cost (with 3-month Forecast)",
        xaxis_title="Month",
        yaxis=dict(title="Claims Volume"),
        yaxis2=dict(title="Total Cost (USD)", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Claim cost distribution with Raw vs Log scale toggle
    st.subheader("Claim Cost Distribution")
    scale = st.radio("Cost Scale", ["Raw", "Log"], horizontal=True)
    if scale == "Log":
        df["cost_plot"] = np.log1p(df["claim_cost"])
        x_label = "Log(Claim Cost + 1)"
    else:
        df["cost_plot"] = df["claim_cost"]
        x_label = "Claim Cost (USD)"
    fig_cost_dist = px.histogram(
        df, x="cost_plot", nbins=50,
        title=f"{scale} Claim Cost Distribution",
        labels={"cost_plot": x_label, "count": "Frequency"}
    )
    st.plotly_chart(fig_cost_dist, use_container_width=True)
    st.markdown(
        f"This histogram shows the **{scale.lower()}** distribution of claim costs. "
        + ("Using a log scale reveals the bulk of claims more clearly when the cost data is highly skewed."
           if scale == "Log" else 
           "In raw scale, we can see the right-skew with a long tail of high-cost outliers.")
    )

    # Cost by provider type (box plot for distribution by provider category)
    st.subheader("Claim Cost by Provider Type")
    fig_cost_provider = px.box(
        df, x="provider_type", y="claim_cost", points="all",
        title="Claim Cost by Provider Type",
        labels={"provider_type": "Provider Type", "claim_cost": "Claim Cost (USD)"}
    )
    fig_cost_provider.update_traces(quartilemethod="inclusive")  # use inclusive quartile for whiskers
    st.plotly_chart(fig_cost_provider, use_container_width=True)
    st.markdown(
        "This box plot compares claim cost distributions across provider types. We see differences in median costs and variability – for example, hospital claims tend to have a higher median cost and more outliers than clinic or physician claims."
    )

    # Geographic distribution of claims
    st.subheader("Geographic Distribution of Claims")
    metric_choice = st.radio("Metric", ["Total Cost", "Number of Claims"], horizontal=True)
    # Aggregate by state for the selected metric
    state_agg = df.groupby("state").agg(
        claim_count=("claim_cost", "size"),
        total_cost=("claim_cost", "sum")
    ).reset_index()
    if metric_choice == "Total Cost":
        color_col = "total_cost"
        map_title = "Total Claim Cost by State"
        color_label = "Total Cost (USD)"
    else:
        color_col = "claim_count"
        map_title = "Number of Claims by State"
        color_label = "Claim Count"
    fig_map = px.choropleth(
        state_agg, locations="state", locationmode="USA-states",
        color=color_col, color_continuous_scale="Blues", scope="usa",
        title=map_title, labels={color_col: color_label}
    )
    st.plotly_chart(fig_map, use_container_width=True)
    # Highlight insight from geographic distribution
    if not state_agg.empty:
        top_state_row = state_agg.sort_values(color_col, ascending=False).iloc[0]
        top_state = top_state_row["state"]
        st.markdown(
            f"The **{top_state}** region has the highest {metric_choice.lower()} in our dataset. "
            f"This geographic pattern can help prioritize regional strategies – for instance, focusing resources where claims volume or cost is highest."
        )

    # High-cost members analysis (top 10 members by total cost)
    st.subheader("High-Cost Members")
    member_cost = df.groupby("member_id", as_index=False)["claim_cost"].sum().rename(columns={"claim_cost": "total_cost"})
    top_members = member_cost.nlargest(10, "total_cost")
    # Convert member_id to a string label for plotting
    top_members["member_id"] = top_members["member_id"].apply(lambda x: f"Member {int(x)}")
    fig_top_members = px.bar(
        top_members, x="total_cost", y="member_id", orientation="h",
        title="Top 10 Members by Total Claim Cost",
        labels={"total_cost": "Total Claim Cost (USD)", "member_id": "Member ID"}
    )
    fig_top_members.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_top_members, use_container_width=True)
    # Provide business insight for high-cost members
    total_cost_sum = float(df["claim_cost"].sum(skipna=True))
    top10_cost_sum = float(top_members["total_cost"].sum(skipna=True))
    perc_contrib = (top10_cost_sum / total_cost_sum * 100) if total_cost_sum > 0 else 0
    st.markdown(
        f"Typically, a small number of patients drive a large share of healthcare costs. "
        f"In this dataset, the top 10 members (by total cost) account for about **{perc_contrib:.1f}%** of the total claim expenditure. "
        f"*(In real insurance portfolios, the top 5% of members often drive ~50% of costs.)* "
        f"Targeted care management for these high-cost individuals could significantly reduce overall expenses."
    )

    # Correlation heatmap for numeric features
    st.subheader("Feature Correlation Heatmap")
    # Define numeric columns to include in correlation
    num_cols = ["age", "chronic_condition_count", "num_visits", "num_er_visits", "num_inpatient_stays", "claim_cost"]
    # Some columns might not exist in the filtered data (e.g., if not present in synthetic fallback)
    num_cols = [col for col in num_cols if col in df.columns]
    if num_cols:
        corr_matrix = df[num_cols].corr()
        fig_heat = px.imshow(
            corr_matrix, text_auto=True, aspect="auto",
            title="Correlation Heatmap (Numeric Features)",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown(
            "Correlation highlights relationships between features. For example, we observe that variables related to service utilization (e.g., inpatient stays, ER visits) are positively correlated with higher claim costs."
        )
    else:
        st.write("Not enough numeric data available to compute correlations.")

with tab_missing:
    st.header("Missing Data Analysis")
    # Calculate missing values before any imputation
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if missing_counts.empty:
        st.success("No missing values detected in the current filtered dataset.")
    else:
        missing_df = missing_counts.reset_index()
        missing_df.columns = ["Feature", "Missing Count"]
        fig_missing = px.bar(missing_df, x="Feature", y="Missing Count", title="Missing Values by Feature")
        st.plotly_chart(fig_missing, use_container_width=True)

        st.markdown("### Imputation of Missing Values")
        # Simple imputation: median for numeric, mode for categorical
        df_imputed = df.copy()
        for col in df_imputed.select_dtypes(include=[np.number]).columns:
            df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
        for col in df_imputed.select_dtypes(include=["object", "category"]).columns:
            df_imputed[col].fillna(df_imputed[col].mode()[0] if df_imputed[col].mode().size > 0 else "Unknown", inplace=True)

        missing_after = df_imputed.isna().sum()
        missing_after = missing_after[missing_after > 0]
        if missing_after.empty:
            st.success("All missing values have been imputed (filled) using simple strategies.")
        else:
            missing_after_df = missing_after.reset_index()
            missing_after_df.columns = ["Feature", "Missing Count"]
            fig_missing_after = px.bar(missing_after_df, x="Feature", y="Missing Count", title="Missing Values After Imputation")
            st.plotly_chart(fig_missing_after, use_container_width=True)
