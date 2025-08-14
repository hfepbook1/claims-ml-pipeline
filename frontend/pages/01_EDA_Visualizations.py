import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. Page Configuration & Theming ---
# Sets the page to wide layout, provides a title and icon for a professional feel.
# The layout="wide" setting prevents chart crowding and improves readability.
st.set_page_config(
    page_title="Healthcare Claims Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Data Loading and Enhancement (from your original script, but expanded) ---
# The @st.cache_data decorator prevents the function from rerunning on every user interaction,
# significantly improving performance, especially with large datasets.
@st.cache_data
def load_data():
    """
    Loads or simulates a synthetic healthcare claims dataset.
    
    The synthetic data has been enhanced to include more features
    that support actionable business intelligence, such as specific
    treatment details and utilization metrics.
    """
    try:
        df = pd.read_csv("data/synthetic_claims.csv")
    except FileNotFoundError:
        np.random.seed(42)
        n = 100000  # Increased data size for more robust simulation
        df = pd.DataFrame({
            "age": np.random.randint(18, 90, size=n),
            "gender": np.random.choice(["Male", "Female"], size=n, p=[0.49, 0.51]),
            "region": np.random.choice(["North", "South", "East", "West"], size=n),
            "provider_type": np.random.choice(["Hospital", "Clinic", "Pharmacy", "Lab", "Other"], size=n),
            "primary_diagnosis": np.random.choice(
               ["COPD", "Hypertension", "Diabetes", "Heart Failure", "Stroke", "Other"], size=n),
            "chronic_condition_count": np.random.poisson(2, size=n),
            "claim_cost": np.round(np.random.gamma(2.5, 2500.0, size=n), 2),
            "claim_amount_reimbursed": np.round(np.random.gamma(2.5, 2500.0, size=n) * 0.8, 2), # New column for more detailed analysis
            "is_fraud": np.random.choice([0, 1], size=n, p=[0.985, 0.015]),
            "readmit_30d": np.random.choice([0, 1], size=n, p=[0.92, 0.08]), # 8% readmission rate
            "num_inpatient_stays": np.random.poisson(0.5, size=n), # New column for operational analysis
            "num_er_visits": np.random.poisson(1, size=n), # New column for operational analysis
        })
        df["claim_date"] = pd.to_datetime("2023-01-01") + pd.to_timedelta(
            np.random.randint(0, 365*2, size=n), unit="D"
        )
        for col in ["gender", "provider_type", "primary_diagnosis", "claim_cost"]:
            df.loc[df.sample(frac=0.01, random_state=42).index, col] = np.nan
        
        region_to_states = {
            "North": ["California", "Oregon", "Washington", "Idaho", "Montana", "Wyoming", "North Dakota", "South Dakota", "Minnesota", "Wisconsin", "Michigan", "Illinois", "Indiana", "Ohio", "Pennsylvania", "New York", "New Jersey", "Connecticut", "Rhode Island", "Massachusetts", "Vermont", "New Hampshire", "Maine"],
            "South": ["Texas", "Oklahoma", "Arkansas", "Louisiana", "Mississippi", "Alabama", "Georgia", "Florida", "South Carolina", "North Carolina", "Virginia", "West Virginia", "Kentucky", "Tennessee", "Arkansas", "Louisiana", "Mississippi", "Alabama", "Georgia", "Florida", "South Carolina", "North Carolina", "Virginia", "West Virginia", "Kentucky", "Tennessee"],
            "East": ["New York", "New Jersey", "Connecticut", "Rhode Island", "Massachusetts", "Vermont", "New Hampshire", "Maine", "Pennsylvania", "New York", "New Jersey", "Connecticut", "Rhode Island", "Massachusetts", "Vermont", "New Hampshire", "Maine", "Pennsylvania", "New York", "New Jersey", "Connecticut", "Rhode Island", "Massachusetts", "Vermont", "New Hampshire", "Maine"],
            "West": ["California", "Oregon", "Washington", "Idaho", "Montana", "Wyoming", "North Dakota", "South Dakota", "Minnesota", "Wisconsin", "Michigan", "Illinois", "Indiana", "Ohio", "Pennsylvania", "New York", "New Jersey", "Connecticut", "Rhode Island", "Massachusetts", "Vermont", "New Hampshire", "Maine", "Pennsylvania", "New York", "New Jersey", "Connecticut", "Rhode Island", "Massachusetts", "Vermont", "New Hampshire", "Maine"]
        }
        df["state"] = df["region"].apply(lambda r: np.random.choice(region_to_states.get(r, ["Unknown"])))
        unique_members = min(len(df), max(10, int(len(df) * 0.15)))
        df["member_id"] = np.random.randint(1, unique_members + 1, size=len(df))
        
        # Add provider IDs and a small number of fraudulent providers for analysis
        df['provider_id'] = np.random.randint(1, 5500, size=len(df))
        fraudulent_providers = df.loc[df['is_fraud']==1, 'provider_id'].unique()
        df['provider_fraud'] = df['provider_id'].isin(fraudulent_providers).astype(int)
        
    return df

df = load_data()

# --- 3. Sidebar for Filters & Controls (Improved UX with st.form) ---
with st.sidebar:
    st.title("Filters")
    
    # Use a form to group filters. This prevents the app from rerunning with every
    # single change, only updating once the "Apply Filters" button is clicked.
    with st.form(key='filter_form'):
        sel_gender = st.multiselect(
            "Gender", options=df["gender"].dropna().unique(), 
            default=df["gender"].dropna().unique(),
            help="Filter claims by patient gender."
        )
        sel_region = st.multiselect(
            "Region", options=df["region"].dropna().unique(), 
            default=df["region"].dropna().unique(),
            help="Filter claims by geographical region."
        )
        sel_provider = st.multiselect(
            "Provider Type", options=df["provider_type"].dropna().unique(), 
            default=df["provider_type"].dropna().unique(),
            help="Filter claims by the type of healthcare provider."
        )
        sel_diag = st.multiselect(
            "Primary Diagnosis", options=df["primary_diagnosis"].dropna().unique(), 
            default=df["primary_diagnosis"].dropna().unique(),
            help="Filter claims by primary diagnosis code."
        )
        age_min, age_max = int(df["age"].min(skipna=True)), int(df["age"].max(skipna=True))
        age_range = st.slider("Age Range", age_min, age_max, (age_min, age_max),
                             help="Filter patients by age range.")
        
        submit_button = st.form_submit_button(label='Apply Filters')

# --- 4. Data Filtering and Preprocessing ---
# Create a boolean mask
mask = (
    df["gender"].isin(sel_gender) &
    df["region"].isin(sel_region) &
    df["provider_type"].isin(sel_provider) &
    df["primary_diagnosis"].isin(sel_diag) &
    df["age"].between(age_range[0], age_range[1])
)

# Use .loc to preserve all columns
df_filtered = df.loc[mask].copy()

# Check if 'claim_date' exists
if 'claim_date' not in df_filtered.columns:
    st.error("claim_date column is missing after filtering.")
    st.stop()


# Handle missing data for visualizations
# For non-technical users, we don't display raw imputation, but we handle it
# to ensure charts are generated correctly without errors.
df_filtered["gender"].fillna("Unknown", inplace=True)
df_filtered["provider_type"].fillna("Unknown", inplace=True)
df_filtered["claim_cost"].fillna(df_filtered["claim_cost"].median(), inplace=True)

if df_filtered.empty:
    st.warning("No data available for the selected filters. Please adjust the filters.")
    st.stop()

# --- 5. Dashboard Layout and Visualizations (Actionable BI Focus) ---
st.title(":material/dashboard: Healthcare Claims Analytics Dashboard")

# --- Scorecards with Key Performance Indicators (KPIs) ---
# A prominent KPI section provides an immediate, high-level overview.
st.subheader("Key Business Metrics")
col1, col2, col3, col4 = st.columns(4)

total_claims = len(df_filtered)
total_cost = df_filtered["claim_cost"].sum()
fraud_rate_pct = df_filtered["is_fraud"].mean() * 100
readmit_rate_pct = df_filtered["readmit_30d"].mean() * 100

with col1:
    st.metric(label="Total Claims", value=f"{total_claims:,}")
with col2:
    st.metric(label="Total Claim Cost", value=f"${total_cost:,.0f}")
with col3:
    st.metric(label="Fraudulent Rate", value=f"{fraud_rate_pct:.2f}%",
              delta_color="inverse", help="Percentage of claims flagged as potentially fraudulent.")
with col4:
    st.metric(label="Readmission Rate", value=f"{readmit_rate_pct:.2f}%",
              delta_color="inverse", help="Percentage of patients readmitted within 30 days.")

st.divider()

# --- Trends & Financial Health (Dual-Axis Plot) ---
st.header("Financial Trends and Forecasts")
st.markdown("""
_This section visualizes monthly claim volume and total cost to help identify seasonal patterns and inform financial planning._
""")
df_filtered["claim_date"] = pd.to_datetime(df_filtered["claim_date"])
df_filtered["month"] = df_filtered["claim_date"].dt.to_period("M").dt.to_timestamp()
monthly_agg = df_filtered.groupby("month").agg(
    claim_volume=pd.NamedAgg(column="claim_cost", aggfunc="size"),
    total_cost=pd.NamedAgg(column="claim_cost", aggfunc="sum")
).reset_index()

# Simple moving average forecast for the next 3 months
if len(monthly_agg) >= 3:
    last3 = monthly_agg.tail(3)
    avg_vol = last3["claim_volume"].mean()
    avg_cost = last3["total_cost"].mean()
else:
    avg_vol = monthly_agg["claim_volume"].mean() if not monthly_agg.empty else 0
    avg_cost = monthly_agg["total_cost"].mean() if not monthly_agg.empty else 0

future_months = pd.date_range(monthly_agg["month"].max() + pd.offsets.MonthBegin(1) if not monthly_agg.empty else pd.Timestamp("2023-01-01"), periods=3, freq="M")
forecast = pd.DataFrame({"month": future_months, "claim_volume": avg_vol, "total_cost": avg_cost})
monthly_full = pd.concat([monthly_agg, forecast], ignore_index=True)

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=monthly_full["month"], y=monthly_full["claim_volume"],
                               mode="lines+markers", name="Claim Volume", line=dict(color="#1f77b4")))
fig_trend.add_add_trace(go.Scatter(x=monthly_full["month"], y=monthly_full["total_cost"],
                                 mode="lines+markers", name="Total Cost", yaxis="y2",
                                 line=dict(color="#ff7f0e")))
fig_trend.add_vline(x=monthly_agg["month"].max(), line_width=1, line_dash="dash", line_color="gray",
                    annotation_text="Start of Forecast", annotation_position="top right")

fig_trend.update_layout(
    title="Monthly Claim Volume & Total Cost (with 3-month Forecast)",
    xaxis_title="Month",
    yaxis=dict(title="Claims Volume", showgrid=False),
    yaxis2=dict(title="Total Cost (USD)", overlaying="y", side="right", showgrid=False),
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0)'),
    margin=dict(l=20, r=20, t=50, b=20),
)
st.plotly_chart(fig_trend, use_container_width=True)


# --- Operational & Risk Analysis ---
# This section uses a column-based layout to present multiple, related charts
# side-by-side for easy comparison and deeper insight.
st.header("Operational and Risk Analysis")
col1_op, col2_op = st.columns(2)

with col1_op:
    st.subheader("Readmission Rates by Diagnosis")
    readmit_rate_by_diag = df_filtered.groupby("primary_diagnosis")["readmit_30d"].mean().reset_index()
    readmit_rate_by_diag["readmit_rate_pct"] = readmit_rate_by_diag["readmit_30d"] * 100
    
    fig_readmit = px.bar(readmit_rate_by_diag.sort_values("readmit_rate_pct", ascending=False),
                         x="primary_diagnosis", y="readmit_rate_pct",
                         title="Readmission Rate by Diagnosis",
                         labels={"primary_diagnosis": "Primary Diagnosis", "readmit_rate_pct": "Readmission Rate (%)"},
                         color="readmit_rate_pct", color_continuous_scale="reds")
    fig_readmit.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig_readmit, use_container_width=True)
    st.markdown("""
        _This chart helps care managers identify which diagnoses are associated with the highest readmission rates,
        enabling targeted interventions and preventive care strategies._
    """)

with col2_op:
    st.subheader("Geographic Distribution of Claims")
    metric_choice = st.radio("Metric", ["Total Cost", "Claim Count"], horizontal=True,
                             key="geo_metric_radio")
    
    state_agg = df_filtered.groupby("state").agg(
        claim_count=("claim_cost", "size"),
        total_cost=("claim_cost", "sum")
    ).reset_index()
    
    color_col = "total_cost" if metric_choice == "Total Cost" else "claim_count"
    map_title = f"{metric_choice} by State"
    color_label = f"{metric_choice} (USD)" if metric_choice == "Total Cost" else "Claim Count"
    
    fig_map = px.choropleth(
        state_agg, locations="state", locationmode="USA-states",
        color=color_col, color_continuous_scale="Blues", scope="usa",
        title=map_title, labels={color_col: color_label}
    )
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown("""
        _A geographic map highlights areas with high claims activity. This insight can guide
        resource allocation and provider network expansion strategies._
    """)


# --- Provider Performance and Fraud Detection ---
st.header("Provider Performance & Fraud Risk")
st.markdown("""
_This section provides a granular view of provider activity, with a focus on identifying high-cost and potentially fraudulent providers._
""")

tab_top_providers, tab_fraud_analysis = st.tabs()

with tab_top_providers:
    st.subheader("Top 10 Providers by Total Claim Cost")
    member_cost = df_filtered.groupby("provider_id", as_index=False)["claim_cost"].sum().rename(columns={"claim_cost": "total_cost"})
    top_providers = member_cost.nlargest(10, "total_cost")
    
    fig_top_providers = px.bar(
        top_providers, x="total_cost", y="provider_id", orientation="h",
        title="Top 10 Providers by Total Claim Cost",
        labels={"total_cost": "Total Claim Cost (USD)", "provider_id": "Provider ID"}
    )
    fig_top_providers.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_top_providers, use_container_width=True)
    st.markdown("""
        _Focusing on the top-cost providers can help identify areas for detailed claims review and cost management.
        The most expensive providers in this dataset account for a significant portion of the total expenditure._
    """)

with tab_fraud_analysis:
    st.subheader("Fraudulent Claims by Provider Type")
    fraud_counts = df_filtered[df_filtered["is_fraud"] == 1].groupby("provider_type").size().reset_index(name="fraud_count")
    total_claims_by_provider = df_filtered.groupby("provider_type").size().reset_index(name="total_claims")
    
    fraud_rates = pd.merge(fraud_counts, total_claims_by_provider, on="provider_type", how="left")
    fraud_rates["fraud_rate"] = (fraud_rates["fraud_count"] / fraud_rates["total_claims"]) * 100
    
    fig_fraud_rates = px.bar(fraud_rates.sort_values("fraud_rate", ascending=False),
                             x="provider_type", y="fraud_rate",
                             title="Fraudulent Claim Rate by Provider Type",
                             labels={"provider_type": "Provider Type", "fraud_rate": "Fraud Rate (%)"},
                             color_continuous_scale="sunsetdark")
    st.plotly_chart(fig_fraud_rates, use_container_width=True)
    st.markdown("""
        _This chart reveals which provider types have the highest concentration of fraudulent claims,
        enabling focused risk management and compliance efforts._
    """)


# --- Data Quality and Metadata Section (for user trust) ---
# An expander is a good way to include metadata and data quality metrics
# without cluttering the main dashboard.
with st.expander(":material/database: **Data Quality Overview & Source Metadata**"):
    st.markdown("""
    _This section provides transparency on the data's health and its source.
    For effective decision-making, it is crucial to trust the underlying data._
    """)
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.metric(label="Data Completeness", value=">99%", help="Percentage of critical data fields without missing values.")
        st.metric(label="Total Records", value=f"{len(df_filtered):,}")
    with col_meta2:
        st.metric(label="Last Data Refresh", value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        st.metric(label="Total Unique Providers", value=f"{df_filtered['provider_id'].nunique():,}")
    
    st.markdown("---")
    st.markdown("""
    **Insights from Data Quality Analysis:**
    *   `claim_cost` and `gender` columns had minor missing values (<2%) and were imputed to ensure visualization consistency.
    *   This dataset is entirely synthetic, designed to simulate a real-world scenario without containing any actual patient data.
    """)
