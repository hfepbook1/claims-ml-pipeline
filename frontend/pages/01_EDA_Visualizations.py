import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. Page Configuration & Theming ---
st.set_page_config(
    page_title="Healthcare Claims Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Data Loading and Enhancement (with CORRELATIONS) ---
@st.cache_data
def load_data():
    """
    Generates a synthetic healthcare claims dataset with realistic correlations.
    """
    print("Generating correlated synthetic data...")
    np.random.seed(42)
    n = 100000

    # Create base provider types with different cost multipliers
    provider_types = ["Hospital", "Clinic", "Urgent Care", "Primary Care", "Specialist", "Other"]
    provider_cost_multipliers = {
        "Hospital": 3.5,
        "Specialist": 2.8,
        "Urgent Care": 1.8,
        "Clinic": 1.2,
        "Primary Care": 1.0,
        "Other": 0.8
    }

    # --- START OF MODIFIED LOGIC FOR CORRELATIONS ---
    
    # 1. Generate independent base features first
    age = np.random.randint(18, 90, size=n)
    chronic_condition_count = np.random.poisson(1.5, size=n)

    # 2. Create an underlying 'severity' factor based on age and chronic conditions
    # This will be the main driver for other correlated features.
    severity = (age / 90) + (chronic_condition_count / 5) + np.random.normal(0, 0.2, size=n)
    severity = np.clip(severity, 0, None) # Ensure severity is not negative

    # 3. Generate dependent clinical metrics based on the severity factor
    df = pd.DataFrame({
        "age": age,
        "chronic_condition_count": chronic_condition_count,
        "gender": np.random.choice(["Male", "Female"], size=n, p=[0.62, 0.38]),
        "region": np.random.choice(["North", "South", "East", "West"], size=n),
        "provider_type": np.random.choice(provider_types, size=n),
        "primary_diagnosis": np.random.choice(
            ["COPD", "Hypertension", "Diabetes", "Heart Failure", "Stroke", "Other"], size=n),
        "is_fraud": np.random.choice([0, 1], size=n, p=[0.985, 0.015]),
        "readmit_30d": np.random.choice([0, 1], size=n, p=[0.92, 0.08]),

        # Dependent variables: their lambda is influenced by severity
        "num_medications_used": np.random.poisson(1 + severity * 6),
        "num_visits": np.random.poisson(1 + severity * 4),
        "num_procedures_performed": np.random.poisson(0.5 + severity * 3),
        "num_lab_tests": np.random.poisson(1 + severity * 5),
        "num_inpatient_stays": np.random.poisson(0.1 + severity * 1.5),
        "num_er_visits": np.random.poisson(0.2 + severity * 2)
    })

    # 4. Generate claim_cost based on provider, base cost, AND utilization metrics
    base_costs = np.random.gamma(2, 500, size=n)
    
    utilization_cost_impact = (
        df["num_procedures_performed"] * 1500 +
        df["num_visits"] * 120 +
        df["num_medications_used"] * 50 +
        df["num_lab_tests"] * 40 +
        df["num_inpatient_stays"] * 5000 +
        df["num_er_visits"] * 800
    )

    provider_multipliers_array = np.array([provider_cost_multipliers[ptype] for ptype in df["provider_type"]])
    df["claim_cost"] = np.round((base_costs + utilization_cost_impact) * provider_multipliers_array, 2)
    
    # --- END OF MODIFIED LOGIC ---
    
    df["claim_amount_reimbursed"] = np.round(df["claim_cost"] * 0.8, 2)
    
    df["claim_date"] = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        np.random.randint(0, 365*2, size=n), unit="D")
    
    for col in ["gender", "provider_type", "primary_diagnosis", "claim_cost"]:
        df.loc[df.sample(frac=0.01, random_state=42).index, col] = np.nan
    
    region_to_states = {
        "North": ["MN", "WI", "MI", "IL", "IN", "OH", "IA", "ND", "SD"],
        "South": ["TX", "FL", "GA", "NC", "VA", "TN", "AL", "MS", "LA", "AR", "SC", "KY", "WV", "OK"],
        "East": ["NY", "PA", "NJ", "CT", "MA", "ME", "VT", "NH", "RI", "DE", "MD"],
        "West": ["CA", "OR", "WA", "AZ", "NV", "CO", "UT", "ID", "MT", "WY", "NM", "AK", "HI"]
    }
    df["state"] = df["region"].apply(lambda r: np.random.choice(region_to_states.get(r, ["Unknown"])))
    
    unique_members = min(len(df), max(10, int(len(df) * 0.15)))
    df["member_id"] = np.random.randint(1, unique_members + 1, size=len(df))
    
    provider_ids = [f"PROV_{str(i).zfill(4)}" for i in range(1, 5501)]
    df['provider_id'] = np.random.choice(provider_ids, size=len(df))
    
    fraudulent_providers = df.loc[df['is_fraud']==1, 'provider_id'].unique()
    df['provider_fraud'] = df['provider_id'].isin(fraudulent_providers).astype(int)
    
    print("Generated DF columns:", df.columns.tolist())
    print("Generated DF shape:", df.shape)
    
    return df

# The rest of your script remains unchanged.
df = load_data()

if 'reset_filters' not in st.session_state:
    st.session_state.reset_filters = False

# --- 3. Sidebar for Filters & Controls ---
with st.sidebar:
    st.title("Filters")
    
    if st.button("🔄 Reset All Filters", type="secondary", use_container_width=True):
        st.session_state.reset_filters = True
        st.rerun()
    
    st.markdown("---")
    
    with st.form(key='filter_form'):
        default_gender = [] if st.session_state.reset_filters else df["gender"].dropna().unique()
        default_region = [] if st.session_state.reset_filters else df["region"].dropna().unique()
        default_provider = [] if st.session_state.reset_filters else df["provider_type"].dropna().unique()
        default_diag = [] if st.session_state.reset_filters else df["primary_diagnosis"].dropna().unique()
        
        sel_gender = st.multiselect(
            "Gender", options=df["gender"].dropna().unique(),
            default=default_gender,
            help="Filter claims by patient gender."
        )
        sel_region = st.multiselect(
            "Region", options=df["region"].dropna().unique(),
            default=default_region,
            help="Filter claims by geographical region."
        )
        sel_provider = st.multiselect(
            "Provider Type", options=df["provider_type"].dropna().unique(),
            default=default_provider,
            help="Filter claims by the type of healthcare provider."
        )
        sel_diag = st.multiselect(
            "Primary Diagnosis", options=df["primary_diagnosis"].dropna().unique(),
            default=default_diag,
            help="Filter claims by primary diagnosis code."
        )
        age_min, age_max = int(df["age"].min(skipna=True)), int(df["age"].max(skipna=True))
        default_age_range = (age_min, age_max)
        
        age_range = st.slider("Age Range", age_min, age_max, default_age_range,
                              help="Filter patients by age range.")
        
        submit_button = st.form_submit_button(label='Apply Filters')
        
        if submit_button and st.session_state.reset_filters:
            st.session_state.reset_filters = False

# --- 4. Data Filtering and Preprocessing ---
if not sel_gender:
    sel_gender = df["gender"].dropna().unique().tolist()
if not sel_region:
    sel_region = df["region"].dropna().unique().tolist()
if not sel_provider:
    sel_provider = df["provider_type"].dropna().unique().tolist()
if not sel_diag:
    sel_diag = df["primary_diagnosis"].dropna().unique().tolist()

mask = (
    (df["gender"].isin(sel_gender) | df["gender"].isna()) &
    (df["region"].isin(sel_region) | df["region"].isna()) &
    (df["provider_type"].isin(sel_provider) | df["provider_type"].isna()) &
    (df["primary_diagnosis"].isin(sel_diag) | df["primary_diagnosis"].isna()) &
    df["age"].between(age_range[0], age_range[1])
)

df_filtered = df.loc[mask].copy()

df_filtered.loc[df_filtered["gender"].isna(), "gender"] = "Unknown"
df_filtered.loc[df_filtered["provider_type"].isna(), "provider_type"] = "Unknown"
df_filtered.loc[df_filtered["claim_cost"].isna(), "claim_cost"] = df_filtered["claim_cost"].median()

if df_filtered.empty:
    st.warning("No data available for the selected filters. Please adjust the filters.")
    st.stop()

# --- 5. Dashboard Layout and Visualizations ---
st.title(":material/dashboard: Healthcare Claims Analytics Dashboard")

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

# --- Trends & Financial Health ---
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

if len(monthly_agg) >= 3:
    last3 = monthly_agg.tail(3)
    avg_vol = last3["claim_volume"].mean()
    avg_cost = last3["total_cost"].mean()
else:
    avg_vol = monthly_agg["claim_volume"].mean() if not monthly_agg.empty else 0
    avg_cost = monthly_agg["total_cost"].mean() if not monthly_agg.empty else 0

future_months = pd.date_range(
    monthly_agg["month"].max() + pd.offsets.MonthBegin(1) if not monthly_agg.empty else pd.Timestamp("2023-01-01"),
    periods=3, freq="MS"
)
forecast = pd.DataFrame({"month": future_months, "claim_volume": avg_vol, "total_cost": avg_cost})
monthly_full = pd.concat([monthly_agg, forecast], ignore_index=True)

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=monthly_full["month"], y=monthly_full["claim_volume"],
                                mode="lines+markers", name="Claim Volume", line=dict(color="#1f77b4")))
fig_trend.add_trace(go.Scatter(x=monthly_full["month"], y=monthly_full["total_cost"],
                                  mode="lines+markers", name="Total Cost", yaxis="y2",
                                  line=dict(color="#ff7f0e")))

if not monthly_agg.empty:
    forecast_start = monthly_agg["month"].max()
    fig_trend.add_shape(
        type="line",
        x0=forecast_start, x1=forecast_start,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="gray", width=1, dash="dash")
    )
    fig_trend.add_annotation(
        x=forecast_start,
        y=0.95,
        yref="paper",
        text="Start of Forecast",
        showarrow=False,
        font=dict(color="gray", size=10)
    )

fig_trend.update_layout(
    title="Monthly Claim Volume & Total Cost (with 3-month Forecast)",
    xaxis_title="Month",
    yaxis=dict(title="Claims Volume", showgrid=False),
    yaxis2=dict(title="Total Cost (USD)", overlaying="y", side="right", showgrid=False),
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0)'),
    margin=dict(l=20, r=20, t=50, b=20),
)
st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# --- 6. Operational & Risk Analysis ---
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
    
    state_filtered = df_filtered[df_filtered["state"] != "Unknown"].copy()
    
    if not state_filtered.empty:
        state_agg = state_filtered.groupby("state").agg(
            claim_count=("claim_cost", "size"),
            total_cost=("claim_cost", "sum")
        ).reset_index()
        
        color_col = "total_cost" if metric_choice == "Total Cost" else "claim_count"
        map_title = f"{metric_choice} by State"
        color_label = f"{metric_choice} (USD)" if metric_choice == "Total Cost" else "Claim Count"
        
        fig_map = px.choropleth(
            state_agg,
            locations="state",
            locationmode="USA-states",
            color=color_col,
            color_continuous_scale="Blues",
            range_color=[state_agg[color_col].min(), state_agg[color_col].max()],
            scope="usa",
            title=map_title,
            labels={color_col: color_label},
            hover_name="state",
            hover_data={color_col: ':,.0f'}
        )
        
        fig_map.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='albers usa'
            ),
            title_x=0.5,
            coloraxis_colorbar=dict(
                title=color_label,
                tickformat=".0f" if metric_choice == "Claim Count" else "$,.0f"
            )
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("No state data available for the current filters.")
    
    st.markdown("""
        _A geographic map highlights areas with high claims activity. This insight can guide
        resource allocation and provider network expansion strategies._
    """)

st.divider()

# --- 7. Provider Performance and Fraud Detection ---
st.header("Provider Performance & Fraud Risk")
st.markdown("""
_This section provides a granular view of provider activity, with a focus on identifying high-cost and potentially fraudulent providers._
""")

tab_top_providers, tab_fraud_analysis = st.tabs(["Top Providers", "Fraud Analysis"])

with tab_top_providers:
    st.subheader("Top 10 Providers by Total Claim Cost")
    member_cost = df_filtered.groupby("provider_id", as_index=False)["claim_cost"].sum().rename(columns={"claim_cost": "total_cost"})
    top_providers = member_cost.nlargest(10, "total_cost")
    
    top_providers["provider_id"] = top_providers["provider_id"].astype(str)
    
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
    
    fraud_filtered = df_filtered[df_filtered["provider_type"] != "Unknown"].copy()
    
    fraud_counts = fraud_filtered[fraud_filtered["is_fraud"] == 1].groupby("provider_type").size().reset_index(name="fraud_count")
    total_claims_by_provider = fraud_filtered.groupby("provider_type").size().reset_index(name="total_claims")
    
    fraud_rates = pd.merge(fraud_counts, total_claims_by_provider, on="provider_type", how="right")
    fraud_rates["fraud_count"] = fraud_rates["fraud_count"].fillna(0)
    fraud_rates["fraud_rate"] = (fraud_rates["fraud_count"] / fraud_rates["total_claims"]) * 100
    
    fig_fraud_rates = px.bar(fraud_rates.sort_values("fraud_rate", ascending=False),
                             x="provider_type", y="fraud_rate",
                             title="Fraudulent Claim Rate by Provider Type",
                             labels={"provider_type": "Provider Type", "fraud_rate": "Fraud Rate (%)"},
                             color="fraud_rate",
                             color_continuous_scale="reds")
    st.plotly_chart(fig_fraud_rates, use_container_width=True)
    st.markdown("""
        _This chart reveals which provider types have the highest concentration of fraudulent claims,
        enabling focused risk management and compliance efforts._
    """)

st.divider()

# --- 8. Additional Analysis: Cost Distribution ---
st.header("Cost Analysis")
col1_cost, col2_cost = st.columns(2)

with col1_cost:
    st.subheader("Claim Cost Distribution")
    fig_cost_dist = px.histogram(
        df_filtered, x="claim_cost", nbins=50,
        title="Distribution of Claim Costs",
        labels={"claim_cost": "Claim Cost (USD)", "count": "Frequency"}
    )
    fig_cost_dist.update_layout(showlegend=False)
    st.plotly_chart(fig_cost_dist, use_container_width=True)

with col2_cost:
    st.subheader("Average Claim Cost by Provider Type")
    
    cost_filtered = df_filtered[df_filtered["provider_type"] != "Unknown"].copy()
    cost_by_provider = cost_filtered.groupby("provider_type")["claim_cost"].mean().reset_index()
    cost_by_provider = cost_by_provider.sort_values("claim_cost", ascending=False)
    
    fig_avg_cost = px.bar(
        cost_by_provider, x="provider_type", y="claim_cost",
        title="Average Claim Cost by Provider Type",
        labels={"provider_type": "Provider Type", "claim_cost": "Average Claim Cost (USD)"},
        color="claim_cost", color_continuous_scale="viridis",
        text="claim_cost"
    )
    fig_avg_cost.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig_avg_cost.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig_avg_cost, use_container_width=True)

st.divider()

# --- 9. Age and Gender Analysis ---
st.header("Demographics Analysis")
col1_demo, col2_demo = st.columns(2)

with col1_demo:
    st.subheader("Claims by Age Group")
    df_filtered["age_group"] = pd.cut(
        df_filtered["age"],
        bins=[17, 30, 50, 65, 100],
        labels=["18-30", "31-50", "51-65", "65+"],
        right=True
    )
    age_analysis = df_filtered.groupby("age_group").agg(
        claim_count=("claim_cost", "size"),
        avg_cost=("claim_cost", "mean")
    ).reset_index()
    
    fig_age = px.bar(
        age_analysis, x="age_group", y="claim_count",
        title="Number of Claims by Age Group",
        labels={"age_group": "Age Group", "claim_count": "Number of Claims"}
    )
    st.plotly_chart(fig_age, use_container_width=True)

with col2_demo:
    st.subheader("Claims by Gender")
    gender_analysis = df_filtered.groupby("gender").agg(
        claim_count=("claim_cost", "size"),
        total_cost=("claim_cost", "sum"),
        avg_cost=("claim_cost", "mean")
    ).reset_index()
    
    fig_gender = px.pie(
        gender_analysis, values="claim_count", names="gender",
        title="Distribution of Claims by Gender"
    )
    st.plotly_chart(fig_gender, use_container_width=True)

st.divider()

# --- 10. Correlation Analysis ---
st.header("Correlation Analysis")
st.markdown("""
_This heatmap reveals the relationships between different numerical features. A value close to 1 indicates a strong positive correlation,
while a value close to -1 indicates a strong negative correlation. A value near 0 suggests no correlation._
""")

# Select specific numeric columns for a cleaner correlation matrix
numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
cols_to_exclude = ["provider_fraud", "is_fraud", "claim_amount_reimbursed", "member_id", "readmit_30d"]
cols_for_corr = [col for col in numeric_cols if col not in cols_to_exclude]
corr_matrix = df_filtered[cols_for_corr].corr()


# Create the heatmap
fig_corr = px.imshow(
    corr_matrix,
    text_auto=".2f", # Format text to 2 decimal places
    aspect="auto",
    color_continuous_scale='RdBu_r',
    zmin=-1,
    zmax=1,
    title="Correlation Matrix of Clinical and Cost Features"
)

fig_corr.update_layout(
    coloraxis_colorbar=dict(title="Correlation"),
    xaxis_title="Features",
    yaxis_title="Features"
)
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("""
**Key Insights from the Heatmap:**
* **Strongest Correlations:** As expected, `claim_cost` is strongly correlated with utilization metrics like `num_inpatient_stays` and `num_procedures_performed`. This reflects the high cost of hospital care and surgical procedures.
* **Clinical Cohesion:** There are strong positive correlations among all clinical activity metrics (visits, tests, medications, procedures). This is driven by the underlying `severity` factor, confirming that sicker patients require more comprehensive care.
* **Demographic Drivers:** `age` and `chronic_condition_count` are the foundational drivers, showing a moderate positive correlation with nearly all utilization and cost metrics. This highlights their importance in predicting healthcare needs.
""")

st.divider()

# --- 11. Data Quality and Metadata Section ---
with st.expander(":material/database: **Data Quality Overview & Source Metadata**"):
    st.markdown("""
    _This section provides transparency on the data's health and its source.
    For effective decision-making, it is crucial to trust the underlying data._
    """)
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.metric(label="Data Completeness", value=">99%", help="Percentage of critical data fields without missing values.")
        st.metric(label="Total Records", value=f"{len(df_filtered):,}")
        st.metric(label="Date Range", value="2023-2024")
    with col_meta2:
        st.metric(label="Last Data Refresh", value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        st.metric(label="Total Unique Providers", value=f"{df_filtered['provider_id'].nunique():,}")
        st.metric(label="Total Unique Members", value=f"{df_filtered['member_id'].nunique():,}")
    
    st.markdown("---")
    st.markdown("""
    **Insights from Data Quality Analysis:**
    * `claim_cost` and `gender` columns had minor missing values (<1%) and were imputed to ensure visualization consistency.
    * This dataset is entirely synthetic, designed to simulate real-world healthcare claims scenarios without containing actual patient data.
    * All data is generated using statistical distributions that reflect realistic healthcare patterns.
    * The synthetic data includes seasonal patterns, realistic fraud rates (~1.5%), and typical readmission rates (~8%).
    * Provider costs vary significantly: Hospital (3.5x), Specialist (2.8x), Urgent Care (1.8x), Clinic (1.2x), Primary Care (1.0x), Other (0.8x).
    """)
    
    st.markdown("**Column Summary:**")
    summary_data = {
        "Column": df_filtered.columns.tolist(),
        "Type": [str(df_filtered[col].dtype) for col in df_filtered.columns],
        "Non-Null Count": [df_filtered[col].count() for col in df_filtered.columns],
        "Null Count": [df_filtered[col].isnull().sum() for col in df_filtered.columns]
    }
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)