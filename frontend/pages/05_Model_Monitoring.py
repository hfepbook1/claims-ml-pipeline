import pandas as pd
from databricks import sql
from sqlalchemy import create_engine
import streamlit as st
import plotly.graph_objects as go

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Model Monitoring Dashboard",
    layout="wide"
)

# ==============================================================================
# 2. DATA LOADING
# ==============================================================================
# This function connects to Databricks and fetches model performance metrics.
# It's cached to prevent re-running on every interaction, refreshing every 24 hours.
@st.cache_data(ttl=24*3600, show_spinner="Fetching latest data from Databricks...")
def load_metrics_data():
    """
    Connects to Databricks, fetches model monitoring metrics,
    and returns them in a pandas DataFrame.
    """
    conn = None  # Initialize conn to None
    try:
        conn = sql.connect(
            server_hostname=st.secrets["DATABRICKS_HOST"],
            http_path=st.secrets["DATABRICKS_HTTP_PATH"],
            access_token=st.secrets["DATABRICKS_TOKEN"]
        )
        engine = create_engine("databricks://", creator=lambda: conn)
        query = "SELECT day, event_ts, rmse_cost, accuracy_fraud, accuracy_readmit FROM workspace.claims_project.monitoring_metrics ORDER BY day"
        df = pd.read_sql(query, engine)
        df['event_ts'] = pd.to_datetime(df['event_ts'])
        return df
    finally:
        if conn:
            conn.close()

# Load the data
df_metrics = load_metrics_data()

# ==============================================================================
# 3. SIDEBAR AND FILTERS
# ==============================================================================
with st.sidebar:
    st.title("🤖 Model Monitoring Dashboard")
    st.markdown("""
    This page provides real-time performance tracking for all models. 
    Metrics are fetched daily from Databricks.
    """)
    
    st.header("Filters")
    st.markdown("Use the slider to zoom into a specific period.")

    if df_metrics.empty:
        st.error("No monitoring data available to set filters.")
        st.stop()

    # Prepare data for filtering
    df_by_day = df_metrics.set_index('day').sort_index()
    all_days = df_by_day.index.unique().tolist()

    # Day range selection slider
    day_range = st.select_slider(
        "Select Day Range",
        options=all_days,
        value=(all_days[0], all_days[-1])
    )
    start_day, end_day = day_range

# ==============================================================================
# 4. DATA PROCESSING AND ANALYSIS
# ==============================================================================
if df_metrics.empty:
    st.error("No monitoring data available.")
    st.stop()

# Convert accuracy to percentage
for col in ["accuracy_fraud", "accuracy_readmit"]:
    if col in df_by_day.columns:
        df_by_day[col] = df_by_day[col] * 100

# Filter data based on sidebar selection
df_filtered = df_by_day.loc[start_day:end_day]

if df_filtered.empty:
    st.warning("No data in the selected day range. Please adjust the range.")
    st.stop()

# Calculate RMSE rolling mean and control bounds (±2σ) on the full dataset
if "rmse_cost" in df_by_day.columns:
    rmse_series = df_by_day["rmse_cost"]
    rmse_mean = rmse_series.rolling(window=7).mean()
    rmse_std = rmse_series.rolling(window=7).std()
    upper_band = rmse_mean + 2 * rmse_std
    lower_band = rmse_mean - 2 * rmse_std
else:
    upper_band = pd.Series(dtype=float)
    lower_band = pd.Series(dtype=float)

# ==============================================================================
# 5. MAIN PAGE DISPLAY
# ==============================================================================
st.title("📈 Model Performance Overview")

# --- ANOMALY ALERTS ---
st.subheader("🚨 Latest Day Alerts")
latest_day = df_by_day.index.max()
previous_day = df_by_day.index.unique()[-2] if len(df_by_day.index.unique()) > 1 else latest_day

# Create columns for alerts for a cleaner layout
alert_cols = st.columns(3)
with alert_cols[0]:
    if "rmse_cost" in df_by_day.columns:
        latest_rmse = df_by_day.loc[latest_day, "rmse_cost"]
        if latest_day in upper_band.index and pd.notnull(upper_band.loc[latest_day]) and latest_rmse > upper_band.loc[latest_day]:
            st.error(f"RMSE Spike: {latest_rmse:.2f}")

with alert_cols[1]:
    if "accuracy_fraud" in df_by_day.columns:
        latest_fraud_acc = df_by_day.loc[latest_day, "accuracy_fraud"]
        if latest_fraud_acc < 80:
            st.error(f"Fraud Accuracy Drop: {latest_fraud_acc:.1f}%")

with alert_cols[2]:
    if "accuracy_readmit" in df_by_day.columns:
        latest_readmit_acc = df_by_day.loc[latest_day, "accuracy_readmit"]
        if latest_readmit_acc < 80:
            st.error(f"Readmit Accuracy Drop: {latest_readmit_acc:.1f}%")

st.markdown("---")

# --- KPI METRICS ---
st.subheader("📊 Key Performance Indicators (Latest Day)")
kpi_cols = st.columns(3)

with kpi_cols[0]:
    if "rmse_cost" in df_by_day.columns:
        latest_rmse = df_by_day.loc[latest_day, "rmse_cost"]
        prev_rmse = df_by_day.loc[previous_day, "rmse_cost"]
        delta_rmse = latest_rmse - prev_rmse
        st.metric(
            label="Claims Cost RMSE",
            value=f"{latest_rmse:.2f}",
            delta=f"{delta_rmse:.2f}",
            delta_color="inverse",
            help="Root Mean Squared Error. Lower is better."
        )

with kpi_cols[1]:
    if "accuracy_fraud" in df_by_day.columns:
        latest_fraud_acc = df_by_day.loc[latest_day, "accuracy_fraud"]
        prev_fraud_acc = df_by_day.loc[previous_day, "accuracy_fraud"]
        delta_fraud = latest_fraud_acc - prev_fraud_acc
        st.metric(
            label="Fraud Detection Accuracy",
            value=f"{latest_fraud_acc:.1f}%",
            delta=f"{delta_fraud:.1f}%",
            help="Model accuracy. Higher is better."
        )

with kpi_cols[2]:
    if "accuracy_readmit" in df_by_day.columns:
        latest_readmit_acc = df_by_day.loc[latest_day, "accuracy_readmit"]
        prev_readmit_acc = df_by_day.loc[previous_day, "accuracy_readmit"]
        delta_readmit = latest_readmit_acc - prev_readmit_acc
        st.metric(
            label="Readmission Prediction Accuracy",
            value=f"{latest_readmit_acc:.1f}%",
            delta=f"{delta_readmit:.1f}%",
            help="Model accuracy. Higher is better."
        )

st.markdown("---")

# --- CHARTS IN TABS ---
st.subheader("Performance Over Time")
tab1, tab2, tab3 = st.tabs(["Cost Model (RMSE)", "Fraud Model (Accuracy)", "Readmission Model (Accuracy)"])

with tab1:
    if "rmse_cost" in df_filtered.columns:
        rmse_plot = df_filtered["rmse_cost"]
        rmse_upper_plot = upper_band.reindex(df_filtered.index)
        rmse_lower_plot = lower_band.reindex(df_filtered.index)
        
        fig_rmse = go.Figure()
        # Control bands (plotted first to be in the background)
        fig_rmse.add_trace(go.Scatter(
            x=rmse_upper_plot.index, y=rmse_upper_plot,
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False, name='Upper Control'
        ))
        fig_rmse.add_trace(go.Scatter(
            x=rmse_lower_plot.index, y=rmse_lower_plot,
            fill='tonexty', fillcolor='rgba(255, 82, 82, 0.15)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False, name='Lower Control'
        ))
        # Main RMSE line
        fig_rmse.add_trace(go.Scatter(
            x=rmse_plot.index, y=rmse_plot,
            mode='lines+markers', name='RMSE', line=dict(color='#007bff')
        ))
        
        fig_rmse.update_layout(
            title="Claims Cost Model - RMSE Over Time",
            xaxis_title="Day", yaxis_title="RMSE",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_rmse, use_container_width=True)
        st.caption("The shaded red area indicates the expected RMSE range (±2 standard deviations) based on a 7-day rolling window. Points above this band may indicate performance degradation.")
    else:
        st.info("RMSE metric not available in the data.")

with tab2:
    if "accuracy_fraud" in df_filtered.columns:
        fig_fraud = go.Figure()
        fig_fraud.add_trace(go.Scatter(
            x=df_filtered.index, y=df_filtered["accuracy_fraud"],
            mode='lines+markers', name='Accuracy', line=dict(color='#28a745')
        ))
        # Add threshold line
        fig_fraud.add_hline(y=80, line_dash="dash", line_color="red",
                            annotation_text="80% Threshold", annotation_position="bottom right")
        fig_fraud.update_layout(
            title="Fraud Detection Model - Accuracy Over Time",
            xaxis_title="Day", yaxis_title="Accuracy (%)",
            yaxis_range=[min(60, df_filtered["accuracy_fraud"].min() - 5), 101]
        )
        st.plotly_chart(fig_fraud, use_container_width=True)
    else:
        st.info("Fraud Detection metrics not available.")

with tab3:
    if "accuracy_readmit" in df_filtered.columns:
        fig_readmit = go.Figure()
        fig_readmit.add_trace(go.Scatter(
            x=df_filtered.index, y=df_filtered["accuracy_readmit"],
            mode='lines+markers', name='Accuracy', line=dict(color='#ffc107')
        ))
        # Add threshold line
        fig_readmit.add_hline(y=80, line_dash="dash", line_color="red",
                              annotation_text="80% Threshold", annotation_position="bottom right")
        fig_readmit.update_layout(
            title="Readmission Prediction Model - Accuracy Over Time",
            xaxis_title="Day", yaxis_title="Accuracy (%)",
            yaxis_range=[min(60, df_filtered["accuracy_readmit"].min() - 5), 101]
        )
        st.plotly_chart(fig_readmit, use_container_width=True)
    else:
        st.info("Readmission Prediction metrics not available.")

# ==============================================================================
# 6. FURTHER ANALYSIS AND NEXT STEPS
# ==============================================================================
with st.expander("🔬 Further Analysis & Next Steps"):
    st.markdown("""
    When an alert is triggered, consider the following analytical steps:

    **1. View Raw Data Statistics:**
    The table below shows the descriptive statistics for the metrics in your selected time range. Use this for a quick quantitative assessment.
    """)
    st.dataframe(df_filtered.describe(), use_container_width=True)
    
    st.markdown("""
    **2. Monitor Input Feature Drift:**
    - A drop in model performance (like RMSE spikes or accuracy drops) is often caused by **data drift**, where the distribution of incoming data changes from what the model was trained on.
    - **Action:** Implement monitoring for key input features. Track metrics like the **Population Stability Index (PSI)** or use statistical tests (e.g., Kolmogorov-Smirnov test) to compare the distribution of recent data against a baseline (e.g., the training data).

    **3. Perform Root Cause Analysis:**
    - **Action:** When a metric drops, segment the data to identify the cause. For example:
        - Is the poor performance concentrated in a specific demographic, region, or claim type?
        - Correlate the misclassified records (e.g., false negatives in fraud detection) with specific feature values to find patterns.
    """)