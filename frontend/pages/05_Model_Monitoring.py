import pandas as pd
from databricks import sql
from sqlalchemy import create_engine
import streamlit as st
import plotly.graph_objects as go

# Set up the page
st.set_page_config(page_title="Model Monitoring Dashboard", layout="wide")

st.title("Model Monitoring Dashboard")
st.markdown("This page provides real-time performance tracking for all models. "
            "Metrics are fetched daily from Databricks and visualized below. "
            "Use the day range filter to zoom into specific periods.")

# Data loading function (with caching) to fetch metrics from Databricks
@st.cache_data(ttl=24*3600, show_spinner=False)
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
        # Optimized query to select only necessary columns
        query = "SELECT day, event_ts, rmse_cost, accuracy_fraud, accuracy_readmit FROM workspace.claims_project.monitoring_metrics ORDER BY day"
        df = pd.read_sql(query, engine)
        df['event_ts'] = pd.to_datetime(df['event_ts'])
        return df
    finally:
        if conn:
            conn.close()

# Load data with a loading spinner
with st.spinner("Loading latest metrics from Databricks..."):
    df_metrics = load_metrics_data()

if df_metrics.empty:
    st.error("No monitoring data available.")
    st.stop()

# Set 'day' as the index for plotting and analysis
df_by_day = df_metrics.set_index('day').sort_index()

# Convert accuracy metrics from fraction to percentage
for col in ["accuracy_fraud", "accuracy_readmit"]:
    if col in df_by_day.columns:
        df_by_day[col] = df_by_day[col] * 100

# Day range selection using a select_slider for categorical/integer axis
all_days = df_by_day.index.unique().tolist()
day_range = st.select_slider("Select Day Range",
                             options=all_days,
                             value=(all_days[0], all_days[-1]))

start_day, end_day = day_range
# Filter data to the selected day range
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

# Anomaly detection alerts for the latest day's metrics
latest_day = df_by_day.index.max()
if "rmse_cost" in df_by_day.columns:
    latest_rmse = df_by_day["rmse_cost"].loc[latest_day]
    if latest_day in upper_band.index and pd.notnull(upper_band.loc[latest_day]) and latest_rmse > upper_band.loc[latest_day]:
        st.error(f"🚨 RMSE spike detected on Day {latest_day}: RMSE={latest_rmse:.2f} (above normal range)")
if "accuracy_fraud" in df_by_day.columns:
    latest_fraud_acc = df_by_day["accuracy_fraud"].loc[latest_day]
    if latest_fraud_acc < 80:
        st.error(f"🚨 Fraud Model Accuracy dropped to {latest_fraud_acc:.1f}% on Day {latest_day} (below 80%)")
if "accuracy_readmit" in df_by_day.columns:
    latest_readmit_acc = df_by_day["accuracy_readmit"].loc[latest_day]
    if latest_readmit_acc < 80:
        st.error(f"🚨 Readmission Model Accuracy dropped to {latest_readmit_acc:.1f}% on Day {latest_day} (below 80%)")

# Use the filtered data for plotting
df_plot = df_filtered

# Plot RMSE over time with control band
if "rmse_cost" in df_plot.columns:
    rmse_plot = df_plot["rmse_cost"]
    rmse_upper_plot = upper_band.reindex(df_plot.index)
    rmse_lower_plot = lower_band.reindex(df_plot.index)
    
    fig_rmse = go.Figure()
    fig_rmse.add_trace(go.Scatter(x=rmse_plot.index, y=rmse_plot,
                                  mode='lines+markers', name='RMSE'))
    if not rmse_upper_plot.isnull().all():
        fig_rmse.add_trace(go.Scatter(x=rmse_plot.index, y=rmse_upper_plot,
                                      line=dict(color='rgba(255,255,255,0)'),
                                      showlegend=False, name='Upper Control'))
        fig_rmse.add_trace(go.Scatter(x=rmse_plot.index, y=rmse_lower_plot,
                                      fill='tonexty', fillcolor='rgba(255,0,0,0.1)',
                                      line=dict(color='rgba(255,255,255,0)'),
                                      showlegend=False, name='Lower Control'))
    fig_rmse.update_layout(title="Claims Cost Model - RMSE Over Time",
                           xaxis_title="Day", yaxis_title="RMSE")
    st.plotly_chart(fig_rmse, use_container_width=True)
    st.caption("**Note:** The shaded red area indicates the expected RMSE range (±2σ) based on a rolling 7-day window.")
else:
    st.info("RMSE metric not available in the data.")

# Plot Fraud Detection model performance
if "accuracy_fraud" in df_plot.columns:
    fig_fraud = go.Figure()
    fig_fraud.add_trace(go.Scatter(x=df_plot.index, y=df_plot["accuracy_fraud"],
                                   mode='lines+markers', name='Accuracy'))
    fig_fraud.update_layout(title="Fraud Detection Model - Accuracy Over Time",
                            xaxis_title="Day", yaxis_title="Accuracy (%)")
    st.plotly_chart(fig_fraud, use_container_width=True)
else:
    st.info("Fraud Detection metrics not available.")

# Plot Readmission Prediction model performance
if "accuracy_readmit" in df_plot.columns:
    fig_readmit = go.Figure()
    fig_readmit.add_trace(go.Scatter(x=df_plot.index, y=df_plot["accuracy_readmit"],
                                     mode='lines+markers', name='Accuracy'))
    fig_readmit.update_layout(title="Readmission Prediction Model - Accuracy Over Time",
                              xaxis_title="Day", yaxis_title="Accuracy (%)")
    st.plotly_chart(fig_readmit, use_container_width=True)
else:
    st.info("Readmission Prediction metrics not available.")