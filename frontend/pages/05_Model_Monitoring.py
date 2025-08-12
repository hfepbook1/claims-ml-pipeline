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
            "Use the date range filter to zoom into specific periods. Alerts will highlight any significant anomalies.")

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
        # Create an SQLAlchemy engine from the DBAPI2 connection
        engine = create_engine("databricks://", creator=lambda: conn)
        
        # NOTE: The LIMIT clause is added for safety during debugging.
        # For production, you may want to remove it or filter more specifically (e.g., a WHERE clause).
        query = "SELECT * FROM workspace.claims_project.monitoring_metrics ORDER BY event_ts LIMIT 1000"
        
        df = pd.read_sql(query, engine)
        df['event_ts'] = pd.to_datetime(df['event_ts'])
        return df
    finally:
        # This ensures the connection is closed even if an error occurs
        if conn:
            conn.close()

# Load data with a loading spinner
with st.spinner("Loading latest metrics from Databricks..."):
    df_metrics = load_metrics_data()

# --- DEBUGGING SECTION ---
# This section helps you find the correct column names from your Databricks table.
st.subheader("🕵️ Data Inspector")
st.write("The first 5 rows of data returned from Databricks:")
st.dataframe(df_metrics.head())
st.write("Discovered column names are:", df_metrics.columns.tolist())
st.info("If the app shows a `KeyError`, check the column names above. You'll need to update the `columns='metric'` part of the pivot function below with the correct column name from the list.")
# --- END DEBUGGING SECTION ---

if df_metrics.empty:
    st.error("No monitoring data available.")
    st.stop()

# Pivot the data so each metric is a column
# !!! IMPORTANT: Check the column name 'metric' below and correct it if needed!
df_wide = df_metrics.pivot(index='event_ts', columns='metric', values='value').sort_index()

# Convert accuracy/precision/recall metrics from fraction to percentage
for col in df_wide.columns:
    if col in ["Fraud Acc", "Fraud Recall", "Readmit Acc", "Readmit Precision"]:
        df_wide[col] = df_wide[col] * 100

# Date range selection
min_date = df_wide.index.min().to_pydatetime()
max_date = df_wide.index.max().to_pydatetime()
date_range = st.slider("Select Date Range",
                       min_value=min_date, max_value=max_date,
                       value=(min_date, max_date))
start_date, end_date = date_range[0], date_range[1]
# Filter data to the selected date range
df_filtered = df_wide.loc[(df_wide.index >= pd.to_datetime(start_date)) &
                          (df_wide.index <= pd.to_datetime(end_date))]

if df_filtered.empty:
    st.warning("No data in the selected date range. Please adjust the range.")
    st.stop()

# Calculate RMSE rolling mean and control bounds (±2σ) on full data
if "RMSE" in df_wide.columns:
    rmse_series = df_wide["RMSE"]
    rmse_mean = rmse_series.rolling(window=7).mean()
    rmse_std = rmse_series.rolling(window=7).std()
    upper_band = rmse_mean + 2 * rmse_std
    lower_band = rmse_mean - 2 * rmse_std
else:
    upper_band = pd.Series(dtype=float)
    lower_band = pd.Series(dtype=float)

# Anomaly detection alerts for latest metrics (full range)
latest_date = df_wide.index.max()
if "RMSE" in df_wide.columns:
    latest_rmse = df_wide["RMSE"].loc[latest_date]
    if latest_date in upper_band.index and pd.notnull(upper_band.loc[latest_date]) and latest_rmse > upper_band.loc[latest_date]:
        st.error(f"🚨 RMSE spike detected on {latest_date.date()}: RMSE={latest_rmse:.2f} (above normal range)")
if "Fraud Acc" in df_wide.columns:
    latest_fraud_acc = df_wide["Fraud Acc"].loc[latest_date]
    if latest_fraud_acc < 80:
        st.error(f"🚨 Fraud Model Accuracy dropped to {latest_fraud_acc:.1f}% on {latest_date.date()} (below 80%)")
if "Readmit Acc" in df_wide.columns:
    latest_readmit_acc = df_wide["Readmit Acc"].loc[latest_date]
    if latest_readmit_acc < 80:
        st.error(f"🚨 Readmission Model Accuracy dropped to {latest_readmit_acc:.1f}% on {latest_date.date()} (below 80%)")

# Use the filtered data for plotting
df_plot = df_filtered

# Plot RMSE over time with control band
if "RMSE" in df_plot.columns:
    rmse_plot = df_plot["RMSE"]
    # Slice the pre-computed control bands to the selected range
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
                           xaxis_title="Date", yaxis_title="RMSE")
    st.plotly_chart(fig_rmse, use_container_width=True)
    st.caption("**Note:** The shaded red area indicates the expected RMSE range (±2σ) based on a rolling 7-day window.")
else:
    st.info("RMSE metric not available in the data.")

# Plot Fraud Detection model performance (Accuracy and Recall)
if "Fraud Acc" in df_plot.columns:
    fig_fraud = go.Figure()
    fig_fraud.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Fraud Acc"],
                                   mode='lines+markers', name='Accuracy'))
    if "Fraud Recall" in df_plot.columns:
        fig_fraud.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Fraud Recall"],
                                       mode='lines+markers', name='Recall'))
    fig_fraud.update_layout(title="Fraud Detection Model - Accuracy/Recall Over Time",
                            xaxis_title="Date", yaxis_title="Performance (%)")
    st.plotly_chart(fig_fraud, use_container_width=True)
else:
    st.info("Fraud Detection metrics not available.")

# Plot Readmission Prediction model performance (Accuracy and Precision)
if "Readmit Acc" in df_plot.columns:
    fig_readmit = go.Figure()
    fig_readmit.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Readmit Acc"],
                                     mode='lines+markers', name='Accuracy'))
    if "Readmit Precision" in df_plot.columns:
        fig_readmit.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Readmit Precision"],
                                         mode='lines+markers', name='Precision'))
    fig_readmit.update_layout(title="Readmission Prediction Model - Accuracy/Precision Over Time",
                              xaxis_title="Date", yaxis_title="Performance (%)")
    st.plotly_chart(fig_readmit, use_container_width=True)
else:
    st.info("Readmission Prediction metrics not available.")