import pandas as pd
from databricks import sql
from sqlalchemy import create_engine
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

# --- Page Configuration ---
# Set the page to a wide layout with a professional title and icon.
st.set_page_config(
    page_title="Enterprise Model Monitoring Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (CSS) ---
# Inject custom CSS for a more polished and professional look and feel.
def load_css():
    """Injects custom CSS to improve the dashboard's aesthetics."""
    st.markdown("""
        <style>
        /* Main page background */
       .stApp {
            background-color: #f0f2f6;
        }
        /* Custom header gradient */
        [data-testid="stHeader"] {
            background-image: linear-gradient(90deg, #003366, #0055a4);
        }
        /* Style for metric cards */
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        /* Style for tabs */
       .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
       .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #F0F2F6;
            border-radius: 8px 8px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
       .stTabs [aria-selected="true"] {
            background-color: #FFFFFF;
        }
        </style>
    """, unsafe_allow_html=True)

# --- Data Loading and Caching ---
# Use st.cache_data for efficient data loading, refreshing every 24 hours.
@st.cache_data(ttl=24*3600, show_spinner="Fetching latest model metrics from Databricks...")
def load_metrics_data():
    """
    Connects to Databricks, fetches model monitoring metrics,
    and returns them in a pandas DataFrame. Caches the data to avoid
    re-fetching on every interaction.
    """
    conn = None  # Initialize conn to None
    try:
        # Securely connect using Streamlit secrets
        conn = sql.connect(
            server_hostname=st.secrets,
            http_path=st.secrets,
            access_token=st.secrets
        )
        engine = create_engine("databricks://", creator=lambda: conn)
        # Optimized query to select only necessary columns
        query = "SELECT day, event_ts, rmse_cost, accuracy_fraud, accuracy_readmit FROM workspace.claims_project.monitoring_metrics ORDER BY day"
        df = pd.read_sql(query, engine)
        df['event_ts'] = pd.to_datetime(df['event_ts'])
        return df
    except Exception as e:
        st.error(f"Failed to connect to Databricks or load data: {e}")
        return pd.DataFrame() # Return empty DataFrame on error
    finally:
        if conn:
            conn.close()

# --- Plotting Functions ---
def plot_rmse_analysis(df_filtered, df_full):
    """Plots RMSE time series with control bands and its distribution."""
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("RMSE Trend with Control Bands")
        # Calculate rolling stats on the full dataset for stable bands
        rmse_series = df_full["rmse_cost"]
        rmse_mean = rmse_series.rolling(window=7).mean()
        rmse_std = rmse_series.rolling(window=7).std()
        upper_band = rmse_mean + 2 * rmse_std
        lower_band = rmse_mean - 2 * rmse_std

        # Filter bands to the selected date range for plotting
        rmse_plot = df_filtered["rmse_cost"]
        upper_plot = upper_band.reindex(df_filtered.index)
        lower_plot = lower_band.reindex(df_filtered.index)

        # Identify anomalies
        anomalies = rmse_plot[rmse_plot > upper_plot]

        fig = go.Figure()
        # Control bands (plotted first to be in the background)
        fig.add_trace(go.Scatter(x=upper_plot.index, y=upper_plot, line=dict(color='rgba(0,0,0,0)'), showlegend=False, name='Upper Band'))
        fig.add_trace(go.Scatter(x=lower_plot.index, y=lower_plot, fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)', line=dict(color='rgba(0,0,0,0)'), showlegend=False, name='Lower Band'))
        # Main RMSE line
        fig.add_trace(go.Scatter(x=rmse_plot.index, y=rmse_plot, mode='lines+markers', name='RMSE', line=dict(color='#003366')))
        # Highlight anomalies
        if not anomalies.empty:
            fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies, mode='markers', name='Anomaly', marker=dict(color='red', size=10, symbol='x')))

        fig.update_layout(xaxis_title="Day", yaxis_title="RMSE", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The shaded red area indicates the expected RMSE range (±2σ) based on a 7-day rolling window. Points marked with 'X' are anomalies outside this range.")

    with col2:
        st.subheader("RMSE Distribution")
        # Create distribution plot
        if not df_filtered['rmse_cost'].empty:
            hist_data = [df_filtered['rmse_cost'].dropna()]
            group_labels =
            fig_dist = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=['#0055a4'])
            fig_dist.update_layout(xaxis_title="RMSE Value", yaxis_title="Density", showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
            st.caption("Distribution of RMSE values within the selected date range, showing the density (KDE) and individual data points (rug plot).")
        else:
            st.info("No RMSE data to display for distribution.")

def plot_accuracy_analysis(df, metric_col, title, threshold):
    """Plots accuracy time series with a threshold and its distribution."""
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader(f"{title} - Accuracy Trend")
        accuracy_plot = df[metric_col]
        anomalies = accuracy_plot[accuracy_plot < threshold]

        fig = go.Figure()
        # Threshold line
        fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold ({threshold}%)", annotation_position="bottom right")
        # Main accuracy line
        fig.add_trace(go.Scatter(x=accuracy_plot.index, y=accuracy_plot, mode='lines+markers', name='Accuracy', line=dict(color='#003366')))
        # Highlight anomalies
        if not anomalies.empty:
            fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies, mode='markers', name='Below Threshold', marker=dict(color='red', size=10, symbol='x')))

        fig.update_layout(xaxis_title="Day", yaxis_title="Accuracy (%)", yaxis_range=[min(70, df[metric_col].min()-5), 101], legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Accuracy trend over time. Points marked with 'X' have fallen below the {threshold}% performance threshold.")

    with col2:
        st.subheader("Accuracy Distribution")
        # Create distribution plot
        if not df[metric_col].empty:
            hist_data = [df[metric_col].dropna()]
            group_labels = ['Accuracy']
            fig_dist = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=['#0055a4'])
            fig_dist.update_layout(xaxis_title="Accuracy (%)", yaxis_title="Density", showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
            st.caption("Distribution of accuracy values within the selected date range.")
        else:
            st.info("No accuracy data to display for distribution.")

def plot_correlation_heatmap(df):
    """Calculates and plots a correlation heatmap for the key metrics."""
    st.subheader("Metric Correlation Heatmap")
    corr_df = df[['rmse_cost', 'accuracy_fraud', 'accuracy_readmit']].corr()
    # Create annotated heatmap
    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Blues',
        labels=dict(color="Correlation")
    )
    fig.update_layout(title_text='Correlation between Model Metrics', title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("This heatmap shows the correlation between the primary metrics of the three models. A value near 1 or -1 indicates a strong positive or negative correlation, respectively. A value near 0 indicates little to no correlation.")

# --- Main Application ---
def main():
    """Main function to run the Streamlit app."""
    load_css()

    st.title("Enterprise Model Monitoring Dashboard")
    st.markdown("""
    This dashboard provides real-time performance tracking for all deployed models.
    Metrics are fetched daily from Databricks and visualized below.
    Use the sidebar filter to zoom into specific time periods and explore each model's performance in its dedicated tab.
    """)

    df_metrics = load_metrics_data()

    if df_metrics.empty:
        st.error("No monitoring data could be loaded. Please check the Databricks connection or the source table.")
        st.stop()

    # --- Data Pre-processing ---
    df_by_day = df_metrics.set_index('day').sort_index()
    for col in ["accuracy_fraud", "accuracy_readmit"]:
        if col in df_by_day.columns:
            df_by_day[col] = df_by_day[col] * 100

    # --- Sidebar for Filters ---
    with st.sidebar:
        st.header("Filters")
        all_days = df_by_day.index.unique().tolist()
        if len(all_days) > 1:
            day_range = st.select_slider(
                "Select Day Range",
                options=all_days,
                value=(all_days, all_days[-1])
            )
            start_day, end_day = day_range
        else:
            start_day = end_day = all_days

        df_filtered = df_by_day.loc[start_day:end_day]

    if df_filtered.empty:
        st.warning("No data available in the selected day range. Please adjust the filter.")
        st.stop()

    # --- Key Performance Indicators (KPIs) ---
    st.header("Latest Day Performance Summary")
    latest_day_data = df_by_day.loc[df_by_day.index.max()]
    prev_day_data = df_by_day.loc[df_by_day.index.max() - 1] if len(df_by_day.index) > 1 else latest_day_data

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric(
            label="Latest Claims Cost RMSE",
            value=f"{latest_day_data['rmse_cost']:.2f}",
            delta=f"{latest_day_data['rmse_cost'] - prev_day_data['rmse_cost']:.2f} vs Previous Day",
            delta_color="inverse" # Lower is better
        )
    with kpi2:
        st.metric(
            label="Latest Fraud Model Accuracy",
            value=f"{latest_day_data['accuracy_fraud']:.1f}%",
            delta=f"{latest_day_data['accuracy_fraud'] - prev_day_data['accuracy_fraud']:.1f}% vs Previous Day"
        )
    with kpi3:
        st.metric(
            label="Latest Readmission Model Accuracy",
            value=f"{latest_day_data['accuracy_readmit']:.1f}%",
            delta=f"{latest_day_data['accuracy_readmit'] - prev_day_data['accuracy_readmit']:.1f}% vs Previous Day"
        )

    st.markdown("---")

    # --- Main Content Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs()

    with tab1:
        if "rmse_cost" in df_filtered.columns:
            plot_rmse_analysis(df_filtered, df_by_day)
        else:
            st.info("RMSE metric not available in the data.")

    with tab2:
        if "accuracy_fraud" in df_filtered.columns:
            plot_accuracy_analysis(df_filtered, "accuracy_fraud", "Fraud Detection", threshold=80)
        else:
            st.info("Fraud Detection metrics not available.")

    with tab3:
        if "accuracy_readmit" in df_filtered.columns:
            plot_accuracy_analysis(df_filtered, "accuracy_readmit", "Readmission Prediction", threshold=80)
        else:
            st.info("Readmission Prediction metrics not available.")

    with tab4:
        plot_correlation_heatmap(df_filtered)


if __name__ == "__main__":
    main()