import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("Claims Data Overview")

# Load data
try:
    df = pd.read_csv('data/synthetic_claims.csv')
except Exception as e:
    st.error("Failed to load data: " + str(e))
    df = pd.DataFrame()

# If claim_date missing, simulate dataset
if 'claim_date' not in df.columns:
    # simulate dates and fields for demonstration
    dates = pd.date_range(start='2023-01-01', periods=24, freq='M')
    df = pd.DataFrame({
        'claim_date': np.random.choice(dates, size=500),
        'claim_cost': np.random.gamma(shape=2, scale=1000, size=500),
        'is_fraud': np.random.choice([0,1], size=500, p=[0.95, 0.05]),
        'readmit_30d': np.random.choice([0,1], size=500, p=[0.9, 0.1])
    })

# Convert claim_date to datetime and extract month
if 'claim_date' in df.columns:
    df['claim_date'] = pd.to_datetime(df['claim_date'])
    df['month'] = df['claim_date'].dt.to_period('M').dt.to_timestamp()
else:
    df['month'] = pd.NaT

# Monthly claim volume with simple forecast
if 'month' in df.columns and not df.empty:
    monthly_volume = df.groupby('month').size().reset_index(name='claims')
    # Forecast next month as average of last 3 months
    if len(monthly_volume) >= 3:
        last3_avg = monthly_volume['claims'].iloc[-3:].mean()
    elif len(monthly_volume) > 0:
        last3_avg = monthly_volume['claims'].mean()
    else:
        last3_avg = 0
    forecast_month = pd.to_datetime(monthly_volume['month'].max()) + pd.offsets.MonthBegin(1)
    # Append forecast month
    monthly_volume = pd.concat([monthly_volume, pd.DataFrame({'month': [forecast_month], 'claims': [last3_avg]})], ignore_index=True)
    fig_volume = px.line(monthly_volume, x='month', y='claims',
                         title='Monthly Claim Volume (with Simple Forecast)', markers=True)
    fig_volume.update_traces(mode='lines+markers')
else:
    fig_volume = None

# Monthly total claim cost with 3-month rolling average
if 'month' in df.columns and 'claim_cost' in df.columns and not df.empty:
    monthly_cost = df.groupby('month')['claim_cost'].sum().reset_index()
    monthly_cost['3mo_avg'] = monthly_cost['claim_cost'].rolling(window=3).mean()
    fig_cost = px.line(monthly_cost, x='month', y='claim_cost',
                       title='Monthly Claim Cost (with 3-Month Rolling Avg)')
    fig_cost.add_scatter(x=monthly_cost['month'], y=monthly_cost['3mo_avg'],
                         mode='lines', name='3-Month Rolling Avg')
else:
    fig_cost = None

# Display charts
st.subheader("Claim Trends")
if fig_volume:
    st.plotly_chart(fig_volume, use_container_width=True)
if fig_cost:
    st.plotly_chart(fig_cost, use_container_width=True)

# Derived KPIs
fraud_cost = df.loc[df['is_fraud']==1, 'claim_cost'].sum() if 'is_fraud' in df.columns and 'claim_cost' in df.columns else 0
baseline_recall = 0.5  # assumed baseline recall
fraud_improvement = 0.60  # 60% lift in detection
fraud_avoided_cost = fraud_improvement * (fraud_cost / baseline_recall) if baseline_recall > 0 else 0

# Readmission savings
if 'readmit_30d' in df.columns and 'claim_cost' in df.columns:
    delta_rate = 0.15  # 15% reduction
    avg_readmit_cost = df.loc[df['readmit_30d']==1, 'claim_cost'].mean() if (df['readmit_30d']==1).any() else df['claim_cost'].mean()
    num_cases = len(df)
    readmit_savings = delta_rate * avg_readmit_cost * num_cases
else:
    readmit_savings = 0

# Preventive care impact: cost of flagged vs general
if 'claim_cost' in df.columns:
    threshold = df['claim_cost'].quantile(0.90)
    flagged = df[df['claim_cost'] >= threshold]
    general = df[df['claim_cost'] < threshold]
    if not general.empty and general['claim_cost'].mean() > 0:
        flagged_cost_ratio = flagged['claim_cost'].mean() / general['claim_cost'].mean()
    else:
        flagged_cost_ratio = None
else:
    flagged_cost_ratio = None

# Triage efficiency: % of claims > $20k
if 'claim_cost' in df.columns:
    high_cost_claims = df[df['claim_cost'] > 20000]
    triage_percent = (len(high_cost_claims) / len(df) * 100) if len(df) > 0 else 0
else:
    triage_percent = 0

# Display KPIs
st.subheader("Key Metrics")
col1, col2 = st.columns(2)
col1.metric("Fraud Avoided Cost", f"${fraud_avoided_cost:,.0f}")
col1.metric("Readmission Savings", f"${readmit_savings:,.0f}")
col2.metric("% High-Cost Claims (> $20k)", f"{triage_percent:.1f}%")
col2.metric("Flagged vs General Cost Ratio", f"{flagged_cost_ratio:.2f}" if flagged_cost_ratio else "N/A")
