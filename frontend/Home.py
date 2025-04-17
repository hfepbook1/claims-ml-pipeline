# frontend/Home.py
import streamlit as st

# Main Page Content
st.set_page_config(page_title="Healthcare Claims ML Pipeline", page_icon="🏥")
st.title("Healthcare Claims ML Pipeline")

# Use st.secrets if available; otherwise default to localhost.
BACKEND_URL = st.secrets.get("backend_url", "https://healthcare-claims-ml-pipeline.onrender.com")

# Sidebar Navigation
st.sidebar.info("Use the sidebar to jump to each section of the app.")

st.markdown("""
Welcome to the **Healthcare Claims ML Pipeline** app!  
This application demonstrates an end-to-end machine learning pipeline for healthcare claims data. It offers predictive insights for:

- **Claim Cost Prediction (Regression):** Estimate the cost of a claim based on patient demographics, clinical information, and service utilization.
- **Fraud Detection (Classification):** Identify potentially fraudulent claims.
- **30-Day Readmission Prediction (Classification):** Predict the likelihood of a patient being readmitted within 30 days.

### How It Works
1. **Data Pipeline:** Synthetic healthcare claims data is generated, preprocessed, and used to train predictive models.
2. **Modeling:** Three models (one for each task) are trained using techniques like XGBoost.
3. **API Integration:** A FastAPI backend serves the models and supports real-time predictions as well as model retraining.
4. **Interactive UI:** Use the navigation sidebar to access each feature page, enter input details, or upload data files for batch predictions or retraining.

Enjoy exploring the app and feel free to retrain the models with your own data if needed!
""")

st.info("If you are new here, start by exploring each section from the sidebar. Happy exploring!")
