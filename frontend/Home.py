# frontend/Home.py
import streamlit as st

st.title("Healthcare Claims ML Pipeline")
st.write("""
This app provides predictions for:
- **Claim Cost Prediction** (Regression)
- **Fraud Detection** (Classification)
- **30-Day Readmission Prediction** (Classification)

Use the sidebar to navigate to each section.
""")
st.info("Ensure the backend API is running and properly connected via st.secrets (see README for instructions).")
