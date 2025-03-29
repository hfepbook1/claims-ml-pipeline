# Healthcare Claims ML Pipeline

This repository implements a full-stack machine learning pipeline for synthetic healthcare claims data. It includes:

- **Data Generation:** A script to generate a synthetic dataset with 10,000+ patient-level records.
- **EDA & Preprocessing:** Code to clean, impute, encode, and scale data.
- **Model Training:** Training of three models:
  - Claim Cost Prediction (Regression using XGBoost)
  - Fraud Detection (Classification using XGBoost)
  - 30-Day Readmission Prediction (Classification using XGBoost)
- **Backend:** A FastAPI application serving prediction and retraining endpoints.
- **Frontend:** A multipage Streamlit app for user input and displaying predictions.
- **Deployment:** A Dockerfile for containerizing the FastAPI backend, with instructions for deployment on services like Render or Railway, and deployment of the Streamlit app on Streamlit Cloud.

