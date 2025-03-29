# frontend/Fraud_Detection.py
import streamlit as st
import requests
import pandas as pd
import numpy as np

st.title("Fraud Detection")
st.write("Enter claim details to predict whether the claim is fraudulent.")

BACKEND_URL = st.secrets["backend_url"] if "backend_url" in st.secrets else "http://localhost:8000"

with st.form("fraud_form"):
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    gender = st.selectbox("Gender", ["F", "M"])
    region = st.selectbox("Region", ["Northeast", "Midwest", "South", "West"])
    provider_type = st.selectbox("Provider Type", ["PrimaryCare", "Specialist", "Hospital", "UrgentCare"])
    chronic_count = st.number_input("Chronic Condition Count", min_value=0, max_value=20, value=0)
    primary_dx = st.selectbox("Primary Diagnosis", ["None", "Diabetes", "Heart Disease", "Orthopedic", "Cancer"])
    num_visits = st.number_input("Number of Visits", min_value=0, max_value=100, value=0)
    num_er_visits = st.number_input("Number of ER Visits", min_value=0, max_value=50, value=0)
    num_inpt_stays = st.number_input("Number of Inpatient Stays", min_value=0, max_value=20, value=0)
    submitted = st.form_submit_button("Predict Fraud")

if submitted:
    payload = {
        "age": age,
        "gender": gender,
        "region": region,
        "provider_type": provider_type,
        "chronic_condition_count": chronic_count,
        "primary_diagnosis": primary_dx,
        "num_visits": num_visits,
        "num_er_visits": num_er_visits,
        "num_inpatient_stays": num_inpt_stays
    }
    try:
        response = requests.post(f"{BACKEND_URL}/predict_fraud", json=payload, timeout=5)
        if response.status_code == 200:
            pred = response.json()["is_fraud"]
            if pred == 1:
                st.error("The claim is predicted to be FRAUDULENT!")
            else:
                st.success("The claim is predicted as NOT fraudulent.")
        else:
            st.error(f"Prediction failed: {response.text}")
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.subheader("Batch Prediction / Retraining")
uploaded_file = st.file_uploader("Upload CSV for batch prediction or retraining", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
    mode = st.radio("Choose action", ["Batch Prediction", "Retrain Model"])
    if st.button("Run", key="fraud_run"):
        if mode == "Batch Prediction":
            predictions = []
            for _, row in df.iterrows():
                data = {
                    "age": int(row["age"]),
                    "gender": row["gender"],
                    "region": row["region"],
                    "provider_type": row["provider_type"],
                    "chronic_condition_count": int(row["chronic_condition_count"]),
                    "primary_diagnosis": row["primary_diagnosis"],
                    "num_visits": int(row["num_visits"]),
                    "num_er_visits": int(row["num_er_visits"]),
                    "num_inpatient_stays": int(row["num_inpatient_stays"])
                }
                res = requests.post(f"{BACKEND_URL}/predict_fraud", json=data)
                if res.status_code == 200:
                    predictions.append(res.json()["is_fraud"])
                else:
                    predictions.append(np.nan)
            df["predicted_is_fraud"] = predictions
            st.write("Batch Prediction Results:")
            st.dataframe(df.head())
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions", data=csv, file_name="fraud_predictions.csv")
        else:
            if "is_fraud" not in df.columns:
                st.error("For retraining, CSV must include 'is_fraud' column.")
            else:
                files = {"file": uploaded_file.getvalue()}
                res = requests.post(f"{BACKEND_URL}/retrain_fraud", files=files)
                if res.status_code == 200:
                    st.success("Fraud model retrained successfully!")
                else:
                    st.error(f"Retraining failed: {res.text}")
