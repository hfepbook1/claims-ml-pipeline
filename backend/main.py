# backend/main.py
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import io
from xgboost import XGBRegressor, XGBClassifier
from sklearn.pipeline import Pipeline
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Healthcare Claims ML API", description="API for Claim Cost, Fraud, and Readmission predictions", version="1.0")

# Setup CORS to allow all origins (modify in production to restrict domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained model pipelines at startup
try:
    model_cost = joblib.load("models/model_cost.pkl")
    model_fraud = joblib.load("models/model_fraud.pkl")
    model_readmit = joblib.load("models/model_readmit.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
except Exception as e:
    print("Error loading models:", e)
    raise e

# Define Pydantic model for claim features (single record input)
class ClaimFeatures(BaseModel):
    age: int
    gender: str
    region: str
    provider_type: str
    chronic_condition_count: int
    primary_diagnosis: str
    num_visits: int
    num_er_visits: int
    num_inpatient_stays: int

# Prediction endpoints
@app.post("/predict_cost")
def predict_cost(item: ClaimFeatures):
    try:
        # Convert input to DataFrame
        data = pd.DataFrame([item.dict()])
        # Use the pipeline to predict
        prediction = model_cost.predict(data)
        return {"claim_cost": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_fraud")
def predict_fraud(item: ClaimFeatures):
    try:
        data = pd.DataFrame([item.dict()])
        pred = model_fraud.predict(data)
        # Optionally, get probability: model_fraud.predict_proba(data)[0][1]
        return {"is_fraud": int(pred[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_readmission")
def predict_readmission(item: ClaimFeatures):
    try:
        data = pd.DataFrame([item.dict()])
        pred = model_readmit.predict(data)
        return {"readmit_30d": int(pred[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Retraining endpoints (example for fraud model; similar endpoints can be made for cost and readmission)
@app.post("/retrain_fraud")
def retrain_fraud(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
        # Check that necessary columns exist
        required_cols = ["age", "gender", "region", "provider_type", "chronic_condition_count", 
                         "primary_diagnosis", "num_visits", "num_er_visits", "num_inpatient_stays", "is_fraud"]
        if not set(required_cols).issubset(df.columns):
            raise HTTPException(status_code=400, detail=f"Uploaded CSV must contain: {required_cols}")
        # Split into features and target
        X_new = df.drop(columns=["is_fraud"])
        y_new = df["is_fraud"]
        # Build a new pipeline for fraud detection
        from xgboost import XGBClassifier
        from sklearn.pipeline import Pipeline
        new_model = XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, 
                                  eval_metric='logloss', random_state=42)
        new_pipeline = Pipeline([("preprocessor", preprocessor), ("xgb_clf", new_model)])
        new_pipeline.fit(X_new, y_new)
        # Update global model and save
        global model_fraud
        model_fraud = new_pipeline
        joblib.dump(new_pipeline, "models/model_fraud.pkl")
        return {"detail": "Fraud model retrained successfully", "samples_used": len(y_new)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
