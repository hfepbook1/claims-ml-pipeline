# backend/routes/prediction.py

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import pandas as pd

router = APIRouter()

# Pydantic model for input features
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

@router.post("/cost")
async def predict_cost(claim: ClaimFeatures, request: Request):
    models = request.app.state.models
    if not models or "cost" not in models:
        raise HTTPException(status_code=500, detail="Cost model not loaded.")
    try:
        data = pd.DataFrame([claim.dict()])
        prediction = models["cost"].predict(data)
        return {"claim_cost": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fraud")
async def predict_fraud(claim: ClaimFeatures, request: Request):
    models = request.app.state.models
    if not models or "fraud" not in models:
        raise HTTPException(status_code=500, detail="Fraud model not loaded.")
    try:
        data = pd.DataFrame([claim.dict()])
        pred_class = int(models["fraud"].predict(data)[0])
        proba = float(models["fraud"].predict_proba(data)[0][1])
        return {"is_fraud": pred_class, "fraud_probability": proba}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/readmission")
async def predict_readmission(claim: ClaimFeatures, request: Request):
    models = request.app.state.models
    if not models or "readmit" not in models:
        raise HTTPException(status_code=500, detail="Readmission model not loaded.")
    try:
        data = pd.DataFrame([claim.dict()])
        pred_class = int(models["readmit"].predict(data)[0])
        proba = float(models["readmit"].predict_proba(data)[0][1])
        return {"readmit_30d": pred_class, "readmission_probability": proba}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
