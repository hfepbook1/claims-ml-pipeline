# backend/routes/retrain.py

from fastapi import APIRouter, HTTPException, UploadFile, File, Request
import pandas as pd
import io
import joblib
from xgboost import XGBRegressor, XGBClassifier
from sklearn.pipeline import Pipeline

router = APIRouter()

@router.post("/cost")
async def retrain_cost(file: UploadFile = File(...), request: Request = None):
    models = request.app.state.models
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        if 'claim_cost' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain a 'claim_cost' column.")
        X = df.drop(columns=['claim_cost'])
        y = df['claim_cost']
        regressor = XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
        new_pipeline = Pipeline([('xgb_reg', regressor)])
        new_pipeline.fit(X, y)
        models["cost"] = new_pipeline
        joblib.dump(new_pipeline, "models/model_cost.pkl")
        return {"detail": "Claim cost model retrained successfully", "samples_used": len(y)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fraud")
async def retrain_fraud(file: UploadFile = File(...), request: Request = None):
    models = request.app.state.models
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        if 'is_fraud' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain an 'is_fraud' column.")
        X = df.drop(columns=['is_fraud'])
        y = df['is_fraud']
        classifier = XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False,
                                     eval_metric='logloss', random_state=42)
        new_pipeline = Pipeline([('xgb_clf', classifier)])
        new_pipeline.fit(X, y)
        models["fraud"] = new_pipeline
        joblib.dump(new_pipeline, "models/model_fraud.pkl")
        return {"detail": "Fraud detection model retrained successfully", "samples_used": len(y)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/readmission")
async def retrain_readmission(file: UploadFile = File(...), request: Request = None):
    models = request.app.state.models
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        if 'readmit_30d' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain a 'readmit_30d' column.")
        X = df.drop(columns=['readmit_30d'])
        y = df['readmit_30d']
        classifier = XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False,
                                     eval_metric='logloss', random_state=42)
        new_pipeline = Pipeline([('xgb_clf', classifier)])
        new_pipeline.fit(X, y)
        models["readmit"] = new_pipeline
        joblib.dump(new_pipeline, "models/model_readmit.pkl")
        return {"detail": "Readmission model retrained successfully", "samples_used": len(y)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
