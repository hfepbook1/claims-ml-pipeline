# backend/models/model_loader.py

import joblib

def load_models():
    models = {}
    try:
        models["cost"] = joblib.load("models/model_cost.pkl")
        models["fraud"] = joblib.load("models/model_fraud.pkl")
        models["readmit"] = joblib.load("models/model_readmit.pkl")
        print("Models loaded successfully.")
    except Exception as e:
        print("Error loading models:", e)
        models = {}
    return models
