{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "1a2580b7-5426-41b6-b372-ceecefdd370a",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'backend.routes'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[28], line 5\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mfastapi\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m FastAPI\n\u001b[0;32m      4\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mfastapi\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmiddleware\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mcors\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m CORSMiddleware\n\u001b[1;32m----> 5\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mbackend\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mroutes\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m prediction, retrain\n\u001b[0;32m      6\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mbackend\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmodels\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmodel_loader\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m load_models\n\u001b[0;32m      7\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01muvicorn\u001b[39;00m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'backend.routes'"
     ]
    }
   ],
   "source": [
    "# backend/main.py\n",
    "\n",
    "from fastapi import FastAPI\n",
    "from fastapi.middleware.cors import CORSMiddleware\n",
    "from backend.routes import prediction, retrain\n",
    "from backend.models.model_loader import load_models\n",
    "import uvicorn\n",
    "\n",
    "app = FastAPI(\n",
    "    title=\"Healthcare Claims ML API\",\n",
    "    description=\"API endpoints for claim cost prediction, fraud detection, and 30-day readmission prediction\",\n",
    "    version=\"1.0\"\n",
    ")\n",
    "\n",
    "# Setup CORS (adjust allowed origins in production)\n",
    "app.add_middleware(\n",
    "    CORSMiddleware,\n",
    "    allow_origins=[\"*\"],\n",
    "    allow_credentials=True,\n",
    "    allow_methods=[\"*\"],\n",
    "    allow_headers=[\"*\"],\n",
    ")\n",
    "\n",
    "# Load models at startup and store them in the app state\n",
    "app.state.models = load_models()\n",
    "\n",
    "# Include routers for prediction and retraining endpoints\n",
    "app.include_router(prediction.router, prefix=\"/predict\", tags=[\"Prediction\"])\n",
    "app.include_router(retrain.router, prefix=\"/retrain\", tags=[\"Retraining\"])\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    uvicorn.run(\"backend.main:app\", host=\"0.0.0.0\", port=8000, reload=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c8610553-9ebc-41f0-aa8f-7e8b03d58412",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
