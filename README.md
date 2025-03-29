🏥 Healthcare Claims ML Pipeline  
A complete end-to-end machine learning project that predicts healthcare claims outcomes using synthetic data and XGBoost. The project includes:

📊 **Exploratory Data Analysis (EDA) & Feature Engineering**  
- Data generation for synthetic patient-level claims data  
- Handling missing values with imputation strategies  
- Visualizing distributions, correlations, and transformations (e.g., log transformation for skewed cost)

🧠 **Model Training & Tuning**  
- Training pipelines for three tasks:
  - **Claim Cost Prediction** (Regression)  
  - **Fraud Detection** (Classification)  
  - **30-Day Readmission Prediction** (Classification)  
- Models built using XGBoost and scikit-learn  
- Preprocessing pipelines combining imputation, one-hot encoding, and scaling  
- Hyperparameter tuning (GridSearchCV optional)

⚙️ **Backend API**  
- FastAPI backend serving real-time predictions and retraining endpoints  
- Automatic interactive API documentation via Swagger UI

🎨 **Frontend Interface**  
- Multipage Streamlit app with separate pages for:
  - Home  
  - EDA & Insights  
  - Claim Cost Prediction  
  - Fraud Detection  
  - 30-Day Readmission Prediction  
- User-friendly forms for single and batch predictions  
- Visualizations and clear insights

☁️ **Deployment**  
- Dockerfile provided for containerizing the FastAPI backend  
- Deployed on Render/Railway (backend) and Streamlit Cloud (frontend)

---

🚀 **Installation**

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/healthcare-claims-ml-pipeline.git
   cd healthcare-claims-ml-pipeline
   ```


2. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

3. **Generate Synthetic Data:**

```bash
python data/generate_synthetic_data.py
```
This command creates data/synthetic_claims.csv.

4. **Train the Models:**

bash
Copy
python notebooks/train_models.py
This will:

Load and preprocess the synthetic data.

Split data into training and testing sets.

Train three pipelines (for claim cost, fraud, and readmission).

Save the trained pipelines and preprocessor into the models/ folder.

Run the FastAPI Backend Locally:

bash
Copy
uvicorn backend.main:app --reload
Open http://localhost:8000/docs to view the interactive API documentation.

Run the Streamlit Frontend Locally:

bash
Copy
streamlit run frontend/Home.py
Use the sidebar to navigate between pages.

🌟 Features
Real-Time Predictions:

Get predictions for claim cost, fraud, and 30-day readmission based on user input.

Comprehensive EDA:

View detailed data visualizations, including distributions (with log transformations if needed), correlation heatmaps, and boxplots.

Model Pipelines:

End-to-end pipelines combining preprocessing (imputation, encoding, scaling) and XGBoost models.

API Integration:

FastAPI backend with endpoints for predictions and retraining (supports file uploads for batch operations).

User-Friendly UI:

Streamlit multipage interface with intuitive navigation, forms, and downloadable results.

Deployment Ready:

Dockerfile provided for backend containerization.

Deployed on Render/Railway (backend) and Streamlit Cloud (frontend).

🔗 Live Demo
👉 Try the live app here: https://your-streamlit-app-url.streamlit.app

📊 Model Performance
Claim Cost Prediction:

Best Model: XGBoost Regressor

R² Score: e.g., 0.82

RMSE: e.g., ~$4,500

Fraud Detection:

Best Model: XGBoost Classifier

Accuracy: e.g., 95%

30-Day Readmission Prediction:

Best Model: XGBoost Classifier

Accuracy: e.g., 88%

Parameters were tuned using GridSearchCV (optional) and compared against multiple baselines.

💡 Technologies Used
Programming Language: Python

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Modeling: Scikit-learn, XGBoost

Backend: FastAPI, Uvicorn

Frontend: Streamlit

Containerization: Docker

Version Control: Git & GitHub

🤝 Contributions
This project is open for feedback, improvement, and collaboration. Feel free to fork, star, and open issues or pull requests!

📜 License
This project is licensed under the MIT License.


