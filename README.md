# 🏥 Healthcare Claims ML Pipeline

A complete end-to-end machine learning project that predicts healthcare claims outcomes using synthetic data and XGBoost. This project includes:

---

## 📊 Exploratory Data Analysis (EDA) & Feature Engineering
- **Data Generation:** Synthetic patient-level claims data  
- **Missing Data:** Handling missing values with imputation strategies  
- **Visualization:** Distributions, correlations, and transformations (e.g., log transformation for skewed cost)

---

## 🧠 Model Training & Tuning
- **Training Pipelines for Three Tasks:**
  - **Claim Cost Prediction** (Regression)
  - **Fraud Detection** (Classification)
  - **30-Day Readmission Prediction** (Classification)
- **Technologies:**
  - Models built using XGBoost and scikit-learn  
  - Preprocessing pipelines combining imputation, one-hot encoding, and scaling  
  - Hyperparameter tuning (GridSearchCV optional)

---

## ⚙️ Backend API
- **Framework:** FastAPI backend serving real-time predictions and retraining endpoints  
- **Documentation:** Automatic interactive API docs via Swagger UI

---

## 🎨 Frontend Interface
- **Platform:** Multipage Streamlit app with dedicated pages for:
  - Home  
  - EDA & Insights  
  - Claim Cost Prediction  
  - Fraud Detection  
  - 30-Day Readmission Prediction
- **Features:** 
  - User-friendly forms for single and batch predictions  
  - Visualizations and clear insights

---

## ☁️ Deployment
- **Containerization:** Dockerfile provided for containerizing the FastAPI backend  
- **Hosting:** Deployed on Render/Railway (backend) and Streamlit Cloud (frontend)

---

## 🚀 Installation

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
   *This command creates `data/synthetic_claims.csv`.*

4. **Train the Models:**
   ```bash
   python notebooks/train_models.py
   ```
   *This script will:*
   - Load and preprocess the synthetic data.
   - Split data into training and testing sets.
   - Train pipelines for claim cost, fraud, and readmission.
   - Save the trained pipelines and preprocessor into the `models/` folder.

5. **Run the FastAPI Backend Locally:**
   ```bash
   uvicorn backend.main:app --reload
   ```
   *Then, open [http://localhost:8000/docs](http://localhost:8000/docs) to view the interactive API documentation.*

6. **Run the Streamlit Frontend Locally:**
   ```bash
   streamlit run frontend/Home.py
   ```
   *Use the sidebar to navigate between pages.*

---

## 🌟 Features

- **Real-Time Predictions:**  
  Get predictions for claim cost, fraud, and 30-day readmission based on user input.

- **Comprehensive EDA:**  
  Detailed visualizations including distributions (with log transformations if needed), correlation heatmaps, and boxplots.

- **End-to-End Model Pipelines:**  
  Preprocessing (imputation, encoding, scaling) combined with XGBoost models.

- **API Integration:**  
  FastAPI backend with endpoints for predictions and retraining, including file uploads for batch operations.

- **User-Friendly UI:**  
  Intuitive, multipage Streamlit interface with forms and downloadable results.

- **Deployment Ready:**  
  Dockerfile for backend containerization, with deployment on Render/Railway and Streamlit Cloud.

---

## 🔗 Live Demo

👉 [Try the live app here](https://your-streamlit-app-url.streamlit.app)

---

## 📊 Model Performance

- **Claim Cost Prediction:**
  - **Best Model:** XGBoost Regressor
  - **R² Score:** e.g., 0.82
  - **RMSE:** e.g., ~$4,500

- **Fraud Detection:**
  - **Best Model:** XGBoost Classifier
  - **Accuracy:** e.g., 95%

- **30-Day Readmission Prediction:**
  - **Best Model:** XGBoost Classifier
  - **Accuracy:** e.g., 88%  
  *(Parameters tuned using GridSearchCV and compared against multiple baselines.)*

---

## 💡 Technologies Used

- **Programming Language:** Python  
- **Data Manipulation:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Modeling:** Scikit-learn, XGBoost  
- **Backend:** FastAPI, Uvicorn  
- **Frontend:** Streamlit  
- **Containerization:** Docker  
- **Version Control:** Git & GitHub  

---

## 🤝 Contributions

This project is open for feedback, improvement, and collaboration. Feel free to fork, star, and open issues or pull requests!

---

## 📜 License

This project is licensed under the MIT License.

---