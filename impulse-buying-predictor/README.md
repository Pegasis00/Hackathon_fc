A machine learning project that predicts a customer’s impulse buying score (0–100) based on demographic, behavioral, purchase, and psychological factors. Designed for e-commerce 
analytics, personalization, and user-behavior insights.

Overview

This system generates synthetic e-commerce data, processes it into a structured dataset, trains an ML model, and provides a Streamlit web interface for real-time predictions.

The project includes:

Synthetic data generation (50k user profiles)

Feature engineering pipeline

LightGBM regression model

Streamlit dashboard for interactive predictions

Saved model, metrics, and feature information

Features

End-to-end ML workflow (data → model → app)

Realistic synthetic dataset

LightGBM model with high accuracy (R² ≈ 0.85)

Clean UI for entering user details

Explanation via feature importance

Ready for deployment on Streamlit Cloud

Project Structure
impulse-buying-predictor/
│
├── data/
│   └── data.csv              # Final processed dataset
│
├── models/
│   └── model.joblib          # Trained ML model
│
├── app.py                    # Streamlit interface
├── 1_generate_data.py        # Synthetic data generator
├── 2_process_data.py         # Feature engineering script
├── 3_train_model.py          # Model training pipeline
├── requirements.txt
└── README.md

Installation
1.Clone the repository
cd impulse-buying-predictor

2.Create a virtual environment
python -m venv venv
venv\Scripts\activate       # Windows

3.Install dependencies
pip install -r requirements.txt

Usage
Step 1: Generate synthetic data
python 1_generate_data.py

Step 2: Process and engineer features
python 2_process_data.py

Step 3: Train the ML model
python 3_train_model.py

Step 4: Launch the Streamlit app
streamlit run app.py

Model Details

Algorithm: LightGBM Regressor

Target: Impulse Buy Score (0–100)

Inputs include:

Age, income, city

Browsing metrics

Purchase history

Stress, mood, discount sensitivity

Performance

R² ≈ 0.85

RMSE ≈ 8–9

MAE ≈ 6–7

Deployment (Streamlit Cloud)

Push project to GitHub

Go to Streamlit Cloud → New App

Select:

Repo: your repo

Branch: main

Main file: app.py

Deploy

Future Improvements

API endpoint for predictions

Real user data integration

Explainability (SHAP)

A/B testing module
Dashboard for analytics
