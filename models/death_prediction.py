import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("predictiing_death_model.pkl")
model_columns = joblib.load("death_model_columns.pkl")

st.set_page_config(page_title="Death Risk Predictor", layout="centered")

st.title("⚕️ Death Risk Prediction System")
st.write("Enter patient details to predict death risk.")

# -----------------------------
# User Inputs
# -----------------------------

age = st.number_input("Age", 0, 120, 40)
bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
days_hospitalized = st.number_input("Days Hospitalized", 0, 100, 3)

gender = st.selectbox("Gender", ["Male", "Female"])
severity = st.selectbox("Severity", ["Mild", "Moderate", "Severe"])
smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
alcohol_use = st.selectbox("Alcohol Use", ["Yes", "No"])
insurance_status = st.selectbox("Insurance Status", ["Insured", "Not Insured"])
hospital_type = st.selectbox("Hospital Type", ["Private", "Government"])

# -----------------------------
# Predict Button
# -----------------------------

if st.button("Predict Death Risk"):

    # Create dataframe
    input_data = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "days_hospitalized": days_hospitalized,
        "gender": gender,
        "severity": severity,
        "smoking_status": smoking_status,
        "alcohol_use": alcohol_use,
        "insurance_status": insurance_status,
        "hospital_type": hospital_type
    }])

    # Apply encoding
    input_encoded = pd.get_dummies(input_data, drop_first=True)

    # Match training columns
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    if prediction == 1:
        st.error(f"⚠ High Death Risk Detected ({probability*100:.2f}% probability)")
    else:
        st.success(f"✅ Low Death Risk ({probability*100:.2f}% probability)")