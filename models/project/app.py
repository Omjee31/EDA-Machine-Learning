import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("student_score_model.pkl")
features = joblib.load("features.pkl")

st.title("🎓 Student Exam Score Predictor")

st.write("Enter student details to predict exam score")

# Inputs
total_study_hours = st.slider("Total Study Hours", 0.0, 15.0, 5.0)
entertainment_hours = st.slider("Entertainment Hours", 0.0, 10.0, 2.0)
focus_index = st.slider("Focus Index", 0.0, 10.0, 5.0)
burnout_level = st.slider("Burnout Level", 0.0, 10.0, 3.0)
productivity_score = st.slider("Productivity Score", 0.0, 10.0, 5.0)

# Create dataframe for prediction
input_data = pd.DataFrame({
    "Total_study_hours":[total_study_hours],
    "entertainment_hours":[entertainment_hours],
    "focus_index":[focus_index],
    "burnout_level":[burnout_level],
    "productivity_score":[productivity_score]
})

# Prediction
if st.button("Predict Exam Score"):

    prediction = model.predict(input_data)

    st.success(f"Predicted Exam Score: {prediction[0]:.2f}")