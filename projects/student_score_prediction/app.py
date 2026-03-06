import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("student_score_model.pkl")

st.title("Student Exam Score Predictor")

st.write("Enter student details to predict exam score")

# ---------------- INPUT FEATURES ---------------- #

age = st.number_input("Age", 15, 30)

total_study_hours = st.slider("Total Study Hours per Day", 0.0, 15.0)

entertainment_hours = st.slider("Entertainment Hours per Day", 0.0, 10.0)

sleep_hours = st.slider("Sleep Hours", 0.0, 12.0)

screen_time_hours = st.slider("Screen Time Hours", 0.0, 12.0)

exercise_minutes = st.number_input("Exercise Minutes Per Day", 0, 180)

caffeine_intake_mg = st.number_input("Caffeine Intake (mg)", 0, 500)

part_time_job = st.selectbox("Part Time Job", [0,1])

upcoming_deadline = st.selectbox("Upcoming Deadline", [0,1])

mental_health_score = st.slider("Mental Health Score", 1,10)

focus_index = st.slider("Focus Index", 0.0,10.0)

burnout_level = st.slider("Burnout Level", 0.0,10.0)

productivity_score = st.slider("Productivity Score", 0.0,10.0)

# ---------------- CREATE DATAFRAME ---------------- #

input_data = pd.DataFrame({
    "age":[age],
    "total_study_hours":[total_study_hours],
    "entertainment_hours":[entertainment_hours],
    "sleep_hours":[sleep_hours],
    "screen_time_hours":[screen_time_hours],
    "exercise_minutes":[exercise_minutes],
    "caffeine_intake_mg":[caffeine_intake_mg],
    "part_time_job":[part_time_job],
    "upcoming_deadline":[upcoming_deadline],
    "mental_health_score":[mental_health_score],
    "focus_index":[focus_index],
    "burnout_level":[burnout_level],
    "productivity_score":[productivity_score]
})

# ---------------- PREDICTION ---------------- #

if st.button("Predict Exam Score"):

    prediction = model.predict(input_data)

    st.success(f"Predicted Exam Score: {prediction[0]:.2f}")
