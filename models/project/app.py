import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("student_score_model.pkl")

st.title("🎓 Student Exam Score Predictor")

st.write("Enter student lifestyle and study details")

# -------- USER INPUTS -------- #

age = st.number_input("Age", 10, 60)

gender = st.selectbox("Gender", [0,1], help="0 = Female, 1 = Male")

academic_level = st.selectbox("Academic Level", [1,2,3,4])

sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)

screen_time_hours = st.slider("Screen Time Hours", 0.0, 12.0, 4.0)

exercise_minutes = st.number_input("Exercise Minutes Per Day", 0, 300)

caffeine_intake_mg = st.number_input("Caffeine Intake (mg)", 0, 500)

part_time_job = st.selectbox("Part Time Job", [0,1])

upcoming_deadline = st.selectbox("Upcoming Deadline", [0,1])

internet_quality = st.selectbox("Internet Quality", [1,2,3,4,5])

mental_health_score = st.slider("Mental Health Score", 1, 10)

focus_index = st.slider("Focus Index", 0.0, 1.0)

burnout_level = st.slider("Burnout Level", 0.0, 1.0)

productivity_score = st.slider("Productivity Score", 0.0, 1.0)

Total_study_hours = st.number_input("Total Study Hours Per Day", 0.0, 15.0)

entertainment_hours = st.number_input("Entertainment Hours", 0.0, 10.0)

# -------- DATAFRAME -------- #

input_data = pd.DataFrame({
    "age":[age],
    "gender":[gender],
    "academic_level":[academic_level],
    "sleep_hours":[sleep_hours],
    "screen_time_hours":[screen_time_hours],
    "exercise_minutes":[exercise_minutes],
    "caffeine_intake_mg":[caffeine_intake_mg],
    "part_time_job":[part_time_job],
    "upcoming_deadline":[upcoming_deadline],
    "internet_quality":[internet_quality],
    "mental_health_score":[mental_health_score],
    "focus_index":[focus_index],
    "burnout_level":[burnout_level],
    "productivity_score":[productivity_score],
    "Total_study_hours":[Total_study_hours],
    "entertainment_hours":[entertainment_hours]
})

# -------- PREDICTION -------- #

if st.button("Predict Exam Score"):
    
    prediction = model.predict(input_data)

    st.success(f"📊 Predicted Exam Score: {prediction[0]:.2f}")