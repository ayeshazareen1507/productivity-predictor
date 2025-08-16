import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Load dataset and train model
# -------------------------------
df = pd.read_csv('personal_productivity_dataset.csv')

# Preprocess
X = df.drop('Productive', axis=1)
y = df['Productive']
X = pd.get_dummies(X, columns=['Weather'], drop_first=True)

model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X, y)


# Helper function for tips

def recommend_tip(sample_row):
    tip = ""
    if sample_row['SleepHours'] < 6.5:
        tip += "Try to sleep at least 7 hours. "
    if sample_row['ScreenTime'] > 6:
        tip += "Reduce screen time to stay focused. "
    if sample_row['ExerciseMins'] < 15:
        tip += "Add some exercise to improve energy. "
    if sample_row['Meals'] < 3:
        tip += "Maintain regular meals. "
    if tip == "":
        tip = "You're on track! Keep the routine."
    return tip


# Streamlit UI

st.title("🧠 Personal Productivity Predictor")
st.header("Enter Your Daily Data:")

wake_time = st.slider("Wake-up Hour", 5, 10, 7)
sleep_hours = st.slider("Sleep Hours", 4.5, 9.0, 7.0)
screen_time = st.slider("Screen Time (hours)", 0, 12, 5)
exercise_mins = st.slider("Exercise Minutes", 0, 60, 30)
weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy"])
meals = st.slider("Number of Meals", 1, 4, 3)


# Prepare input for prediction

# Start with user input
input_data = pd.DataFrame({
    'WakeTimeHour': [wake_time],
    'SleepHours': [sleep_hours],
    'ScreenTime': [screen_time],
    'ExerciseMins': [exercise_mins],
    'Meals': [meals],
    'Weather_Cloudy': [1 if weather=='Cloudy' else 0],
    'Weather_Rainy': [1 if weather=='Rainy' else 0]
})

# Align input columns with training features
model_features = X.columns
for col in model_features:
    if col not in input_data.columns:
        input_data[col] = 0  # Add missing columns as zeros

# Reorder columns to match training
input_data = input_data[model_features]


if st.button("Predict Productivity"):
    prediction = model.predict(input_data)[0]
    label = "Productive" if prediction == 1 else "Not Productive"
    st.subheader(f"🧠 Predicted Label: {label}")
    
    tip = recommend_tip(input_data.iloc[0])
    st.info(f"💡 Tip: {tip}")
