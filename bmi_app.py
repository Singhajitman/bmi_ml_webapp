import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("bmi_model.pkl")  # Load the saved model
scaler = joblib.load("scaler.pkl")    # Load the saved scaler

# Streamlit UI
st.title("BMI Prediction App")
st.write("Enter details to predict BMI category")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)

# Convert gender to numerical value
gender_num = 1 if gender == "Male" else 0

# Predict when button is clicked
if st.button("Predict BMI Category"):
    # Prepare input data
    user_data = pd.DataFrame([[gender_num, height, weight]], columns=["Gender", "Height", "Weight"])
    user_data_scaled = scaler.transform(user_data)  # Apply the same scaling

    # Make prediction
    prediction = model.predict(user_data_scaled)
    
    # Show result
    st.success(f"Predicted BMI Category: **{prediction[0]}**")
