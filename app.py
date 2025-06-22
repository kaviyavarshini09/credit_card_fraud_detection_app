import streamlit as st
import pickle
import xgboost as xgb
import numpy as np

# Load trained model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Credit Card Fraud Detection App")

st.write("Enter transaction features below (comma-separated):")

# Collect user input
user_input = st.text_input("Input 30 values:")

if user_input:
    try:
        input_list = [float(i) for i in user_input.split(",")]
        input_array = np.array(input_list).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)

        if prediction[0] == 1:
            st.error("❌ Fraud Detected!")
        else:
            st.success("✅ No Fraud Detected.")
    except:
        st.warning("Please enter exactly 30 comma-separated numeric values.")
