import streamlit as st
import pickle
import numpy as np
import pickle

with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)


model = pickle.load(open('xgb_model.pkl', 'rb'))

st.title("💳 Credit Card Fraud Detection App")

st.write("Enter 29 numbers (V1 to V28 and normAmount) separated by commas")

user_input = st.text_input("Example: 0.1, -1.3, ..., 0.9")

if st.button("Predict"):
    try:
        values = [float(i) for i in user_input.split(',')]
        if len(values) != 29:
            st.error("❗ Please enter exactly 29 values.")
        else:
            result = model.predict(np.array(values).reshape(1, -1))
            label = "🚨 Fraud" if result[0] == 1 else "✅ Legit"
            st.success(f"Prediction: {label}")
    except:
        st.error("❗ Invalid input format.")
