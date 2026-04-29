import streamlit as st
import joblib
import numpy as np

# Load the trained model
try:
    model = joblib.load('rating_model.pkl')
except:
    st.error("Model file 'rating_model.pkl' not found!")

st.set_page_config(page_title="Product Sentiment Predictor", layout="centered")
st.title("🛍️ Product Sentiment Predictor")
st.write("Enter product details to predict the rating score.")

# Input fields for the 6 features
col1, col2 = st.columns(2)

with col1:
    dp = st.number_input("Discounted Price", value=150.0)
    ap = st.number_input("Actual Price", value=200.0)
    perc = st.number_input("Discount Percentage", value=25.0)

with col2:
    rt = st.number_input("Rating Score", value=4.0)
    rc = st.number_input("Rating Count", value=1500.0)
    ps = st.number_input("Popularity Score", value=0.8)

if st.button("Predict Rating"):
    # Array of 6 features to match the model's training
    features = np.array([[dp, ap, perc, rt, rc, ps]])
    
    try:
        prediction = model.predict(features)[0]
        st.success(f"The Predicted Result is: {prediction:.2f}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
