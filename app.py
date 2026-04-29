
import streamlit as st
import joblib
import numpy as np

# Load the model
try:
    model = joblib.load('rating_prediction_model.pkl')
except:
    st.error("Error: Model file 'rating_prediction_model.pkl' not found.")

st.set_page_config(page_title="Product Sentiment Predictor", page_icon="🛍️")
st.title("🛍️ Product Sentiment Predictor")
st.markdown("Use this tool to predict customer sentiment based on product metrics.")

# Input fields based on your project
st.subheader("Enter Product Metrics:")
col1, col2 = st.columns(2)

with col1:
    dp = st.number_input("Discounted Price", min_value=0.0)
    ap = st.number_input("Actual Price", min_value=0.0)
    perc = st.number_input("Discount Percentage", min_value=0.0)

with col2:
    count = st.number_input("Rating Count", min_value=0)
    pop = st.number_input("Popularity Score", min_value=0.0)

# Prediction Logic
if st.button("Analyze Sentiment"):
    input_data = np.array([[dp, ap, perc, count, pop]])
    prediction = model.predict(input_data)[0]
    
    st.divider()
    st.subheader(f"Predicted Score: {prediction:.2f}")
    
    if prediction >= 4.0:
        st.success("Result: **Positive Sentiment** 😊")
    else:
        st.error("Result: **Negative Sentiment** 😞")
