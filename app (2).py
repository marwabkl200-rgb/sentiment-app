
import streamlit as st
import joblib
import numpy as np

# تحميل الموديل
model = joblib.load('rating_model.pkl')

st.title("🛍️ Product Sentiment Predictor")

# 6 ديال الخانات كيفما عندك في الكولاب
col1, col2 = st.columns(2)
with col1:
    dp = st.number_input("Discounted Price", value=150.0)
    ap = st.number_input("Actual Price", value=200.0)
    perc = st.number_input("Discount Percentage", value=25.0)
with col2:
    rt = st.number_input("Rating", value=4.0)
    rc = st.number_input("Rating Count", value=1500.0)
    ps = st.number_input("Popularity Score", value=0.8)

if st.button("Analyze"):
    # الترتيب الصحيح للمصفوفة
    features = np.array([[dp, ap, perc, rt, rc, ps]])
    try:
        prediction = model.predict(features)[0]
        st.success(f"Prediction: {prediction:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
