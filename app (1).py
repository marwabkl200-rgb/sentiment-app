
import streamlit as st
import joblib
import numpy as np

# تحميل الموديل
try:
    model = joblib.load('rating_model.pkl')
except:
    st.error("Error: Model file 'rating_model.pkl' not found.")

st.set_page_config(page_title="Product Sentiment Predictor", page_icon="🛍️")
st.title("🛍️ Product Sentiment Predictor")

st.subheader("Enter Product Metrics:")
col1, col2 = st.columns(2)

with col1:
    dp = st.number_input("Discounted Price", min_value=0.0)
    ap = st.number_input("Actual Price", min_value=0.0)
    perc = st.number_input("Discount Percentage", min_value=0.0)

with col2:
    count = st.number_input("Rating Count", min_value=0)
    pop = st.number_input("Popularity Score", min_value=0.0)
    # هادي هي الخانة السادسة اللي كانت ناقصة (مثال: Rating)
    extra_feat = st.number_input("Base Rating Score", min_value=0.0, max_value=5.0)

if st.button("Analyze Sentiment"):
    # دابا عندنا 6 ديال القيم:
    input_data = np.array([[dp, ap, perc, count, pop, extra_feat]])
    
    try:
        prediction = model.predict(input_data)[0]
        st.divider()
        st.subheader(f"Predicted Result: {prediction:.2f}")
        
        if prediction >= 4.0:
            st.success("Result: Positive Sentiment 😊")
        else:
            st.error("Result: Negative Sentiment 😞")
    except Exception as e:
        st.error(f"Prediction Error: Make sure you enter all 6 required values.")
