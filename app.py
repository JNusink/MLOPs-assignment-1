import streamlit as st
import joblib
import numpy as np

# Cache the model loading for performance
@st.cache_data
def load_model():
    return joblib.load('sentiment_model.pkl')

# Load the model
model = load_model()

# App title and description
st.title("Movie Review Sentiment Analyzer")
st.write("Enter a movie review and get an instant sentiment prediction (Positive or Negative) along with confidence scores.")

# User input interface
user_input = st.text_area("Enter a movie review to analyze:", height=150)
analyze_button = st.button("Analyze")

# Prediction logic
if analyze_button:
    if user_input.strip() == "":
        st.warning("Please enter a review before analyzing.")
    else:
        # Make prediction
        prediction = model.predict([user_input])[0]
        probabilities = model.predict_proba([user_input])[0]
        
        # Get probability for positive and negative
        pos_prob = probabilities[model.classes_.tolist().index('positive')] * 100
        neg_prob = probabilities[model.classes_.tolist().index('negative')] * 100
        
        # Display results with colored output
        if prediction == 'positive':
            st.success(f"Predicted Sentiment: Positive 👍 ({pos_prob:.2f}% confidence)")
        else:
            st.error(f"Predicted Sentiment: Negative 👎 ({neg_prob:.2f}% confidence)")
        
        # Display probability details
        st.write(f"Positive Probability: {pos_prob:.2f}%")
        st.write(f"Negative Probability: {neg_prob:.2f}%")