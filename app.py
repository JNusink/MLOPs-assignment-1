import streamlit as st
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    filename='app.log',  # Log file name
    level=logging.DEBUG,  # Capture all levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

# Cache the model loading for performance
@st.cache_data
def load_model():
    try:
        logging.info("Attempting to load model.pkl")
        model = joblib.load('model.pkl')
        logging.info("Model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Failed to load model.pkl: {e}")
        raise

# Load the model
logging.info("Starting application")
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
        logging.warning("No review entered")
    else:
        logging.info("Processing user input")
        try:
            prediction = model.predict([user_input])[0]
            probabilities = model.predict_proba([user_input])[0]
            logging.info("Prediction and probabilities calculated")
            
            pos_prob = probabilities[model.classes_.tolist().index('positive')] * 100
            neg_prob = probabilities[model.classes_.tolist().index('negative')] * 100
            
            logging.debug(f"Positive probability: {pos_prob:.2f}%, Negative probability: {neg_prob:.2f}%")
            
            if prediction == 'positive':
                st.success(f"Predicted Sentiment: Positive 👍 ({pos_prob:.2f}% confidence)")
                logging.info(f"Predicted: Positive with {pos_prob:.2f}% confidence")
            else:
                st.error(f"Predicted Sentiment: Negative 👎 ({neg_prob:.2f}% confidence)")
                logging.info(f"Predicted: Negative with {neg_prob:.2f}% confidence")
            
            st.write(f"Positive Probability: {pos_prob:.2f}%")
            st.write(f"Negative Probability: {neg_prob:.2f}%")
            logging.info("Probabilities displayed")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            logging.error(f"Prediction error: {e}")