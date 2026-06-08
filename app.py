
import streamlit as st
import pandas as pd
import pickle
import re
import string
import nltk

# Download NLTK data (if not already present) - crucial for deployment environments
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the preprocess function dependencies
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # 1. Lowercase
    text = str(text).lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # 3. Remove numbers + special characters (only keep a-z and spaces)
    text = re.sub(r'[^a-z\s]', '', text)

    # 4. Tokenization
    tokens = word_tokenize(text)

    # 5. Remove stopwords + lemmatization
    clean_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]
    return " ".join(clean_tokens) # Join back to string for vectorizer

# Load the saved model and vectorizer
try:
    with open('untuned_logistic_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    st.success("Model and vectorizer loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model or vectorizer files not found. Please ensure 'untuned_logistic_regression_model.pkl' and 'vectorizer.pkl' are in the same directory as your app.py.")
    st.stop() # Stop the app if files are not found

# Streamlit App Layout
st.title("IMDb Primary Title Adult Content Classifier")
st.write("Enter a primary title to predict if it's classified as adult content (1) or not (0).")

user_input = st.text_area("Enter Primary Title here:", "Example: A movie about a happy family")

if st.button("Predict"): # Button to trigger prediction
    if user_input:
        # Preprocess the input text
        processed_input = preprocess(user_input)

        # Vectorize the preprocessed text
        # The vectorizer expects a list of strings, even for a single input
        vectorized_input = vectorizer.transform([processed_input])

        # Make prediction
        prediction = model.predict(vectorized_input)

        # Display result
        if prediction[0] == 1:
            st.error(f"Prediction: **Adult Content (1)**") # Use st.error for adult content
        else:
            st.success(f"Prediction: **Not Adult Content (0)**") # Use st.success for non-adult content
    else:
        st.warning("Please enter some text to make a prediction.")
