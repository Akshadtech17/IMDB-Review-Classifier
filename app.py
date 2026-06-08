import streamlit as st
import pandas as pd
import pickle
import re
import string
import nltk

# Download required NLTK resources
resources = {
    'corpora/stopwords': 'stopwords',
    'tokenizers/punkt': 'punkt',
    'corpora/wordnet': 'wordnet'
}

for path, resource in resources.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Text preprocessing function
def preprocess(text):
    # Convert to lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    clean_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]

    return " ".join(clean_tokens)


# Page configuration
st.set_page_config(
    page_title="IMDb Review Classifier",
    page_icon="🎬",
    layout="centered"
)

# App Title
st.title("🎬 IMDb Review Sentiment Classifier")
st.markdown(
    "Analyze movie reviews and predict whether the sentiment is **Positive 😊** or **Negative 😞**."
)

# Load model and vectorizer
try:
    with open('untuned_logistic_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

except FileNotFoundError:
    st.error(
        "❌ Model or vectorizer file not found.\n\n"
        "Make sure the following files exist in your repository:\n"
        "- untuned_logistic_regression_model.pkl\n"
        "- vectorizer.pkl"
    )
    st.stop()

except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()


# User Input
user_input = st.text_area(
    "Enter a movie review:",
    height=150,
    placeholder="Example: This movie was absolutely fantastic. The acting and storyline were amazing!"
)

# Prediction Button
if st.button("Predict Sentiment"):

    if user_input.strip():

        try:
            # Preprocess text
            processed_input = preprocess(user_input)

            # Transform text
            vectorized_input = vectorizer.transform([processed_input])

            # Predict
            prediction = model.predict(vectorized_input)[0]

            # Prediction probability (if available)
            try:
                probability = model.predict_proba(vectorized_input)
                confidence = max(probability[0]) * 100
            except:
                confidence = None

            # Display result
            if prediction == 1:
                st.success("😊 Positive Review")

                if confidence:
                    st.info(f"Confidence: {confidence:.2f}%")

            else:
                st.error("😞 Negative Review")

                if confidence:
                    st.info(f"Confidence: {confidence:.2f}%")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

    else:
        st.warning("⚠️ Please enter a review before predicting.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, NLTK, Scikit-Learn, and Machine Learning")
