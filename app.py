import streamlit as st
import pickle
import re
import string
import nltk

# Download required NLTK resources
resources = {
    'corpora/stopwords': 'stopwords',
    'corpora/wordnet': 'wordnet',
    'corpora/omw-1.4': 'omw-1.4'
}

for path, resource in resources.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    """
    Preprocess input text:
    - Lowercase
    - Remove URLs
    - Remove special characters/numbers
    - Remove stopwords
    - Lemmatize words
    """

    # Lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Simple tokenization (avoids punkt/punkt_tab issues)
    tokens = text.split()

    # Remove stopwords and lemmatize
    clean_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]

    return " ".join(clean_tokens)


# Streamlit page configuration
st.set_page_config(
    page_title="IMDb Review Classifier",
    page_icon="🎬",
    layout="centered"
)

# Title
st.title("🎬 IMDb Review Sentiment Classifier")
st.write(
    "Enter a movie review below and the model will predict whether the sentiment is **Positive 😊** or **Negative 😞**."
)

# Load model and vectorizer
try:
    with open("untuned_logistic_regression_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

except FileNotFoundError:
    st.error(
        "❌ Model or vectorizer file not found.\n\n"
        "Ensure these files exist in your repository root:\n"
        "- untuned_logistic_regression_model.pkl\n"
        "- vectorizer.pkl"
    )
    st.stop()

except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# User input
user_input = st.text_area(
    "Enter Movie Review:",
    placeholder="Example: This movie was absolutely fantastic! The acting and storyline were amazing.",
    height=180
)

# Prediction button
if st.button("Predict Sentiment"):

    if user_input.strip():

        try:
            # Preprocess
            processed_input = preprocess(user_input)

            # Vectorize
            vectorized_input = vectorizer.transform([processed_input])

            # Predict
            prediction = model.predict(vectorized_input)[0]

            # Get confidence score if available
            confidence = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(vectorized_input)[0]
                confidence = max(probabilities) * 100

            # Display result
            st.markdown("---")

            if prediction == 1:
                st.success("😊 Positive Review")

                if confidence is not None:
                    st.info(f"Confidence: {confidence:.2f}%")

            else:
                st.error("😞 Negative Review")

                if confidence is not None:
                    st.info(f"Confidence: {confidence:.2f}%")

        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")

    else:
        st.warning("⚠️ Please enter a review.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit • NLTK • Scikit-Learn • Machine Learning")
