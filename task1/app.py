import re
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
# Keep negations
negations = {"not", "no", "nor", "never", "didn't", "doesn't", "isn't", "wasn't", "won't"}
stop_words = stop_words.difference(negations)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation and numbers
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)

    # Remove stopwords but keep negations
    tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)
# ✅ Load pipelines
lr_pipeline = joblib.load("logistic_regression_pipeline.pkl")
nb_pipeline = joblib.load("naive_bayes_pipeline.pkl")

# ✅ Sentiment mapping
sentiment_map = {1: "Positive 😊", 0: "Negative 😞"}

# ✅ UI
st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬", layout="wide")

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Compare predictions from **Logistic Regression** and **Naive Bayes** models.")

review = st.text_area("✍️ Enter a movie review:")

if st.button("Predict"):
    if review.strip():
        # ✅ Predictions
        pred_lr = lr_pipeline.predict([review])[0]
        pred_nb = nb_pipeline.predict([review])[0]

        # ✅ Probabilities
        prob_lr = lr_pipeline.predict_proba([review])[0]
        prob_nb = nb_pipeline.predict_proba([review])[0]

        # ✅ Layout: Two columns for models
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Logistic Regression")
            st.write(f"Prediction: **{sentiment_map[pred_lr]}**")
            st.progress(float(max(prob_lr)))  # Progress bar for confidence
            st.write(f"Confidence: {max(prob_lr)*100:.2f}%")

            # ✅ Probability breakdown
            prob_df_lr = pd.DataFrame({
                "Sentiment": ["Negative", "Positive"],
                "Probability": prob_lr
            })
            st.bar_chart(prob_df_lr.set_index("Sentiment"))

        with col2:
            st.subheader("Naive Bayes")
            st.write(f"Prediction: **{sentiment_map[pred_nb]}**")
            st.progress(float(max(prob_nb)))
            st.write(f"Confidence: {max(prob_nb)*100:.2f}%")

            # ✅ Probability breakdown
            prob_df_nb = pd.DataFrame({
                "Sentiment": ["Negative", "Positive"],
                "Probability": prob_nb
            })
            st.bar_chart(prob_df_nb.set_index("Sentiment"))
    else:
        st.warning("Please enter a review.")
