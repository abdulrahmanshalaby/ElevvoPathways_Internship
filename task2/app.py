import re
import streamlit as st
import joblib
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars & numbers
    text = text.lower()  # Lowercase
    tokens = word_tokenize(text)  # Tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Remove stopwords + lemmatize
    return ' '.join(tokens)

# Load model
model = joblib.load("fake_new_classifier_linearsvc.pkl")

# Custom CSS for better UI
st.markdown("""
    <style>
        .main {background-color: #f9f9f9;}
        .stTextArea textarea {border-radius: 10px; font-size: 16px;}
        .stButton button {background-color: #4CAF50; color: white; font-size: 18px; border-radius: 8px;}
        .result-box {font-size: 22px; font-weight: bold; padding: 10px; border-radius: 10px; text-align: center;}
        .fake {background-color: #ffcccc; color: #cc0000;}
        .real {background-color: #ccffcc; color: #006600;}
    </style>
""", unsafe_allow_html=True)

st.title("📰 Fake News Detector")
st.write("Detect if a news article is **Real** or **Fake** using AI.")

# --- Single News Input ---
st.subheader("🔍 Check a Single News Article")
news_text = st.text_area("Enter news text:", height=150)

if st.button("Check News"):
    if news_text.strip():
        prediction = model.predict([news_text])[0]
        confidence = max(model.decision_function([news_text]))
        label = "Real News ✅" if prediction == 1 else "Fake News ❌"
        color_class = "real" if prediction == 1 else "fake"

        st.markdown(f'<div class="result-box {color_class}">{label}<br>Confidence: {confidence:.2f}</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter some text.")

# --- Batch News Upload ---
st.subheader("📂 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with a column 'text'", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'text' in df.columns:
        predictions = model.predict(df['text'])
        df['Prediction'] = ['Real ✅' if p == 1 else 'Fake ❌' for p in predictions]
        st.write("### Results:")
        st.dataframe(df[['text', 'Prediction']])
        st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")
    else:
        st.error("CSV must contain a column named 'text'.")
