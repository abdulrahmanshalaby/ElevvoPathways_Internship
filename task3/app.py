import streamlit as st
from summa.summarizer import summarize as textrank_summarize
from concurrent.futures import ThreadPoolExecutor

# --- Assume your cached BART setup stays exactly the same ---
from transformers import pipeline
import os

@st.cache_resource
def load_bart_model():
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers")
    os.makedirs(cache_dir, exist_ok=True)
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1
    )
    return summarizer

bart_summarizer = load_bart_model()

@st.cache_data
def get_bart_summary(text):
    return bart_summarizer(
        text,
        max_length=60,
        min_length=50,
        do_sample=False,
        truncation=True
    )[0]['summary_text']

# --- TextRank (extractive) ---
@st.cache_data
def get_textrank_summary(text):
    return textrank_summarize(text, ratio=0.5)

# --- Streamlit interface ---
st.title("Dual Summarization: TextRank Async + Cached BART")

text = st.text_area("Paste your article here:", height=200)

if st.button("Generate Summaries"):
    if not text.strip():
        st.warning("Please enter some text!")
    else:
        # --- Run TextRank immediately ---
        extractive_summary = get_textrank_summary(text)

        # Display TextRank immediately in left column
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Extractive (TextRank)")
            st.write(extractive_summary)

        # --- Run BART in background thread (right column) ---
        with st.spinner("Generating BART summary..."):
            with ThreadPoolExecutor() as executor:
                future = executor.submit(get_bart_summary, text)
                abstractive_summary = future.result()

        with col2:
            st.subheader("Abstractive (BART)")
            st.write(abstractive_summary)
