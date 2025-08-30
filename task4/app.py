import streamlit as st
from sentence_transformers import SentenceTransformer, util
import re
import nltk
from nltk.corpus import stopwords
import tempfile
import docx
import fitz  # PyMuPDF for PDF parsing

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------------
# Caching Model
# -------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -------------------------
# Helper Functions
# -------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

def extract_keywords(text1, text2):
    words1 = set(text1.split())
    words2 = set(text2.split())
    return list(words1.intersection(words2))

def read_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def read_docx(file):
    doc = docx.Document(file)
    text = "\n".join([p.text for p in doc.paragraphs])
    return text
# Define normalization function
def normalize_score(score, min_val=0.2, max_val=0.8):
    normalized = (score - min_val) / (max_val - min_val)
    normalized = max(0, min(1, normalized))  # Clamp between 0 and 1
    return round(normalized * 100, 2)  # Convert to percentage
# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title("📄 Resume & Job Description Matcher")
st.markdown("Upload your resume and paste the job description to check similarity.")

# Two columns for UI
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Your Resume")
    uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])

with col2:
    st.subheader("Job Description")
    job_description = st.text_area("Paste the job description here")

# Action Button
if st.button("Compute Similarity"):
    if uploaded_file is None:
        st.error("Please upload a resume.")
    elif not job_description.strip():
        st.error("Please enter a job description.")
    else:
        # Extract resume text
        if uploaded_file.type == "application/pdf":
            resume_text = read_pdf(uploaded_file)
        else:
            resume_text = read_docx(uploaded_file)

        # Preprocess texts
        job_clean = preprocess_text(job_description)
        resume_clean = preprocess_text(resume_text)

        # Compute embeddings
        with st.spinner("Calculating similarity..."):
            job_embedding = model.encode(job_clean, convert_to_tensor=True)
            resume_embedding = model.encode(resume_clean, convert_to_tensor=True)
            similarity_score = util.cos_sim(job_embedding, resume_embedding).item()
            similarity_percentage = normalize_score(similarity_score)

        # Extract keywords
        keywords_matched = extract_keywords(job_clean, resume_clean)

        # Display results
        st.success(f"✅ Similarity Score: **{similarity_percentage}%**")
        st.subheader("🔍 Matched Keywords")
        if keywords_matched:
            st.write(", ".join(keywords_matched))
        else:
            st.write("No significant keyword matches found.")
