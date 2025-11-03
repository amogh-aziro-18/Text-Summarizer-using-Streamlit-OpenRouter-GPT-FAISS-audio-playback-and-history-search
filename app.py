import os
import time
import pdfplumber
import numpy as np
import pandas as pd
import streamlit as st
import requests
import tiktoken
import faiss
import joblib
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from gtts import gTTS

# ============================================
# 🌍 Setup
# ============================================
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("MODEL", "gpt-oss-20b")

if not API_KEY:
    st.error("⚠️ Missing API key. Please add it to your .env file.")
    st.stop()

st.set_page_config(
    page_title="🧠 Smart Summarizer",
    page_icon="🧠",
    layout="wide"
)

# ============================================
# 🌙 Custom Styling (darker, more neon accents)
# ============================================
st.markdown("""
<style>
    .main {
        background-color: #080A12;
        color: #EAEAEA;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #00C6FF;
    }
    .card {
        background-color: #121625;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 20px rgba(0,255,255,0.1);
        margin-bottom: 25px;
        position: relative;
    }
    .copy-btn {
        position: absolute;
        top: 12px;
        right: 15px;
        background-color: #0077FF;
        border: none;
        color: white;
        border-radius: 8px;
        padding: 4px 10px;
        cursor: pointer;
        font-size: 14px;
        transition: 0.2s;
    }
    .copy-btn:hover {
        background-color: #00C6FF;
        transform: scale(1.05);
    }
    .stButton>button {
        background: linear-gradient(90deg, #0077FF, #00C6FF);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6em 1.2em;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #00C6FF, #0077FF);
        transform: scale(1.02);
    }
    .stTextArea textarea, .stTextInput>div>div>input {
        background-color: #161A2D !important;
        color: #EAEAEA !important;
        border-radius: 8px !important;
    }
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background: #00C6FF;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 🧩 Helper Functions
# ============================================
def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()

def num_tokens(text):
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4

def summarize_text(text, model=MODEL, max_tokens=500):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert summarizer. Keep it short and clear."},
            {"role": "user", "content": f"Summarize this text in a concise way:\n\n{text}"}
        ],
        "max_tokens": max_tokens,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"API Error: {r.status_code} - {r.text}")

def save_summary(text, summary):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    row = pd.DataFrame([[timestamp, text[:800], summary]], columns=["time", "text", "summary"])
    os.makedirs("history", exist_ok=True)
    path = "history/summaries.csv"
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([row, old], ignore_index=True)
    else:
        df = row
    df.to_csv(path, index=False)

def update_faiss_index():
    path = "history/summaries.csv"
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    if df.empty:
        return None, None
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["summary"]).toarray().astype("float32")
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    joblib.dump(vectorizer, "history/vectorizer.pkl")
    faiss.write_index(index, "history/faiss.index")
    return index, vectorizer

def search_similar(query, index, vectorizer, top_k=3):
    q_vec = vectorizer.transform([query]).toarray().astype("float32")
    D, I = index.search(q_vec, top_k)
    return I[0]

def clear_history():
    folder = "history"
    if os.path.exists(folder):
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
        return True
    return False

# ============================================
# 🎨 UI
# ============================================
st.markdown("<h1 style='text-align:center;'>🧠 Smart Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#aaa;'>Summarize PDFs or text, listen to them, and search past summaries.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📝 Summarize", "📂 History & FAISS Search"])

# ============================================
# TAB 1: Summarize + Audio
# ============================================
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📄 Upload a PDF or paste your text")

    file = st.file_uploader("Upload a PDF", type=["pdf"])
    manual_text = st.text_area("Or paste your text below:", height=180)

    text = ""
    if file:
        text = read_pdf(file)
        st.success(f"✅ Extracted {len(text)} characters from {file.name}")
    elif manual_text.strip():
        text = manual_text.strip()

    if text:
        st.caption(f"Token estimate: {num_tokens(text)}")

    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "audio_bytes" not in st.session_state:
        st.session_state.audio_bytes = None

    if st.button("✨ Generate Summary"):
        if not text:
            st.warning("Please provide text first.")
        else:
            with st.spinner("Generating summary... ⏳"):
                try:
                    summary = summarize_text(text)
                    save_summary(text, summary)
                    update_faiss_index()
                    st.session_state.summary = summary
                    st.session_state.audio_bytes = None
                    st.success("✅ Summary generated successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.summary:
        summary = st.session_state.summary
        st.markdown("### 🧾 Your Summary")
        st.markdown(f"<div class='card'>{summary}</div>", unsafe_allow_html=True)
        st.download_button("⬇️ Download Summary", summary, file_name="summary.txt")

        if st.button("🔊 Listen to Audio Summary"):
            tts = gTTS(summary)
            tts.save("summary_audio.mp3")
            with open("summary_audio.mp3", "rb") as f:
                st.session_state.audio_bytes = f.read()

        if st.session_state.audio_bytes:
            st.audio(st.session_state.audio_bytes, format="audio/mp3")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# TAB 2: History & Search
# ============================================
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔍 Search through your summaries (FAISS)")
    st.markdown("<p style='color:#00C6FF;'>Powered by FAISS Vector Search</p>", unsafe_allow_html=True)

    path = "history/summaries.csv"
    if not os.path.exists(path):
        st.info("No summaries found yet.")
    else:
        df = pd.read_csv(path)

        col1, col2 = st.columns([3,1])
        with col1:
            query = st.text_input("Search summaries by meaning:", "")
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                if clear_history():
                    st.success("History cleared successfully! Refresh the app.")
                else:
                    st.warning("No history to clear.")

        if query:
            index, vectorizer = update_faiss_index()
            if index and vectorizer:
                results = search_similar(query, index, vectorizer, top_k=3)
                for i in results:
                    if 0 <= i < len(df):
                        with st.expander(f"🕒 {df.iloc[i]['time']} — View Summary"):
                            st.markdown(f"<div class='card'>{df.iloc[i]['summary']}</div>", unsafe_allow_html=True)
            else:
                st.warning("Index not found. Try summarizing something first.")
        else:
            st.dataframe(df[["time", "summary"]])

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# 🌙 Footer
# ============================================
st.markdown("""
<style>
footer {
    position: relative;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 1.2em 0;
    margin-top: 2em;
    color: #888;
    font-size: 14px;
    background: rgba(15, 17, 25, 0.6);
    border-top: 1px solid rgba(0, 198, 255, 0.2);
    backdrop-filter: blur(10px);
}
footer a {
    color: #00C6FF;
    text-decoration: none;
    font-weight: 500;
}
footer a:hover {
    text-decoration: underline;
}
</style>

<footer>
    <p>💡 Built by <a href="https://github.com/amogh-aziro-18" target="_blank">Amogh</a></p>
</footer>
""", unsafe_allow_html=True)
