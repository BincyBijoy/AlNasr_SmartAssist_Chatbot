import os
import faiss
import pickle
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import re
import json
import requests

# CONFIG
st.set_page_config(page_title="Enterprise AI Assistant", layout="wide")

# SIDEBAR
with st.sidebar:
    st.title("🏢 Al Nasr AI")
    st.caption("Enterprise RAG Assistant")
    st.markdown("---")

    if st.button("🧹 Clear Chat"):
        st.session_state.chat = []
        st.rerun()

# HEADER
st.title("💬 Chat Assistant")

# PATHS
INDEX_PATH = "storage/index_v1.faiss"
CHUNKS_PATH = "storage/chunks_v1.pkl"
META_PATH = "storage/meta.json"

os.makedirs("storage", exist_ok=True)

# TEXT CLEANING
def clean_pdf_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r'-\s+', '', text)
    text = re.sub(r'\b(\w{1,4})\s+\1\b', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_text(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = ansi_escape.sub('', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
    return text.strip()

# MODELS
model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# SAFE SAVE
def safe_save(path, data):
    temp_path = path + ".tmp"
    with open(temp_path, "wb") as f:
        pickle.dump(data, f)
    os.replace(temp_path, path)

# BUILD KNOWLEDGE BASE
def build_kb():
    folder = "data/"
    if not os.path.exists(folder):
        os.makedirs(folder)
        st.error("Add PDFs inside /data folder")
        st.stop()

    texts = []

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder, file))
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    texts.append(clean_pdf_text(t))

    chunks = []
    for t in texts:
        chunks += [t[i:i+500] for i in range(0, len(t), 250)]

    chunks = [clean_pdf_text(c) for c in chunks if len(c.strip()) > 20]

    embeddings = model.encode(chunks, batch_size=32, show_progress_bar=False)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    faiss.write_index(index, INDEX_PATH)
    safe_save(CHUNKS_PATH, (chunks, bm25))

    with open(META_PATH, "w") as f:
        json.dump({"version": 1, "chunks": len(chunks)}, f)

    return index, chunks, bm25

# LOAD OR BUILD
@st.cache_resource
def load_or_build():
    try:
        if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
            index = faiss.read_index(INDEX_PATH)
            with open(CHUNKS_PATH, "rb") as f:
                chunks, bm25 = pickle.load(f)
            return index, chunks, bm25
    except Exception as e:
        st.warning(f"Rebuilding index... ({e})")

    return build_kb()

index, chunks, bm25 = load_or_build()

# RETRIEVAL (OPTIMIZED)
def retrieve(query):
    q_emb = model.encode([query])

    # reduced k for speed
    _, I = index.search(np.array(q_emb), k=12)

    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_results = np.argsort(bm25_scores)[::-1][:12]

    return list(set(I[0].tolist() + bm25_results.tolist()))

# RERANK (OPTIMIZED INPUT SIZE)
def rerank(query, candidates):
    # limit rerank load
    candidates = list(candidates)[:15]

    pairs = [(query, chunks[i]) for i in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked[:5]]

# LLM CALL (OPTIMIZED TIMEOUT)
def ask_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        },
        timeout=60
    )
    return clean_text(response.json()["response"])

# CHAT STATE
if "chat" not in st.session_state:
    st.session_state.chat = []

# INPUT
query = st.chat_input("Ask about Al Nasr Contracting Company...")

# PIPELINE (OPTIMIZED)
if query:
    with st.spinner("Thinking..."):
        candidates = retrieve(query)

        # adaptive rerank (speed boost)
        if len(query.split()) < 6:
            top_chunks = candidates[:3]
        else:
            top_chunks = rerank(query, candidates)

        # trimmed context (faster LLM)
        context = "\n\n".join([
            chunks[i][:300] for i in top_chunks
        ])

        prompt = f"""
You are an enterprise AI assistant for Al Nasr Contracting Company LLC.

Use ONLY the context below:

{context}

Question:
{query}

Answer clearly and professionally:
"""

        answer = ask_llm(prompt)

    st.session_state.chat.append(("user", query))
    st.session_state.chat.append(("bot", answer))

# CHAT UI
for role, msg in st.session_state.chat:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(
                f"<div style='text-align:right;background:#DCF8C6;padding:10px;border-radius:10px'>{msg}</div>",
                unsafe_allow_html=True
            )
    else:
        with st.chat_message("assistant"):
            st.markdown(
                f"<div style='background:#F1F1F1;padding:12px;border-radius:10px'>{msg}</div>",
                unsafe_allow_html=True
            )
