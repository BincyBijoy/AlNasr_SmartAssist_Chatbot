import os
import faiss
import pickle
import numpy as np
import streamlit as st
import re
import json

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import pipeline

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Enterprise AI Assistant", layout="wide")

# =========================
# PATHS
# =========================
INDEX_PATH = "storage/index.faiss"
CHUNKS_PATH = "storage/chunks.pkl"
META_PATH = "storage/meta.json"
os.makedirs("storage", exist_ok=True)

# =========================
# CLEANING
# =========================
def clean_pdf_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r'-\s+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
    return text.strip()

# =========================
# MODELS
# =========================
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-mpnet-base-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    return embedder, reranker

embedder, reranker = load_models()

# =========================
# LLM (FIXED FOR STREAMLIT CLOUD)
# =========================
@st.cache_resource
def load_llm():
    return pipeline(
        "text-generation",
        model="google/flan-t5-base"
    )

llm = load_llm()

# =========================
# SAFE SAVE
# =========================
def safe_save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp, path)

# =========================
# BUILD KNOWLEDGE BASE
# =========================
def build_kb():
    folder = "data/"
    if not os.path.exists(folder):
        os.makedirs(folder)
        st.error("Add PDF files inside /data folder")
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
    chunks = [c for c in chunks if len(c) > 30]

    embeddings = embedder.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    safe_save(CHUNKS_PATH, chunks)

    with open(META_PATH, "w") as f:
        json.dump({"chunks": len(chunks)}, f)

    return index, chunks

# =========================
# LOAD OR BUILD
# =========================
def load_or_build():
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    return build_kb()

index, chunks = load_or_build()

# =========================
# BM25
# =========================
@st.cache_resource
def build_bm25(chunks):
    tokenized = [c.lower().split() for c in chunks]
    return BM25Okapi(tokenized)

bm25 = build_bm25(chunks)

# =========================
# RETRIEVAL (HYBRID)
# =========================
def retrieve(query, faiss_k=40, bm25_k=40):
    q_emb = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    _, faiss_ids = index.search(q_emb, faiss_k)

    faiss_candidates = set(faiss_ids[0])

    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_candidates = set(np.argsort(bm25_scores)[-bm25_k:])

    return list(faiss_candidates | bm25_candidates)

# =========================
# RERANK
# =========================
def rerank(query, candidates, top_k=5):
    if not candidates:
        return []

    pairs = [(query, chunks[i]) for i in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked[:top_k]]

# =========================
# LLM ANSWER (FIXED OUTPUT)
# =========================
def llm_answer(query, context):
    prompt = f"""
You are an enterprise assistant.

RULES:
- Use ONLY the context below
- If answer is not found, say "Not found in documents"
- Do not hallucinate

Question:
{query}

Context:
{context}

Answer:
"""

    try:
        result = llm(prompt, max_new_tokens=256, do_sample=False)
        return result[0]["generated_text"]
    except:
        return None

# =========================
# CHAT STATE
# =========================
if "chat" not in st.session_state:
    st.session_state.chat = []

# =========================
# UI
# =========================
st.title("💬 Chat Assistant")

with st.sidebar:
    st.title("🏢 Al Nasr AI")
    st.caption("Enterprise Document Assistant")
    st.markdown("---")
    if st.button("🧹 Clear Chat"):
        st.session_state.chat = []
        st.rerun()

query = st.chat_input("Ask your question...")

# =========================
# PIPELINE
# =========================
if query:
    with st.spinner("Searching documents..."):
        candidates = retrieve(query)
        top_chunks = rerank(query, candidates)

        contexts = [clean_pdf_text(chunks[i]) for i in top_chunks]
        context_text = "\n\n".join(contexts)

        answer = llm_answer(query, context_text)

        if not answer:
            answer = clean_text(context_text[:800])

    st.session_state.chat.append(("user", query))
    st.session_state.chat.append(("bot", answer))

# =========================
# CHAT DISPLAY
# =========================
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