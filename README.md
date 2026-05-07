AlNasr SmartAssistant

An intelligent AI-powered document and knowledge assistant designed for enterprise-level search, question answering, and document understanding.  
AlNasr SmartAssistant transforms your documents into a smart, searchable knowledge system using advanced Retrieval-Augmented Generation (RAG) techniques.

---

🚀 Key Features

- 🔍 **Smart Semantic Search** across documents  
- 🧠 **AI-powered Question Answering (RAG System)**  
- 📄 Supports PDF, DOCX, PPT, and TXT files  
- ⚡ Hybrid Retrieval System (BM25 + FAISS Vector Search)  
- 🤖 AI Reranking for highly accurate results  
- 🧾 Automatic document summarization  
- 💬 Natural language query interface  
- 🌐 Interactive Streamlit web application  

---

🏗️ System Architecture

AlNasr SmartAssistant uses a multi-layer intelligence pipeline:

1. 📥 Document Ingestion
   - Upload and parse documents (PDF, DOCX, PPT, TXT)

2. ✂️ Text Chunking
   - Splits documents into meaningful semantic segments

3. 🔢 Embedding Generation
   - Converts text into dense vector representations

4. 🔎 Hybrid Retrieval
   - BM25 keyword-based search  
   - FAISS vector similarity search  

5. 🎯 Reranking Layer
   - Improves relevance of retrieved results  

6. 🤖 LLM Answer Generation
   - Produces final context-aware responses  

---

 🛠️ Tech Stack

- Python   
- Streamlit 🎨  
- FAISS (Vector Database)  
- BM25 (Keyword Search)  
- Sentence Transformers 🤗  
- Hugging Face Transformers  
- PyPDF / python-docx / python-pptx  

---

📂 Project Structure
 AlNasr-SmartAssistant
├── app.py
├── chatbot.py
├── utils/
│ ├── document_loader.py
│ ├── chunker.py
│ ├── retriever.py
│ ├── embedder.py
│ └── reranker.py
├── models/
├── data/
├── requirements.txt
└── README.md

---

⚙️ Installation & Setup

```bash
# Clone repository
git clone https://github.com/your-username/AlNasr-SmartAssistant.git

# Navigate into project
cd AlNasr-SmartAssistant

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py

🧠 How It Works
Upload your company or personal documents
System processes and indexes all content
Ask questions in natural language
AI retrieves the most relevant context
Generates accurate and contextual answers
🏢 Use Cases
Enterprise knowledge management
Construction & contracting document search
HR policy and internal document assistant
Legal & compliance document analysis
Research and technical documentation exploration
🔮 Future Enhancements
Multi-user enterprise authentication
Cloud deployment (AWS / Azure / GCP)
Real-time document syncing
Voice-based AI assistant
API integration for external systems
