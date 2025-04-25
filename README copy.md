# 🧠 FactEntry Fixed Income Expert

**FactEntry Fixed Income Expert** is an internal knowledge assistant designed for new joinees at FactEntry to learn about our company operations, workflows, and fixed income bonds. Team members can upload company PDFs, which are processed, chunked, embedded, and stored for semantic search and LLM-based question answering.

---

## 🚀 Features

- 📄 Upload internal PDF documents  
- 🔍 Semantic search over company-specific content  
- 💬 Question-answering using local or API-based LLMs  
- 🧠 Uses vector embeddings + FAISS for context retrieval  
- 🖥️ Streamlit frontend + FastAPI backend  

---

## 🛠️ Tech Stack

| Layer        | Tech Used |
|--------------|-----------|
| Frontend     | Streamlit |
| Backend      | FastAPI |
| Vector Store | FAISS |
| Embeddings   | Google's Gecko Embeddings (`text-embedding-gecko-001`) or `nomic-embed-text` for local |
| LLM (QnA)    | Ollama (local: `llama3:70b`, `phi4`, `deepseek`, `qwen2.5`) or API-based (e.g., Groq) |
| PDF Handling | PyMuPDF (`fitz`), `pdfplumber`, `pdf2image`, `Pillow` |
| Chunking     | Custom 100-word chunking |
| DevOps       | Docker, Docker Compose, GitHub Actions |
| Hardware     | Runs locally with GPU support (e.g., NVIDIA RTX 3050) |

---

## 🔄 Project Pipeline

```mermaid
graph TD
A[PDF Upload] --> B[Extract Text & Images]
B --> C[Chunk Text (100-word chunks)]
C --> D[Embed with Google or Local Model]
D --> E[Store in FAISS Index]
F[User Asks Question] --> G[Embed Query]
G --> H[Search Top-k Chunks in FAISS]
H --> I[Send Chunks + Question to LLM]
I --> J[Display Answer in UI]
```

---

## 🧩 Core Components

### 1. PDF Upload  
PDFs from internal training documents or knowledge base are uploaded.  
Extracted using `pdfplumber` and `PyMuPDF`.

### 2. Chunking  
Content is split into 100-word segments.  
Metadata includes document name, page number, and chunk ID.

### 3. Embedding  
Choose between:  
- Google API (`text-embedding-gecko-001`)  
- Local: `nomic-embed-text` via Ollama  

### 4. FAISS Indexing  
All embedded chunks are stored in a persistent FAISS index.  
Supports fast similarity search.

### 5. Question Answering  
- User enters a question  
- Query is embedded  
- Top relevant chunks are retrieved from FAISS  
- Chunks + query sent to LLM  
- Supports both local (via Ollama) and cloud (e.g., Groq) models  

---

## ⚙️ Setup & Installation

### 🔧 Prerequisites
- Python 3.10+  
- Docker + Docker Compose  
- Google API Key (for Gecko embeddings, optional)  
- Ollama with models downloaded locally  

---

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up --build
```

---

## 🧪 Local Development (non-Docker)

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

```bash
# Frontend
cd frontend
streamlit run app.py
```

---

## 🧠 Supported LLMs via Ollama

Make sure these models are pulled locally:

```bash
ollama pull llama3
ollama pull phi4
ollama pull deepseek
ollama pull qwen2.5
ollama pull nomic-embed-text
```

---

## 🧠 Example Use Case

- **Upload:** Company onboarding guide (PDF)  
- **Ask:** “What is the fixed income data tagging workflow?”  
- ✅ **Answer:** Pulled from FAISS context, answered by LLaMA 3.3 via Ollama.

---

## 📁 Directory Structure

```bash
factentry-expert/
│
├── backend/
│   ├── main.py                # FastAPI server
│   ├── moderation.py          # Optional content moderation
│   └── utils/                 # Chunking, embedding, FAISS utils
│
├── frontend/
│   ├── app.py                 # Streamlit interface
│   └── assets/                # Logos, UI components
│
├── faiss_index/
│   └── index.bin              # FAISS index storage
│
├── models/
│   └── ollama_models.txt      # List of required local models
│v
└── requirements.txt
```

---

## 👨‍💼 Use Case: Internal Knowledge for New Joiners

This tool helps train new employees by exploring internal resources on:
- Fixed income bond workflows
- Company tools and processes
- Tagging standards and examples
- Research documentation

---

## 📌 Roadmap

- [x] PDF upload + chunking  
- [x] FAISS + embedding setup  
- [x] LLM Q&A (local + cloud)  
- [x] Streamlit UI  
- [ ] Admin dashboard for upload history  
- [ ] User feedback on answers  
- [ ] Role-based access (Employee vs Admin)  

---

## 🛡️ Security & Privacy

- Local-only processing for sensitive data  
- No external APIs unless configured  
- Option to turn off moderation layer  

---

## 🤝 Contributing

We're open to internal contributions!  
Fork the repo, create a branch, and submit a PR with your updates.

