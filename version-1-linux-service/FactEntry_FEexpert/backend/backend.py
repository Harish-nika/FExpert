from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
import uuid
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import json
import ollama
from dotenv import load_dotenv
from groq import Groq  
# ----------------------------- #
# Environment and Initialization
# ----------------------------- #
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY")) 

app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL = SentenceTransformer("all-MiniLM-L6-v2")
DIM = 384
INDEX = faiss.IndexFlatL2(DIM)
CHUNKS = []
METADATA = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------- #
# Utilities
# ----------------------------- #
def chunk_text(text: str, chunk_size: int = 100 ) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def extract_text_from_pdf(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)
    return [page.get_text() for page in doc if page.get_text()]

def create_embedding_and_store(chunks: List[str], source_name: str):
    global INDEX, CHUNKS, METADATA
    embeddings = MODEL.encode(chunks)
    INDEX.add(np.array(embeddings, dtype=np.float32))
    CHUNKS.extend(chunks)
    METADATA.extend([source_name] * len(chunks))

def semantic_search(query: str, top_k: int = 5):
    query_vec = MODEL.encode([query]).astype(np.float32)
    distances, indices = INDEX.search(query_vec, top_k)
    results = [{
        "text": CHUNKS[i],
        "score": float(distances[0][j]),
        "source": METADATA[i]
    } for j, i in enumerate(indices[0])]
    return results

def generate_answer_groq(prompt: str, context: str) -> str:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a financial analyst expert for fixed income."
            },
            {
                "role": "user",
                "content": f"{prompt}\n\nContext:\n{context}"
            }
        ],
        model="llama3-8b-8192",  # Make sure this matches your original use
    )
    return chat_completion.choices[0].message.content.strip()

def generate_answer_ollama(model_name: str, prompt: str, context: str) -> str:
    full_prompt = f"You are a fixed income finance expert.\n\nContext:\n{context}\n\nQuestion:\n{prompt}"
    response = ollama.chat(model=model_name, messages=[
        {"role": "user", "content": full_prompt}
    ])
    return response["message"]["content"].strip()

# ----------------------------- #
# Request Schemas
# ----------------------------- #
class QuestionRequest(BaseModel):
    question: str
    model_choice: dict  # {'type': 'groq' or 'ollama', 'model_name': 'name'}

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

# ----------------------------- #
# API Endpoints
# ----------------------------- #
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        raw_text = extract_text_from_pdf(file_path)
        chunks = []
        for page in raw_text:
            chunks.extend(chunk_text(page))

        create_embedding_and_store(chunks, file.filename)
        return {"message": "PDF indexed successfully"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask_question/")
async def ask_question(payload: QuestionRequest):
    try:
        results = semantic_search(payload.question, top_k=5)
        context = "\n\n".join([r["text"] for r in results])
        sources = [r["source"] for r in results]

        if payload.model_choice["type"] == "groq":
            answer = generate_answer_groq(payload.question, context)
        else:
            model_name = payload.model_choice["model_name"]
            answer = generate_answer_ollama(model_name, payload.question, context)

        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/search-chunks")
async def search_chunks(payload: SearchRequest):
    try:
        results = semantic_search(payload.query, top_k=payload.top_k)
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}
