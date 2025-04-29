import os
import shutil
import uuid
import uvicorn
import json
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Optional
from dotenv import load_dotenv
import asyncio
import faiss
import pickle
from langchain_community.chat_models import ChatOpenAI, ChatOllama
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

UPLOAD_DIR = "/home/harish/Project_works/FExpert/backup/uploaded_pdfs"
FAISS_INDEX_PATH = "/home/harish/Project_works/FExpert/backup/faiss_index/index.bin"
FAISS_DOCS_PATH = "/home/harish/Project_works/FExpert/backup/faiss_index/docs.pkl"
PDF_METADATA_PATH = "/home/harish/Project_works/FExpert/backup/metadata.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chains: Dict[str, ConversationalRetrievalChain] = {}

class AskRequest(BaseModel):
    question: str
    model_choice: Dict[str, str]
    file_name: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    top_k: int

def load_metadata():
    if os.path.exists(PDF_METADATA_PATH):
        with open(PDF_METADATA_PATH, "r") as f:
            return json.load(f)
    return []

def save_metadata(metadata):
    with open(PDF_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def encode_fn(texts):
    return embedding_model.encode(texts, show_progress_bar=False)

def load_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        with open(FAISS_DOCS_PATH, "rb") as f:
            stored_docs = pickle.load(f)
        index = faiss.read_index(FAISS_INDEX_PATH)
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(stored_docs)})
        return FAISS(encode_fn, index, docstore)
    return None

def save_faiss_index(faiss_index):
    if not faiss_index._documents:
        raise ValueError("No documents to save in FAISS index.")
    faiss.write_index(faiss_index.index, FAISS_INDEX_PATH)
    with open(FAISS_DOCS_PATH, "wb") as f:
        pickle.dump(faiss_index._documents, f)

def add_to_faiss_index(new_docs):
    new_index = FAISS.from_documents(new_docs, encode_fn)
    if not new_index._documents:
        raise ValueError("No documents were added to FAISS index.")
    existing_index = load_faiss_index()
    if existing_index:
        existing_index.merge_from(new_index)
        save_faiss_index(existing_index)
        return existing_index
    else:
        save_faiss_index(new_index)
        return new_index

def load_or_create_chain(file_path: str, model_type: str, model_name: str):
    file_id = os.path.basename(file_path)
    if file_id in chains:
        return chains[file_id]

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    faiss_index = add_to_faiss_index(chunks)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    if model_type == "ollama":
        llm = ChatOllama(model=model_name)
    elif model_type == "groq":
        llm = ChatOpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1", model_name=model_name)
    else:
        raise ValueError("Unsupported model type")

    chain = ConversationalRetrievalChain.from_llm(llm, faiss_index.as_retriever(), memory=memory, return_source_documents=True)
    chains[file_id] = chain
    return chain

preloaded_chains = {}

async def preload_model(file_path: str, model_type: str, model_name: str):
    file_id = os.path.basename(file_path)
    if file_id not in preloaded_chains:
        preloaded_chains[file_id] = load_or_create_chain(file_path, model_type, model_name)

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_name = f"{file_id}_{os.path.basename(file.filename)}"
    file_path = os.path.join(UPLOAD_DIR, file_name)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    metadata = load_metadata()
    metadata.append({
        "file_name": file_name,
        "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chunks": len(chunks)
    })
    save_metadata(metadata)

    return JSONResponse({"message": "File uploaded and metadata stored.", "file_name": file_name})

@app.get("/list_uploaded_pdfs")
async def list_uploaded_pdfs():
    metadata = load_metadata()
    return JSONResponse({"pdfs": metadata})

@app.post("/ask_question")
async def ask_question(payload: AskRequest):
    file_name = payload.file_name
    if not file_name:
        return JSONResponse(status_code=400, content={"error": "file_name is required"})

    file_path = os.path.join(UPLOAD_DIR, file_name)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found"})

    try:
        await preload_model(file_path, payload.model_choice["type"], payload.model_choice["model_name"])
        chain = preloaded_chains[os.path.basename(file_path)]
        result = await asyncio.to_thread(chain.invoke, {"question": payload.question})
        if not result:
            return JSONResponse(status_code=500, content={"error": "No response from chain"})
        answer = result.get("answer")
        sources = [doc.metadata.get("source", "") for doc in result.get("source_documents", [])]
        return JSONResponse({"answer": answer, "sources": sources, "model_name": payload.model_choice["model_name"]})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/search-chunks")
async def search_chunks(payload: SearchRequest):
    faiss_index = load_faiss_index()
    if not faiss_index:
        return JSONResponse(status_code=404, content={"error": "No FAISS index found"})

    matches = await asyncio.to_thread(faiss_index.similarity_search_with_relevance_scores, payload.query, payload.top_k)
    results = [{"text": doc.page_content, "score": score} for doc, score in matches]
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)[:payload.top_k]
    return JSONResponse({"results": sorted_results})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8007, reload=True)
