import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Optional
from dotenv import load_dotenv

from langchain.chat_models import ChatGroq, ChatOllama
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
UPLOAD_DIR = "./uploaded_pdfs"
CHROMA_DIR = "./chroma_dbs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

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


def load_or_create_chain(file_path: str, model_type: str, model_name: str):
    file_id = os.path.basename(file_path)
    if file_id in chains:
        return chains[file_id]

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    vectordb_path = os.path.join(CHROMA_DIR, file_id)
    if os.path.exists(vectordb_path):
        vectordb = Chroma(persist_directory=vectordb_path, embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
    else:
        vectordb = Chroma.from_documents(chunks, OllamaEmbeddings(model="nomic-embed-text"), persist_directory=vectordb_path)
        vectordb.persist()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if model_type == "ollama":
        llm = ChatOllama(model=model_name)
    elif model_type == "groq":
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model_name)
    else:
        raise ValueError("Unsupported model type")

    chain = ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever(), memory=memory, return_source_documents=True)
    chains[file_id] = chain
    return chain


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, file_id + "_" + file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return JSONResponse({"message": "File uploaded and stored.", "file_name": os.path.basename(file_path)})


@app.post("/ask_question")
async def ask_question(payload: AskRequest):
    file_name = payload.file_name
    if not file_name:
        return JSONResponse(status_code=400, content={"error": "file_name is required"})

    file_path = os.path.join(UPLOAD_DIR, file_name)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found"})

    try:
        chain = load_or_create_chain(file_path, payload.model_choice["type"], payload.model_choice["model_name"])
        result = chain.invoke({"question": payload.question})
        answer = result.get("answer")
        sources = [doc.metadata.get("source", "") for doc in result.get("source_documents", [])]
        return JSONResponse({"answer": answer, "sources": sources, "model_name": payload.model_choice["model_name"]})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/search-chunks")
async def search_chunks(payload: SearchRequest):
    # Search all vector DBs
    results = []
    for subdir in os.listdir(CHROMA_DIR):
        vectordb = Chroma(persist_directory=os.path.join(CHROMA_DIR, subdir), embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
        matches = vectordb.similarity_search_with_score(payload.query, k=payload.top_k)
        for doc, score in matches:
            results.append({"text": doc.page_content, "score": score})
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)[:payload.top_k]
    return JSONResponse({"results": sorted_results})
