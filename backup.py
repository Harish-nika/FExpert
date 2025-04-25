# main.py (FastAPI Backend for Fixed Income Expert Chatbot)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import os
import numpy as np
import faiss
import pickle
import re
import time
from typing import List
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Directories
EMBEDDING_DIR = "./embeddings"
TEXT_CHUNKS_DIR = "./text_chunks"
FAISS_INDEX_PATH = "./faiss_index.index"
EMBEDDINGS_PATH = "./embeddings.npy"
CHUNKS_PATH = "./chunk_texts.pkl"

os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(TEXT_CHUNKS_DIR, exist_ok=True)

# Load models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192", temperature=0)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load existing knowledge base if available
if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(EMBEDDINGS_PATH) and os.path.exists(CHUNKS_PATH):
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    all_embeddings = np.load(EMBEDDINGS_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        all_chunks = pickle.load(f)
    print("[INFO] Loaded existing knowledge base.")
else:
    faiss_index = None
    all_embeddings = None
    all_chunks = []
    print("[INFO] No saved knowledge base found.")

# Utility Functions
class QueryRequest(BaseModel):
    query: str

def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def split_text_by_headings(text):
    heading_pattern = re.compile(r"^[A-Z][A-Za-z0-9\s\-]+:")
    chunks, current_chunk = [], []
    for line in text.split('\n'):
        if heading_pattern.match(line):
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            current_chunk = [line]
        else:
            current_chunk.append(line)
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks

def get_embeddings_batch(texts):
    return embedding_model.encode(texts, show_progress_bar=True, batch_size=16)

def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def load_embeddings_and_chunks():
    embeddings_list, text_chunks = [], []
    for file in os.listdir(EMBEDDING_DIR):
        if file.endswith(".npy"):
            embedding = np.load(os.path.join(EMBEDDING_DIR, file))
            text_file = file.replace(".npy", "_chunks.pkl")
            try:
                with open(os.path.join(TEXT_CHUNKS_DIR, text_file), "rb") as f:
                    texts = pickle.load(f)
                    embeddings_list.append(embedding)
                    text_chunks.extend(texts)
            except Exception as e:
                print(f"Failed to load {file}: {e}")
    return np.vstack(embeddings_list), text_chunks

def search_faiss_index(query, faiss_index, text_chunks, k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(np.array(query_embedding).astype(np.float32), k)
    return [text_chunks[i] for i in indices[0]]

def create_prompt(chunks, query):
    context = "\n".join(chunks)
    prompt = ChatPromptTemplate.from_template(
        """
        You are a Financial Expert Bot. Your task is to provide accurate and concise answers to the user's questions based on the provided context.
        Instructions:
        You should not provide any disclaimers or unnecessary information. Just answer the question based on the context.  
        You should not say "I don't know" or "I am not sure". If the answer is not in the context, you should say "The answer is not in the context".
        You should not say "I am a language model" or "I am an AI". Just answer the question based on the context.
        From the augmented context, extract the relevant information based on user query and provide a concise answer.
        You should not provide any additional information or opinions outside of the context given.`
        You should not provide any disclaimers or unnecessary information. Just answer the question based on the context.

        <context>
        {context}
        </context>

        Question: {input}
        """
    )
    return prompt.format(context=context, input=query)

# Routes
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        text = extract_text_from_pdf(file_bytes)
        chunks = split_text_by_headings(text)
        all_chunks = []
        for chunk in chunks:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            all_chunks.extend(text_splitter.split_text(chunk))
        embeddings = get_embeddings_batch(all_chunks)

        save_name = file.filename.replace(".pdf", "")
        np.save(os.path.join(EMBEDDING_DIR, f"{save_name}.npy"), np.array(embeddings, dtype=np.float32))
        with open(os.path.join(TEXT_CHUNKS_DIR, f"{save_name}_chunks.pkl"), "wb") as f:
            pickle.dump(all_chunks, f)

        # Save unified index and data for persistent query access
        np.save(EMBEDDINGS_PATH, np.array(embeddings, dtype=np.float32))
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(all_chunks, f)
        index = create_faiss_index(np.array(embeddings, dtype=np.float32))
        faiss.write_index(index, FAISS_INDEX_PATH)

        return {"message": "PDF processed and embeddings stored successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def ask_question(request: QueryRequest):
    try:
        global faiss_index, all_chunks
        if faiss_index is None or not all_chunks:
            return {"answer": "No knowledge base loaded. Please upload some PDFs first."}

        query_embedding = embedding_model.encode([request.query])
        distances, indices = faiss_index.search(np.array(query_embedding).astype(np.float32), k=5)
        top_chunks = [all_chunks[i] for i in indices[0] if i < len(all_chunks)]
        prompt = create_prompt(top_chunks, request.query)
        response = llm.invoke(prompt)
        return {"answer": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# NEW ENDPOINT TO LIST ALL EMBEDDED FILES
@app.get("/list-files")
async def list_uploaded_files():
    try:
        files = [f.replace(".npy", ".pdf") for f in os.listdir(EMBEDDING_DIR) if f.endswith(".npy")]
        return JSONResponse(content={"uploaded_files": files})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8002, reload=True)