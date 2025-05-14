import json
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import os
import numpy as np
import faiss
import pickle
import re
from typing import List
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroqF
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import subprocess  # For running Ollama models
import glob

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise HTTPException(status_code=500, detail="GROQ_API_KEY not found in environment variables")

# Directories
EMBEDDING_DIR = "/home/harish/Project_works/FactEntry_FEexpert/backend/embeddings"
TEXT_CHUNKS_DIR = "/home/harish/Project_works/FactEntry_FEexpert/backend/text_chunks"
METADATA_DIR = "/home/harish/Project_works/FactEntry_FEexpert/backend/metadata"
os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(TEXT_CHUNKS_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# Load models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192", temperature=0)

# Ollama model settings
Ollama_model_dict = {
    "deepseek": "deepseek-r1:8b", 
    "llama": "llama3.3:70b"
}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility Functions
class QueryRequest(BaseModel):
    query: str
    model: str = "groq"  # default to groq

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

def search_faiss_index(query, faiss_index, text_chunks, k=20):
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(np.array(query_embedding).astype(np.float32), k)
    return [text_chunks[i] for i in indices[0]]

def create_prompt(chunks, query):
    context = "\n".join(chunks)
    prompt = ChatPromptTemplate.from_template(
        """
        You are a Financial Expert Bot. Your task is to provide accurate and Detailed answers to the user's questions based on the provided context.
        Instructions:
        You should not provide any disclaimers or unnecessary information. Just answer the question based on the context.  
        You should not say "I don't know" or "I am not sure". If the answer is not in the context, you should say "The answer is not in the context".
        You should not say "I am a language model" or "I am an AI". Just answer the question based on the context.
        From the augmented context, extract the relevant information based on user query and provide a concise answer.
        <context>
        {context}
        </context>
        Question: {input}
        """
    )
    return prompt.format(context=context, input=query)

# Save metadata function
def save_metadata(filename, num_chunks):
    metadata = {
        "filename": filename,
        "upload_time": datetime.now().isoformat(),
        "num_chunks": num_chunks,
    }
    with open(os.path.join(METADATA_DIR, f"{filename}.json"), "w") as f:
        json.dump(metadata, f)

# FAISS index optimization: Load index once
def load_faiss_index():
    faiss_index_path = os.path.join(METADATA_DIR, "faiss_index.index")
    if os.path.exists(faiss_index_path):
        return faiss.read_index(faiss_index_path)
    else:
        embeddings, text_chunks = load_embeddings_and_chunks()
        faiss_index = create_faiss_index(embeddings)
        faiss.write_index(faiss_index, faiss_index_path)
        return faiss_index

faiss_index = load_faiss_index()

# Routes
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        text = extract_text_from_pdf(file_bytes)
        chunks = split_text_by_headings(text)
        all_chunks = []
        for chunk in chunks:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            all_chunks.extend(text_splitter.split_text(chunk))
        embeddings = get_embeddings_batch(all_chunks)

        save_name = file.filename.replace(".pdf", "")
        np.save(os.path.join(EMBEDDING_DIR, f"{save_name}.npy"), np.array(embeddings, dtype=np.float32))
        with open(os.path.join(TEXT_CHUNKS_DIR, f"{save_name}_chunks.pkl"), "wb") as f:
            pickle.dump(all_chunks, f)

        # Save metadata
        save_metadata(save_name, len(all_chunks))

        return {"message": "PDF processed and embeddings stored successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def ask_question(request: QueryRequest):
    try:
        # Load all embeddings and chunks
        embedding_files = sorted(glob.glob(os.path.join(EMBEDDING_DIR, "*.npy")))
        chunk_files = sorted(glob.glob(os.path.join(TEXT_CHUNKS_DIR, "*.pkl")))

        if not embedding_files or not chunk_files:
            raise HTTPException(status_code=400, detail="No knowledge base available. Please upload PDFs first.")

        all_chunks = []
        all_embeddings = []

        for efile, cfile in zip(embedding_files, chunk_files):
            all_embeddings.append(np.load(efile))
            all_chunks.extend(pickle.load(open(cfile, "rb")))

        all_embeddings = np.vstack(all_embeddings)

        # Create a temporary FAISS index
        dim = all_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(all_embeddings)

        # Embed the query and find top-k similar chunks
        query_embedding = embedding_model.encode([request.query], normalize_embeddings=True)
        scores, indices = index.search(query_embedding, k=20)
        relevant_chunks = [all_chunks[i] for i in indices[0]]

        # Build prompt from context
        prompt = create_prompt(relevant_chunks, request.query)

        # Use Groq or Ollama
        if request.model == "groq":
            try:
                response = llm.invoke(prompt)
                answer = response.content
            except Exception as e:
                raise HTTPException(status_code=500, detail="Error calling Groq API: " + str(e))
        else:
            model_name = Ollama_model_dict.get(request.model, "deepseek-r1:8b")
            try:
                result = subprocess.run(
                    ["ollama", "run", model_name, prompt],
                    capture_output=True, text=True, timeout=60
                )
                if result.returncode != 0:
                    raise HTTPException(status_code=500, detail=f"Ollama command failed: {result.stderr}")
                answer = result.stdout.strip()
            except subprocess.TimeoutExpired:
                raise HTTPException(status_code=500, detail="Ollama model execution timed out")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Ollama execution error: {str(e)}")

        return {"answer": answer, "source": relevant_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/view-uploads")
async def view_uploaded_files():
    try:
        files = []
        # Check if the directory exists and iterate through the files
        for file in os.listdir(EMBEDDING_DIR):
            if file.endswith(".npy"):
                file_name = file.replace(".npy", ".pdf")
                metadata_path = os.path.join(METADATA_DIR, f"{file_name.replace('.pdf', '')}.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    files.append(metadata)

        if not files:
            raise HTTPException(status_code=404, detail="No uploaded files found.")
        
        return {"uploaded_files": files}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-chunks")
def search_chunks(query: str = Body(...), top_k: int = Body(...)):
    try:
        # Load embeddings and text chunks
        embeddings, text_chunks = load_embeddings_and_chunks()
        
        # Load or create the FAISS index
        faiss_index_path = os.path.join(METADATA_DIR, "faiss_index.index")
        if os.path.exists(faiss_index_path):
            faiss_index = faiss.read_index(faiss_index_path)
        else:
            faiss_index = create_faiss_index(embeddings)
            faiss.write_index(faiss_index, faiss_index_path)

        # Embed the query
        query_embedding = embedding_model.encode([query])
        
        # Search the FAISS index
        distances, indices = faiss_index.search(np.array(query_embedding).astype(np.float32), top_k)
        
        # Collect results
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue  # Skip invalid index
            result = {
                "text": text_chunks[idx],
                "score": float(score)
            }
            results.append(result)
        
        if not results:
            raise HTTPException(status_code=404, detail="No relevant chunks found.")
        
        return {"results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8002, reload=True)
