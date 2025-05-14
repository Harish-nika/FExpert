import os
import streamlit as st
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

API_URL = "http://localhost:8002"

# List of available local Ollama models
LOCAL_MODELS = {
    "wizardlm2": "wizardlm2:7b",
    "llama3.3": "llama3.3:70b",
    "deepseek-r1-8b": "deepseek-r1:8b",
    "deepseek-r1-1.5b": "deepseek-r1:1.5b"
}

st.set_page_config(page_title="FactEntry Fixed Income Expert", layout="wide")
st.title("📄 FactEntry Fixed Income Expert")

# Initialize chat history
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# --- Upload PDF ---
st.header("1. Upload PDF")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    file_size_kb = len(uploaded_file.getvalue()) / 1024
    st.write(f"**Filename:** {uploaded_file.name}")
    st.write(f"**File Size:** {file_size_kb:.2f} KB")

    if st.button("Upload and Index PDF"):
        with st.spinner("Uploading and indexing PDF..."):
            try:
                response = requests.post(
                    f"{API_URL}/upload_pdf",
                    files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                    timeout=60
                )
                if response.status_code == 200:
                    st.success("✅ PDF indexed successfully!")
                else:
                    st.error(f"❌ Error: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Request failed: {e}")

# --- Ask Question ---
st.header("2. Ask a Question (LLM Q&A)")
question = st.text_area("Enter your question:")

model_type = st.selectbox("Choose Model Type:", ["Local (Ollama)", "Remote (Groq API)"])

headers = {}
model_choice = {}

if model_type == "Local (Ollama)":
    selected_local_model = st.selectbox("Select Local Model (Ollama):", list(LOCAL_MODELS.keys()))
    selected_model = LOCAL_MODELS[selected_local_model]
    model_choice = {"type": "ollama", "model_name": selected_model}
else:
    model_choice = {"type": "groq", "model_name": "llama3-8b-8192"}
    if not GROQ_API_KEY:
        st.error("🚫 GROQ_API_KEY not found. Please set it in your .env file.")
    else:
        headers["Authorization"] = f"Bearer {GROQ_API_KEY}"

if st.button("Ask"):
    if question.strip() == "":
        st.warning("⚠️ Please enter a question first.")
    else:
        with st.spinner("Fetching answer..."):
            try:
                res = requests.post(
                    f"{API_URL}/ask_question/",
                    headers=headers,
                    json={
                        "question": question,
                        "model_choice": model_choice
                    },
                    timeout=60
                )
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "No answer returned.")
                    st.subheader("📬 Answer:")
                    st.success(answer)
                    st.caption(f"Model Used: `{model_choice['model_name']}`")

                    # Update chat history
                    st.session_state.qa_history.append((question, answer))

                    if data.get("sources"):
                        st.markdown("#### 📚 Sources:")
                        for source in set(data["sources"]):
                            st.write(f"- {source}")
                else:
                    st.error(f"❌ {res.status_code}: {res.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Request failed: {e}")

# Display Q&A history
if st.session_state.qa_history:
    with st.expander("🕑 Q&A History"):
        for q, a in st.session_state.qa_history:
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")

# --- Semantic Search ---
st.header("3. Semantic Chunk Search")
query = st.text_input("Search phrase:")
top_k = st.slider("Top K results", min_value=1, max_value=10, value=5)

if st.button("Search"):
    if query.strip() == "":
        st.warning("⚠️ Please enter a search phrase first.")
    else:
        with st.spinner("Searching chunks..."):
            try:
                res = requests.post(
                    f"{API_URL}/search-chunks",
                    json={"query": query, "top_k": top_k},
                    timeout=60
                )
                if res.status_code == 200:
                    results = res.json().get("results", [])
                    if results:
                        st.markdown("### 🔍 Matching Chunks:")
                        for r in results:
                            st.markdown(f"**Score:** `{r['score']:.4f}`")
                            st.write(r["text"])
                            st.markdown("---")
                    else:
                        st.info("ℹ️ No matching chunks found.")
                else:
                    st.error(f"❌ {res.status_code}: {res.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Request failed: {e}")
