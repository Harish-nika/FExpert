import os
import streamlit as st
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8007")  # Fallback to localhost if not set
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LOCAL_MODELS = {
    "wizardlm2": "wizardlm2:7b",
    "llama3.3": "llama3.3:70b",
    "deepseek-r1-8b": "deepseek-r1:8b",
    "deepseek-r1-1.5b": "deepseek-r1:1.5b"
}

st.set_page_config(page_title="FactEntry Fixed Income Expert", layout="wide")
st.title("📄 FactEntry Fixed Income Expert")

# Session State
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "file_name" not in st.session_state:
    st.session_state.file_name = None

# --- Tabs Layout ---
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload PDF", "❓ Ask Question", "🔍 Semantic Search", "📁 Uploaded Files"])

# --- 1. Upload PDF ---
with tab1:
    st.header("Upload and Index a PDF")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file:
        file_size_kb = len(uploaded_file.getvalue()) / 1024
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**Size:** {file_size_kb:.2f} KB")

        if st.button("Upload and Index PDF"):
            with st.spinner("Uploading and indexing..."):
                try:
                    res = requests.post(
                        f"{API_URL}/upload_pdf",
                        files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                        timeout=120
                    )
                    if res.status_code == 200:
                        st.session_state.file_name = res.json().get("file_name", "")
                        st.success("✅ PDF indexed successfully!")
                    else:
                        st.error(f"❌ {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"❌ Request failed: {e}")

    if st.button("Reset Selected PDF"):
        st.session_state.file_name = None
        st.success("Selection cleared.")

# --- 2. Ask a Question ---
with tab2:
    st.header("Ask a Question")
    question = st.text_area("Enter your question:")

    model_type = st.radio("Choose Model Type:", ["Local (Ollama)", "Remote (Groq API)"])
    headers = {}
    model_choice = {}

    if model_type == "Local (Ollama)":
        selected = st.selectbox("Select Local Model:", list(LOCAL_MODELS.keys()))
        model_choice = {"type": "ollama", "model_name": LOCAL_MODELS[selected]}
    else:
        if not GROQ_API_KEY:
            st.error("🚫 GROQ_API_KEY missing. Set it in your `.env` file.")
        else:
            headers["Authorization"] = f"Bearer {GROQ_API_KEY}"
            model_choice = {"type": "groq", "model_name": "llama3-8b-8192"}

    if st.button("Ask"):
        if not question.strip():
            st.warning("⚠️ Please enter a question.")
        elif not st.session_state.file_name:
            st.warning("⚠️ Please upload a PDF first.")
        else:
            with st.spinner("Getting answer..."):
                try:
                    res = requests.post(
                        f"{API_URL}/ask_question",
                        headers=headers,
                        json={
                            "question": question,
                            "model_choice": model_choice,
                            "file_name": st.session_state.file_name
                        },
                        timeout=120
                    )
                    if res.status_code == 200:
                        data = res.json()
                        answer = data.get("answer", "No answer.")
                        st.subheader("📬 Answer")
                        st.success(answer)
                        st.caption(f"Model Used: `{model_choice['model_name']}`")

                        st.session_state.qa_history.append((question, answer))

                        if data.get("sources"):
                            st.markdown("#### 📚 Sources:")
                            for src in set(data["sources"]):
                                st.write(f"- {src}")
                    else:
                        st.error(f"❌ {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"❌ Request failed: {e}")

    if st.session_state.qa_history:
        with st.expander("🕑 Q&A History"):
            for q, a in st.session_state.qa_history:
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
                st.markdown("---")

# --- 3. Semantic Search ---
with tab3:
    st.header("Semantic Chunk Search")
    query = st.text_input("Enter phrase to search:")
    top_k = st.slider("Top K results", 1, 10, 5)

    if st.button("Search"):
        if not query.strip():
            st.warning("⚠️ Please enter a query.")
        elif not st.session_state.file_name:
            st.warning("⚠️ Please upload a PDF first.")
        else:
            with st.spinner("Searching chunks..."):
                try:
                    res = requests.post(
                        f"{API_URL}/search-chunks",
                        json={"query": query, "top_k": top_k, "file_name": st.session_state.file_name},
                        timeout=60
                    )
                    if res.status_code == 200:
                        results = res.json().get("results", [])
                        if results:
                            st.markdown("### 🔍 Top Matching Chunks")
                            for r in results:
                                st.markdown(f"**Score:** `{r['score']:.4f}`")
                                st.write(r["text"])
                                st.markdown("---")
                        else:
                            st.info("ℹ️ No matching chunks found.")
                    else:
                        st.error(f"❌ {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"❌ Request failed: {e}")

# --- 4. List Uploaded PDFs ---
with tab4:
    st.header("Uploaded PDFs")

    try:
        response = requests.get(f"{API_URL}/list_uploaded_pdfs", timeout=30)
        if response.status_code == 200:
            pdfs = response.json().get("pdfs", [])
            if pdfs:
                for pdf in pdfs:
                    st.markdown(f"**• {pdf['name']}** - {pdf.get('upload_date', 'N/A')}")
            else:
                st.info("No PDFs uploaded yet.")
        else:
            st.warning("Couldn't fetch uploaded files.")
    except Exception as e:
        st.warning(f"Error: {e}")
