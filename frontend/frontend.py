import os
import streamlit as st
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_URL = "http://localhost:8005"

# Local Ollama models
LOCAL_MODELS = {
    "wizardlm2": "wizardlm2:7b",
    "llama3.3": "llama3.3:70b",
    "deepseek-r1-8b": "deepseek-r1:8b",
    "deepseek-r1-1.5b": "deepseek-r1:1.5b"
}

st.set_page_config(page_title="FactEntry Fixed Income Expert", layout="wide")
st.title("📄 FactEntry Fixed Income Expert")

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if "model_choice" not in st.session_state:
    st.session_state.model_choice = {}

# Tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload PDF", "❓ Ask Expert", "🔍 Semantic Search"])

# --- 1. Upload PDF Tab ---
with tab1:
    st.header("Upload and Index PDF")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**Size:** {len(uploaded_file.getvalue()) / 1024:.2f} KB")

        if st.button("Upload and Index PDF"):
            with st.status("📥 Uploading and processing PDF...", expanded=True) as status:
                try:
                    res = requests.post(
                        f"{API_URL}/upload_pdf",
                        files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                        timeout=120
                    )
                    if res.status_code == 200:
                        status.update(label="✅ PDF indexed successfully!", state="complete")
                    else:
                        status.update(label=f"❌ {res.status_code}: {res.text}", state="error")
                except Exception as e:
                    status.update(label=f"❌ Request failed: {e}", state="error")

# --- 2. Ask Question Tab ---
with tab2:
    st.header("Ask a Question")
    question = st.text_area("Enter your question:")

    model_type = st.radio("Choose Model Type:", ["Local (Ollama)", "Remote (Groq API)"])
    headers = {}

    if model_type == "Local (Ollama)":
        selected = st.selectbox("Select Local Model:", list(LOCAL_MODELS.keys()), format_func=lambda x: f"{x} ({LOCAL_MODELS[x]})")
        st.session_state.model_choice = {"type": "ollama", "model_name": LOCAL_MODELS[selected]}
    else:
        if not GROQ_API_KEY:
            st.error("🚫 GROQ_API_KEY missing. Set it in your `.env` file.")
        else:
            headers["Authorization"] = f"Bearer {GROQ_API_KEY}"
            st.session_state.model_choice = {"type": "groq", "model_name": "llama3-8b-8192"}

    if st.button("Ask"):
        if not question.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            with st.spinner("Getting answer..."):
                try:
                    res = requests.post(
                        f"{API_URL}/ask_question",
                        headers=headers,
                        json={"question": question, "model_choice": st.session_state.model_choice},
                        timeout=60
                    )
                    if res.status_code == 200:
                        data = res.json()
                        answer = data.get("answer", "No answer.")
                        st.subheader("📬 Answer")
                        st.success(answer)
                        st.caption(f"Model Used: `{st.session_state.model_choice['model_name']}`")

                        st.session_state.qa_history.append((question, answer))

                        if data.get("sources"):
                            st.markdown("#### 📚 Sources:")
                            for src in set(data["sources"]):
                                st.write(f"- {src}")
                    else:
                        st.error(f"❌ {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"❌ Request failed: {e}")

    # History
    if st.session_state.qa_history:
        with st.expander("🕑 Q&A History"):
            for q, a in st.session_state.qa_history:
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
                st.markdown("---")

# --- 3. Semantic Search Tab ---
with tab3:
    st.header("Semantic Chunk Search")
    query = st.text_input("Enter phrase to search:")
    top_k = st.slider("Top K results", 1, 10, 5)

    if st.button("Search"):
        if not query.strip():
            st.warning("⚠️ Please enter a query.")
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
                            st.markdown("### 🔍 Top Chunks:")
                            for r in results:
                                st.markdown(f"**Score:** `{r['score']:.4f}`")
                                st.write(r["text"])
                                st.markdown("---")
                        else:
                            st.info("ℹ️ No matches found.")
                    else:
                        st.error(f"❌ {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"❌ Request failed: {e}")
