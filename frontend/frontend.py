import streamlit as st
import requests

BACKEND_URL = "http://localhost:8002"

st.set_page_config(page_title="Fixed Income Expert", layout="centered")
st.title("📘 Fixed Income Expert Chatbot")

tab1, tab2, tab3 = st.tabs(["📤 Add Knowledge", "💬 Ask Questions", "📄 View Uploaded Files"])

# Tab 1: Upload PDFs and Add Knowledge
with tab1:
    st.header("Upload Fixed Income PDFs")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_file:
        if st.button("Upload and Process"):
            with st.spinner("Processing PDF..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                res = requests.post(f"{BACKEND_URL}/upload", files=files)
                if res.status_code == 200:
                    st.success(res.json()["message"])
                else:
                    st.error(f"Error: {res.text}")

# Tab 2: Ask Questions
with tab2:
    st.header("Ask a Question")
    query = st.text_input("Type your question:")
    if st.button("Get Answer") and query:
        with st.spinner("Thinking..."):
            res = requests.post(f"{BACKEND_URL}/query", json={"query": query})
            if res.status_code == 200:
                st.success(res.json()["answer"])
            else:
                st.error(f"Error: {res.text}")

# Tab 3: View Uploaded Files and Metadata
with tab3:
    st.header("Uploaded Files and Metadata")
    try:
        res = requests.get(f"{BACKEND_URL}/list-files")
        if res.status_code == 200:
            uploaded_files = res.json().get("uploaded_files", [])
            if uploaded_files:
                for file in uploaded_files:
                    st.subheader(f"📄 {file['filename']}")
                    st.write(f"Upload Time: {file['upload_time']}")
                    st.write(f"Number of Chunks: {file['num_chunks']}")
                    st.write("---")
            else:
                st.info("No files uploaded yet.")
        else:
            st.error(f"Error: {res.text}")
    except Exception as e:
        st.error(f"Failed to retrieve files: {str(e)}")
