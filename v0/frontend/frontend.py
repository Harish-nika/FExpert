import streamlit as st
import requests
import json

# Backend URL
BACKEND_URL = "http://localhost:8002"

st.set_page_config(page_title="FExpert - FactEntry BOT", layout="wide")
st.title("📊 FactEntry Fixed Income Expert")

# Sidebar for navigation
page = st.sidebar.radio("Navigation", ["Upload PDF", "Ask a Question", "Uploaded Files", "Search Content"])

# Upload PDF
if page == "Upload PDF":
    st.header("📤 Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file and st.button("Upload"):
        with st.spinner("Uploading and processing..."):
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            response = requests.post(f"{BACKEND_URL}/upload", files=files)
            if response.status_code == 200:
                st.success("✅ PDF uploaded and processed!")
            else:
                st.error(f"❌ Upload failed: {response.json()['detail']}")

# Ask a Question
elif page == "Ask a Question":
    st.header("🧠 Ask a Question")
    query = st.text_area("Enter your question:", height=100)
    model = st.selectbox("Choose a model:", ["groq", "deepseek", "llama"])

    if st.button("Get Answer"):
        if not query:
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Fetching answer..."):
                payload = {"query": query, "model": model}
                response = requests.post(f"{BACKEND_URL}/query", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    st.subheader("📌 Answer")
                    st.success(result["answer"])
                    with st.expander("📄 Source Chunks"):
                        for i, chunk in enumerate(result["source"]):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.code(chunk)
                else:
                    st.error(f"❌ Error: {response.json()['detail']}")

# Search Content
elif page == "Search Content":
    st.header("🔍 Search Knowledge Base")
    user_query = st.text_input("Enter a search query:")
    top_k = st.number_input("How many top chunks to retrieve?", min_value=1, max_value=20, value=5)

    # Only proceed if the search button is clicked and there is a query
    if user_query:
        if st.button("Search Embeddings"):
            with st.spinner("Searching FAISS index..."):
                try:
                    # Send the search query and top_k value to the backend API
                    res = requests.post(
                        f"{BACKEND_URL}/search-chunks",
                        json={"query": user_query, "top_k": top_k}
                    )

                    # Handle the response from the server
                    if res.status_code == 200:
                        chunks = res.json().get("results", [])
                        
                        # Check if any chunks are returned
                        if chunks:
                            for i, chunk in enumerate(chunks):
                                st.subheader(f"🔹 Chunk #{i+1}")
                                st.markdown(f"**Text:** {chunk['text']}")
                                st.markdown(f"**Score:** {chunk['score']:.4f}")
                                st.write("---")
                        else:
                            st.info("No chunks matched your query.")
                    
                    # Handle any error responses from the backend
                    elif res.status_code != 200:
                        st.error(f"Error: {res.json().get('detail', res.text)}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Request failed: {str(e)}")

                except Exception as e:
                    st.error(f"Failed to retrieve search results: {str(e)}")
    
    # If no query is entered, display a prompt to enter one
    elif st.button("Search Embeddings"):
        st.warning("Please enter a search query first.")

# View Uploaded Files
elif page == "Uploaded Files":
    st.header("📁 Uploaded Files")
    with st.spinner("Loading uploaded files..."):
        response = requests.get(f"{BACKEND_URL}/view-uploads")
        if response.status_code == 200:
            files = response.json()["uploaded_files"]
            for file in files:
                st.markdown(f"📄 **{file['filename']}**")
                st.write(f"Uploaded at: {file['upload_time']}")
                st.write(f"Chunks: {file['num_chunks']}")
                st.markdown("---")
        else:
            st.warning(response.json()["detail"])
