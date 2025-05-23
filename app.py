# app.py
import streamlit as st
from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredURLLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    CSVLoader,
    JSONLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import tempfile
import os
import base64

# --- App Title and Config ---
st.set_page_config(page_title="BrainWave RAG", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>🧠 BrainWave RAG Assistant</h1>
    <p style='text-align: center; color: grey;'>Upload documents, URLs, or code files and ask smart questions</p>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar: User OpenAI API Key input ---
user_openai_key = st.sidebar.text_input(
    "🔑 Enter your OpenAI API Key",
    type="password",
    placeholder="sk-...",
    help="Your API key is only used locally and not stored",
)

if not user_openai_key:
    st.sidebar.warning("Please enter your OpenAI API Key to use this app.")
    st.stop()

# --- Sidebar Input ---
st.sidebar.header("📂 Upload or Paste URLs")

uploaded_files = st.sidebar.file_uploader(
    "Upload documents or code files:",
    type=[
        "pdf",
        "docx",
        "txt",
        "json",
        "csv",
        "py",
        "js",
        "java",
        "cpp",
        "c",
        "html",
        "css",
        "pptx",
        "xlsx",
    ],
    accept_multiple_files=True,
)

url_input = st.sidebar.text_area("Paste URLs (one per line):")

process_button = st.sidebar.button("🚀 Upload & Process")

# --- Session State Initialization ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "all_text" not in st.session_state:
    st.session_state.all_text = ""

loaders = []

if process_button:
    st.session_state.all_text = ""

    for file in uploaded_files:
        suffix = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            if suffix == "pdf":
                loader = PyPDFLoader(tmp_path)
            elif suffix == "docx":
                loader = UnstructuredWordDocumentLoader(tmp_path)
            elif suffix == "txt":
                loader = TextLoader(tmp_path)
            elif suffix == "csv":
                loader = CSVLoader(tmp_path)
            elif suffix == "json":
                loader = JSONLoader(tmp_path)
            elif suffix in ["py", "js", "java", "cpp", "c", "html", "css"]:
                loader = TextLoader(tmp_path)
            elif suffix in ["pptx", "xlsx"]:
                loader = TextLoader(tmp_path)
            else:
                st.warning(f"Unsupported file type: {suffix}")
                continue

            data = loader.load()
            for doc in data:
                st.session_state.all_text += doc.page_content + "\n"
            loaders.append(data)

        except Exception as e:
            st.error(f"Error loading {file.name}: {str(e)}")

    urls = [url.strip() for url in url_input.splitlines() if url.strip() != ""]
    if urls:
        url_loader = UnstructuredURLLoader(urls=urls)
        url_docs = url_loader.load()
        for doc in url_docs:
            st.session_state.all_text += doc.page_content + "\n"
        loaders.append(url_docs)

    # Split documents, embed, and build vector store using user's API key
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    documents = [doc for docs in loaders for doc in docs]
    split_docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=user_openai_key)
    vectordb = FAISS.from_documents(split_docs, embeddings)
    st.session_state.vector_store = vectordb
    st.success("✅ Documents processed and indexed!")

    with st.spinner("🤖 Generating suggested questions..."):
        llm = ChatOpenAI(openai_api_key=user_openai_key, temperature=0)
        prompt = f"""
You are an intelligent assistant. Based on the content below, suggest 5 helpful questions a user might ask:

Content:
{st.session_state.all_text[:5000]}

Questions:
1.
2.
3.
4.
5.
"""
        response = llm.predict(prompt)
        st.session_state.suggested_questions = response

if "suggested_questions" in st.session_state:
    st.markdown("### 💡 Suggested Questions")
    st.markdown(st.session_state.suggested_questions)

if st.session_state.vector_store:
    st.markdown("---")
    st.markdown("### 💬 Ask a Question")

    col1, col2 = st.columns([4, 1])
    with col1:
        user_query = st.text_input("Type your question:", key="query_input")
    with col2:
        send_button = st.button("📨 Send", key="send_btn")

    if send_button and user_query:
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(openai_api_key=user_openai_key, temperature=0),
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever(),
        )
        answer = qa.run(user_query)
        st.markdown(f"**Answer:**\n{answer}")

        encoded = base64.b64encode(answer.encode()).decode()
        copy_code = f"""
            <button onclick="navigator.clipboard.writeText(atob('{encoded}'))"
                    style="padding:6px 12px;margin-top:10px;border:none;background:#2196F3;color:white;border-radius:6px;cursor:pointer;">
                📋 Copy Answer
            </button>
        """
        st.markdown(copy_code, unsafe_allow_html=True)
else:
    st.info("⬅️ Upload documents or enter URLs and click 'Upload & Process' to begin.")
