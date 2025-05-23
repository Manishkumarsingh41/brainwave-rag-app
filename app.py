import streamlit as st
from langchain.document_loaders import (
    PyPDFLoader,
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

# --- App Config ---
st.set_page_config(page_title="BrainWave RAG", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>🧠 BrainWave RAG Assistant</h1>
    <p style='text-align: center; color: grey;'>Upload documents or code files and ask smart questions</p>
    """,
    unsafe_allow_html=True,
)

# --- API Key from Secrets ---
user_openai_key = st.secrets["OPENAI_API_KEY"]

# --- Sidebar File Upload ---
st.sidebar.header("📂 Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload documents or code files:",
    type=["pdf", "docx", "txt", "json", "csv", "py", "js", "java", "cpp", "c", "html", "css"],
    accept_multiple_files=True,
)
process_button = st.sidebar.button("🚀 Upload & Process")

# --- Session State ---
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
            elif suffix in ["docx", "txt", "py", "js", "java", "cpp", "c", "html", "css"]:
                loader = TextLoader(tmp_path)
            elif suffix == "csv":
                loader = CSVLoader(tmp_path)
            elif suffix == "json":
                loader = JSONLoader(tmp_path)
            else:
                st.warning(f"Unsupported file type: {suffix}")
                continue

            data = loader.load()
            for doc in data:
                st.session_state.all_text += doc.page_content + "\n"
            loaders.append(data)

        except Exception as e:
            st.error(f"Error loading {file.name}: {str(e)}")

    # Split, Embed, Store
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

# --- Show Suggested Questions ---
if "suggested_questions" in st.session_state:
    st.markdown("### 💡 Suggested Questions")
    st.markdown(st.session_state.suggested_questions)

# --- Q&A Interface ---
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
    st.info("⬅️ Upload documents and click 'Upload & Process' to begin.")
