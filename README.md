# 🧠 BrainWave RAG Assistant

An intelligent **Retrieval-Augmented Generation (RAG)** assistant built with **Streamlit + LangChain + FAISS** that lets you upload files or URLs and **ask questions from their content using OpenAI's GPT**.

> "Give your documents a brain — Ask questions, get instant answers!"

---

🔗 **[🚀 Live Demo Here](https://brainwaverag.streamlit.app/)** (no setup needed!)

---

## 🔍 Features

- 🔐 **Secure API key input** (local-only usage)
- 📂 **Upload multiple documents** (PDF, TXT, DOCX, CSV, JSON, code files)
- 🌐 **URL support** – paste web page links and extract content
- ✂️ **Smart Text Chunking** using `RecursiveCharacterTextSplitter`
- 🧠 **Embeddings** generated via `OpenAIEmbeddings`
- 📚 **FAISS Vector Store** for efficient retrieval
- 🤖 **Chat with your data** using `RetrievalQA` & GPT
- 💡 Auto-suggests 5 smart questions after processing
- 📋 One-click answer copy button

---

## 🚀 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/Manishkumarsingh41/brainwave-rag-app.git
cd brainwave-rag-app
```

### 2. Install dependencies

It's recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

### OR Just Use the Web Version  
🔗 **[Click here for the live app →](https://brainwaverag.streamlit.app/)**

---

## 🧪 Supported File Types

- `.pdf`, `.docx`, `.txt`
- `.json`, `.csv`
- `.py`, `.js`, `.java`, `.cpp`, `.c`, `.html`, `.css`

---

## 📷 UI Preview

> _Coming Soon: GIF or screenshots of the app in action_

---

## 🛠 Built With

- [LangChain](https://www.langchain.com/)
- [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [OpenAI API](https://platform.openai.com/)

---

## 🙋‍♂️ About Me

**Manish Kumar Singh**  
📍 B.E. in Artificial Intelligence & Data Science (RNSIT, Bengaluru)  
🔗 [Portfolio](https://iammanishsinghrajput.netlify.app/) • [GitHub](https://github.com/Manishkumarsingh41) • [LinkedIn](https://linkedin.com/in/manish-kumar-singh-5a8162214/)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

⭐️ _Found this useful? Star the repo and share your feedback!_
