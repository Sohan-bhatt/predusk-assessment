# 🧠 Mini RAG Application

A **Retrieval-Augmented Generation (RAG)** demo built with **Streamlit**, **Pinecone**, **Cohere**, and **OpenAI**.  
Upload a document, store its embeddings in Pinecone, and ask questions with sources cited.

---

## 🚀 Features
- **Document Ingestion**  
  - Upload `.txt` files  
  - Split into chunks (`1000` chars, `150` overlap)  
  - Embed with `text-embedding-3-small`  
  - Store in Pinecone with metadata

- **Query Flow**  
  - Semantic search with Pinecone (`top_k` adjustable)  
  - Cohere Rerank (`rerank-english-v3.0`)  
  - LLM response via `gpt-4o` with [n]-style citations

- **UI / UX**  
  - Modern Streamlit layout (cards, chips, gradient header)  
  - Sidebar controls (status, sliders, clear history)  
  - Chat-like Q&A history with timestamps  
  - Debug tab (shows full prompt + context)  

---

## 📂 Project Structure

```
project-root/
│
├─ backend/
│ ├─ main.py # Streamlit app entrypoint
│ └─ a.py  # pinecone index name initializing
│ └─ requirements.txt
├─ .streamlit/
│ └─ config.toml # UI theme (colors, fonts)
│
└─ README.md
```



---

## ⚙️ Setup

1. **Clone repo & install deps**
   ```bash
   git clone https://github.com/Sohan-bhatt/predusk-assessment.git
   cd backened
   Create a virt env.
   pip install -r requirements.txt
## Add environment variables
  Create a .env file in project root with:
  
  - PINECONE_API_KEY=your_pinecone_key
  - OPENAI_API_KEY=your_openai_key
  - COHERE_API_KEY=your_cohere_key
  - PINECONE_INDEX_NAME=your_index_name

## (Optional) Configure theme
    Inside .streamlit/config.toml:
    
    [theme]
    primaryColor="#7c3aed"
    backgroundColor="#0b0f14"
    secondaryBackgroundColor="#121922"
    textColor="#e6edf3"
    font="sans serif"



   ▶️ Run the App

  ## From project root:
  
  - #### streamlit run backend/main.py
