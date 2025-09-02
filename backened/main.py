import streamlit as st
import cohere
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import time
import os
from dotenv import load_dotenv
from datetime import datetime

# -----------------------------
# Page meta + base CSS styling
# -----------------------------
st.set_page_config(
    page_title="Mini RAG App",
    page_icon="🧠",
    layout="wide"
)

# --- Small CSS helpers for a modern look (assessment: safe, no external deps) ---
CUSTOM_CSS = """
<style>
/* App-wide font + spacing polish */
html, body, [class*="css"] { font-family: "Inter", system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }

/* Header gradient */
.gradient-header {
    background: linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%);
    padding: 18px 22px; border-radius: 16px; color: white; margin-bottom: 12px;
}
.gradient-header h1 { margin: 0; font-size: 1.6rem; line-height: 1.3; }
.gradient-header p { margin: 6px 0 0 0; opacity: 0.95; }

/* Card styling for results/sections */
.card {
    border-radius: 16px; padding: 18px; border: 1px solid rgba(0,0,0,0.08);
    background: rgba(255,255,255,0.65); backdrop-filter: blur(6px);
}
.dark .card {
    background: rgba(17,17,17,0.6); border-color: rgba(255,255,255,0.08);
}

/* Source chips */
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 10px; border-radius: 999px; font-size: 12px;
    border: 1px solid rgba(0,0,0,0.08); background: #f8fafc;
}
.dark .chip { background: rgba(255,255,255,0.06); border-color: rgba(255,255,255,0.08); }

/* Subtle section title */
.section-title {
    font-weight: 700; letter-spacing: 0.2px; margin-bottom: 6px; font-size: 1rem;
}

/* Small mono for metrics */
.kv { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 12.5px; padding: 10px 12px; border-radius: 10px;
      background: rgba(148,163,184,0.08);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# Load secrets (.env)
# -----------------------------
load_dotenv()

try:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
    COHERE_API_KEY   = os.getenv("COHERE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

    if not all([PINECONE_API_KEY, OPENAI_API_KEY, COHERE_API_KEY, PINECONE_INDEX_NAME]):
        raise ValueError("One or more required environment variables are missing.")
except (KeyError, ValueError):
    st.error("API keys or Pinecone index name not found. Please create a .env file and add your keys.", icon="🚨")
    st.stop()

# -----------------------------
# Cache heavy clients
# -----------------------------
@st.cache_resource
def initialize_services():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
        llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
        co = cohere.Client(api_key=COHERE_API_KEY)

        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            st.error(f"Index '{PINECONE_INDEX_NAME}' does not exist. Please create it in the Pinecone console.", icon="❌")
            st.stop()
        index = pc.Index(PINECONE_INDEX_NAME)

        stats = index.describe_index_stats()
        if stats.dimension != 1536:
            st.error(f"Pinecone index '{PINECONE_INDEX_NAME}' has dimension {stats.dimension}, but model 'text-embedding-3-small' requires 1536.", icon="❌")
            st.stop()

        return embeddings, llm, co, index
    except Exception as e:
        st.error(f"Failed to initialize services: {e}", icon="🚨")
        st.stop()

embeddings, llm, co, index = initialize_services()


# ----------------------------------------------------
# --- OPTIMIZED BATCH PROCESSING FOR DOCUMENT UPLOAD ---
# ----------------------------------------------------
def upsert_text(text: str, source: str, batch_size: int = 64):
    """
    Optimized function to chunk, embed, and upsert text to Pinecone in batches.
    This is more memory-efficient and reliable for larger documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(text)
    
    total_chunks = len(chunks)
    if total_chunks == 0:
        return 0

    # Process and upsert in batches
    for i in range(0, total_chunks, batch_size):
        # Get the current batch of chunks
        batch_chunks = chunks[i:i + batch_size]
        
        # 1. Embed the current batch
        batch_embeddings = embeddings.embed_documents(batch_chunks)

        # 2. Prepare vectors for upsert
        vectors_to_upsert = []
        for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
            chunk_index = i + j
            vectors_to_upsert.append({
                "id": f"{source}-{chunk_index}",
                "values": embedding,
                "metadata": {"text": chunk, "source": source}
            })
        
        # 3. Upsert the batch to Pinecone
        index.upsert(vectors=vectors_to_upsert)
    
    return total_chunks
# ----------------------------------------------------
# --- END OF OPTIMIZED SECTION ---
# ----------------------------------------------------

def build_prompt(context: str, query: str) -> str:
    return f"""
You are a strict RAG assistant. Answer ONLY using the provided context.
- If unsure or answer not present, say you cannot answer from the provided info.
- Cite sources like [1], [2] matching the context blocks.

Context:
{context}

User Query: {query}
Answer:
""".strip()

def perform_query(query: str, top_k: int = 10, rerank_n: int = 3):
    t0 = time.time()

    q_emb = embeddings.embed_query(query)
    resp = index.query(vector=q_emb, top_k=top_k, include_metadata=True)

    docs_for_rerank = []
    original_docs = {}
    for i, match in enumerate(resp["matches"]):
        doc_text = match["metadata"]["text"]
        docs_for_rerank.append({"text": doc_text})
        original_docs[i] = {"text": doc_text, "source": match["metadata"]["source"]}

    reranked = co.rerank(query=query, documents=docs_for_rerank, top_n=rerank_n, model="rerank-english-v3.0")

    context_blocks, unique_sources, seen, selected_chunks = [], [], set(), []
    for j, result in enumerate(reranked.results):
        orig = original_docs[result.index]
        context_blocks.append(f"Source [{j+1}]: {orig['text']}")
        selected_chunks.append({"block_no": j+1, "text": orig["text"], "source": orig["source"]})
        if orig["source"] not in seen:
            unique_sources.append(orig["source"])
            seen.add(orig["source"])

    context = "\n\n".join(context_blocks)
    prompt = build_prompt(context, query)

    llm_response = llm.invoke(prompt)
    answer = llm_response.content

    t1 = time.time()
    return {
        "answer": answer, "sources": unique_sources, "selected_chunks": selected_chunks,
        "latency_sec": round(t1 - t0, 2), "top_k": top_k, "rerank_n": rerank_n,
        "prompt": prompt, "context": context
    }

# -----------------------------
# Session state
# -----------------------------
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False
if "history" not in st.session_state:
    st.session_state.history = []
if "last_stats" not in st.session_state:
    st.session_state.last_stats = None

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """<div class="gradient-header">
      <h1>🧠 Mini RAG Application</h1>
      <p>Upload a document → store chunks in Pinecone → ask questions with reranking & sources.</p>
    </div>""", unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.caption("Status & Controls")
    st.divider()
    ok = "✅" if st.session_state.doc_processed else "🟡"
    st.info(f"{ok} Document processed: **{st.session_state.doc_processed}**")
    st.write("**Search Settings**")
    top_k = st.slider("Top-K (vector search)", 5, 20, 10, help="Number of candidates from Pinecone before reranking")
    rerank_n = st.slider("Rerank N (final context)", 1, 5, 3, help="Top documents (by rerank score) fed to the LLM")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.history = []
        st.toast("Chat history cleared.", icon="🧼")
    st.caption("Tip: Use the Debug tab to show your full prompt and context.")

# -----------------------------
# Main area — Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📄 Ingest & Ask", "🧪 Debug", "ℹ️ About"])

with tab1:
    left, right = st.columns((1, 1), gap="large")
    with left:
        st.markdown("### 1) Add Document")
        with st.container(border=True):
            uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"], help="Simple text ingestion keeps the demo focused.")
            if uploaded_file:
                document_text = uploaded_file.read().decode("utf-8")
                source_name = uploaded_file.name
                st.text_area("Preview", value=document_text, height=220, disabled=True, label_visibility="collapsed")

            if st.button("🚀 Process & Store Document", type="primary", use_container_width=True):
                if 'document_text' in locals() and document_text:
                    with st.spinner(f"Processing '{source_name}'..."):
                        try:
                            chunk_count = upsert_text(document_text, source_name)
                            st.success(f"Stored **{chunk_count}** chunks from “{source_name}”.", icon="✅")
                            st.session_state.doc_processed = True
                            st.balloons()
                        except Exception as e:
                            st.error(f"An error occurred: {e}", icon="🚨")
                else:
                    st.warning("Please upload a file first.", icon="⚠️")

    with right:
        st.markdown("### 2) Ask a Question")
        with st.container(border=True):
            query = st.text_input("Ask about your document", placeholder="e.g., What are the key steps mentioned?", disabled=not st.session_state.doc_processed)
            go = st.button("🔎 Get Answer", type="primary", use_container_width=True, disabled=not st.session_state.doc_processed)

            if go and query.strip():
                with st.spinner("Running retrieval + rerank + generation…"):
                    try:
                        result = perform_query(query, top_k=top_k, rerank_n=rerank_n)
                        st.session_state.last_stats = result

                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown('<div class="section-title">Answer</div>', unsafe_allow_html=True)
                        st.markdown(result["answer"])
                        st.markdown('</div>', unsafe_allow_html=True)

                        st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
                        st.markdown('<div class="section-title">Sources</div>', unsafe_allow_html=True)
                        if result["sources"]:
                            st.markdown('<div class="chips">', unsafe_allow_html=True)
                            for s in result["sources"]:
                                st.markdown(f'<span class="chip">📄 {s}</span>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.info("No sources found.")
                        st.markdown(f'<div class="section-title" style="margin-top:12px;">Request Metrics</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="kv">latency_sec={result["latency_sec"]} | top_k={result["top_k"]} | rerank_n={result["rerank_n"]}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        with st.expander("View matched chunks (after rerank)"):
                            for c in result["selected_chunks"]:
                                st.markdown(f"**Source [{c['block_no']}]** — _{c['source']}_")
                                st.write(c["text"])
                                st.markdown("---")
                        st.session_state.history.append({"q": query, "a": result["answer"], "sources": result["sources"], "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                    except Exception as e:
                        st.error(f"An error occurred: {e}", icon="🚨")

        st.markdown("### Conversation")
        if not st.session_state.history:
            st.caption("Your Q&A will appear here.")
        else:
            for turn in reversed(st.session_state.history):
                st.markdown('<div class="card" style="margin-bottom:10px;">', unsafe_allow_html=True)
                st.markdown(f"**🧑‍💻 You (@ {turn['ts']})**")
                st.write(turn["q"])
                st.markdown("---")
                st.markdown("**🤖 Assistant**")
                st.write(turn["a"])
                if turn["sources"]:
                    chips = " ".join([f'<span class="chip">📄 {s}</span>' for s in turn["sources"]])
                    st.markdown(chips, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### Debug / Assessment View")
    st.caption("This section reveals internals for grading (prompt + final context).")
    if not st.session_state.last_stats:
        st.info("Run a query to populate this section.")
    else:
        dbg = st.session_state.last_stats
        st.markdown("#### Prompt sent to LLM")
        st.code(dbg["prompt"], language="markdown")
        st.markdown("#### Final Context Blocks (post-rerank)")
        st.code(dbg["context"], language="markdown")
        st.markdown("#### Raw Metrics")
        st.json({"latency_sec": dbg["latency_sec"], "top_k": dbg["top_k"], "rerank_n": dbg["rerank_n"]})

with tab3:
    st.markdown("### About This Demo")
    st.write(
        """
        **Pipeline**: Upload → Chunk (1000/150) → Embed (text-embedding-3-small) → Pinecone search → 
        Cohere Rerank v3.0 (top_n=3) → gpt-4o answer with [n] citations.

        **Why these choices?**
        - *Embedding Model*: 1536-dim, cheap + strong for English.
        - *Rerank*: Improves precision by re-scoring candidate chunks.
        - *Prompt discipline*: Model is instructed to *only* use given context and admit uncertainty.
        """
    )
    st.caption("Built with ❤️ using Streamlit, Pinecone, Cohere, and OpenAI.")

