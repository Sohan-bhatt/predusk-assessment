import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cohere
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from dotenv import load_dotenv

# --- Initialization ---
load_dotenv() # Load environment variables from .env file for local development

# Load API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Check if keys are available
if not all([PINECONE_API_KEY, OPENAI_API_KEY, COHERE_API_KEY]):
    raise ValueError("One or more API keys are missing. Please check your .env file or environment variables.")

# Initialize clients and models
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
co = cohere.Client(api_key=COHERE_API_KEY)

# Connect to your Pinecone index
# Make sure to replace "your-index-name" with the actual name of your index in Pinecone
INDEX_NAME = "developer-quickstart-py"
# check if the index exists
if INDEX_NAME not in pc.list_indexes().names():
    raise ValueError(f"Index '{INDEX_NAME}' does not exist. Please create it in the Pinecone console.")
index = pc.Index(INDEX_NAME)

app = FastAPI()

# --- Pydantic Models for API validation ---
class UpsertRequest(BaseModel):
    text: str
    source: str

class QueryRequest(BaseModel):
    query: str

# --- API Endpoints ---
@app.post("/upsert")
async def upsert_text(request: UpsertRequest):
    try:
        # 1. Chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_text(request.text)

        # 2. Create embeddings for each chunk
        chunk_embeddings = embeddings.embed_documents(chunks)

        # 3. Prepare vectors for Pinecone upsert
        vectors_to_upsert = []
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            vectors_to_upsert.append({
                "id": f"{request.source}-{i}",
                "values": embedding,
                "metadata": {"text": chunk, "source": request.source}
            })

        # 4. Upsert to Pinecone in batches for efficiency
        # You can adjust batch_size as needed
        index.upsert(vectors=vectors_to_upsert, batch_size=100)
        
        return {"message": "Text processed and upserted successfully.", "chunk_count": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def handle_query(request: QueryRequest):
    try:
        # 1. Embed the query
        query_embedding = embeddings.embed_query(request.query)

        # 2. Retrieve top-k documents from Pinecone
        query_response = index.query(
            vector=query_embedding,
            top_k=10,  # Retrieve more than you need for reranking
            include_metadata=True
        )
        
        # 3. Prepare documents for reranking
        retrieved_docs = []
        for match in query_response['matches']:
            # We pass the metadata as well to keep track of the source
            retrieved_docs.append({
                "text": match['metadata']['text'],
                "source": match['metadata']['source'],
                "id": match['id']
            })

        # 4. Rerank the results using Cohere
        reranked_results = co.rerank(
            query=request.query,
            documents=retrieved_docs,
            top_n=3, # Rerank and keep the top 3
            model="rerank-english-v2.0"
        )
        
        # 5. Build the context and citations for the LLM
        context = ""
        citations = []
        # Create a unique list of sources from the reranked results
        unique_sources = []
        for result in reranked_results.results:
            doc = result.document
            if doc['source'] not in [s['source'] for s in unique_sources]:
                unique_sources.append({'source': doc['source'], 'id': doc['id']})

        # Build the context with numbered sources
        for i, doc_info in enumerate(unique_sources):
             # Find the full text for the source from the reranked documents
            full_text = next((r.document['text'] for r in reranked_results.results if r.document['id'] == doc_info['id']), "")
            context += f"Source [{i+1}] (from {doc_info['source']}): {full_text}\n\n"
            citations.append(f"[{i+1}] {doc_info['source']}")

        # 6. Generate Answer with LLM
        prompt = f"""
        Answer the user's query based only on the following context.
        Cite your sources using the format [1], [2], etc., at the end of each relevant sentence.
        If the context does not provide an answer, state that you cannot answer the question based on the provided information.

        Context:
        {context}

        Query: {request.query}
        Answer:
        """
        
        llm_response = llm.invoke(prompt)
        
        return {"answer": llm_response.content, "citations": citations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))