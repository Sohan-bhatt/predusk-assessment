from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

# Make sure you have your PINECONE_API_KEY in your .env file
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY not found in .env file")

pc = Pinecone(api_key=api_key)

# Define the index name
index_name = "rag-app-index" 

# Check if the index already exists
if index_name not in pc.list_indexes().names():
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Index created successfully.")
else:
    print(f"Index '{index_name}' already exists.")