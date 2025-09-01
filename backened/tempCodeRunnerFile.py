from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="pcsk_4BMx9P_A47TbSJxCh3SfGgktAgh1Am66Zfx8PVg65hYRhoGHXhUnCMtuJnA1T2WUcPemXs")
index_name = "developer-quickstart-py"

if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )