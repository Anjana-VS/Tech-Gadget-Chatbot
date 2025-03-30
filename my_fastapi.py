from fastapi import FastAPI, Query
import chromadb
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
product_collection = chroma_client.get_collection(name="products")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/recommend")
def recommend_gadget(query: str = Query(..., description="Describe your needs (e.g., best gaming laptop under $1500)")):
    # Convert query to embedding
    query_embedding = model.encode(query).tolist()
    
    # Search in ChromaDB
    results = product_collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # Top 3 results
    )
    
    return results["metadatas"]
