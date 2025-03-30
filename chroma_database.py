import chromadb

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection for storing product embeddings
product_collection = chroma_client.get_or_create_collection(name="products")
