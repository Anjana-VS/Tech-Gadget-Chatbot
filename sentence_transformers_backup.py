from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example product
product = {
    "name": "Apple MacBook Pro 14-inch",
    "category": "Laptop",
    "price": 1999.99,
    "description": "Apple M2 Pro chip with 10-core CPU and 16-core GPU, 16GB RAM, 512GB SSD.",
}

# Generate embeddings
embedding = model.encode(product["description"]).tolist()

# Store in ChromaDB
product_collection.add(
    ids=["1"],  # Unique ID
    metadatas=[{"name": product["name"], "category": product["category"], "price": product["price"]}],
    embeddings=[embedding]
)
