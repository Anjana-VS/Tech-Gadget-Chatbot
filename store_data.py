import chromadb
import pandas as pd

# Load dataset
df = pd.read_csv("gadgets_dataset.csv")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="chroma_db")

# Create collection
collection = client.get_or_create_collection(name="gadgets")

# Insert data into ChromaDB
for index, row in df.iterrows():
    collection.add(
        ids=[str(index)],
        documents=[row.to_json()]
    )

print("✅ Data stored in ChromaDB successfully!")
