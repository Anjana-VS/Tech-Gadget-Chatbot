import chromadb
import pandas as pd

# Load dataset
csv_path = "gadgets_dataset.csv"
df = pd.read_csv(csv_path)

# Set up ChromaDB Persistent Client (Update the correct path for your local system)
client = chromadb.PersistentClient(path="E:/STEPPING EDGE/ELEC_AND_GADGETS_CHATBOT")

# Create or get the collection
collection_name = "electronics"
try:
    collection = client.create_collection(name=collection_name)
except chromadb.errors.UniqueConstraintError:
    collection = client.get_collection(name=collection_name)

# Extract necessary columns
documents = df["Product Name"].tolist()  # Replace with the correct column name
metadata = df.to_dict(orient="records")  # Store all data as metadata
ids = [str(i) for i in range(len(df))]  # Generate unique IDs

# Insert into ChromaDB
collection.add(documents=documents, metadatas=metadata, ids=ids)

print("✅ Data inserted successfully!")
print("Available collections:", client.list_collections())

