import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from langchain_community.llms import LlamaCpp


# Load ChromaDB
client = chromadb.PersistentClient(path="E:/STEPPING EDGE/ELEC_AND_GADGETS_CHATBOT")
collection = client.get_collection("electronics")

# Load Embedding Model (Hugging Face)
embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load AI Model (Llama 3.1B)
llm = LlamaCpp(model_path="E:/STEPPING EDGE/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")


def chatbot_query(user_query):
    """Processes user queries, retrieves relevant documents, and generates responses."""
    
    # Convert user query into embedding
    query_embedding = embedding_model.get_text_embedding(user_query)


    # Retrieve relevant products
    search_results = collection.query(query_embeddings=[query_embedding], n_results=5)

    # Format response
    response_text = "Here are some recommendations:\n"
    for idx, doc in enumerate(search_results['documents'][0]):
        response_text += f"{idx+1}. {doc}\n"
    
    # Generate AI-enhanced response
    full_response = llm(f"User: {user_query}\nBot: {response_text}")

    return full_response

# Testing chatbot
if __name__ == "__main__":
    user_input = input("Ask about electronics: ")
    bot_response = chatbot_query(user_input)
    print("\n🤖 Chatbot:", bot_response)
