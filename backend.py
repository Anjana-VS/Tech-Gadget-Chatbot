from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import chromadb
import os
from langchain_community.llms import LlamaCpp
from uuid import uuid4

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChromaDB
CHROMA_DB_PATH = "E:\STEPPING EDGE\models"
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

try:
    collection = chroma_client.get_collection("electronics")
except:
    collection = chroma_client.create_collection("electronics")

# Load Llama Model with Optimized Settings
MODEL_PATH = "E:/STEPPING EDGE/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=2048)

# Temporary in-memory session storage
sessions = {}

@app.post("/start_session")
async def start_session():
    """Creates a new QnA session and returns session ID."""
    session_id = str(uuid4())  # Generate unique session ID
    sessions[session_id] = {"questions": [], "answers": []}  # Store QnA history
    return {"session_id": session_id, "message": "QnA session started!"}

@app.post("/ask")
async def ask_question(query: str, session_id: str):
    """Handles guided QnA and retrieves related gadgets."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = sessions[session_id]

    # Retrieve top 3 most relevant gadgets from ChromaDB
    results = collection.query(query_texts=[query], n_results=3)

    # Extract relevant details
    if results["documents"]:
        gadgets = [doc for doc in results["documents"][0]]
        response_text = f"Here are some gadgets that match your search: {', '.join(gadgets)}"
    else:
        response_text = "Sorry, I couldn't find any relevant gadgets."

    # Determine the next guided question
    if not session["questions"]:  # First question
        next_question = "What is your budget range?"
    elif "budget" in session["questions"][-1].lower():
        budget = session["answers"][-1].lower()
        if "low" in budget:
            next_question = "Do you prefer a refurbished or new product?"
        elif "high" in budget:
            next_question = "Are you looking for gaming or business use?"
        else:
            next_question = "Can you specify your preferred price range?"
    else:
        next_question = "Would you like more details on any of these products?"

    # Store session history
    session["questions"].append(query)
    session["answers"].append(next_question)

    return {"response": response_text, "next_question": next_question}

@app.post("/final_recommendation")
async def final_recommendation(session_id: str):
    """Generates a final product recommendation using ChromaDB & Llama."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    user_answers = " ".join(sessions[session_id]["answers"])

    # Retrieve the most relevant product based on user answers
    results = collection.query(query_texts=[user_answers], n_results=1)
    
    if results["documents"]:
        recommended_product = results["documents"][0][0]
    else:
        recommended_product = "No specific match found, but I can suggest some options."

    # Generate response using Llama
    prompt = f"Based on the following user preferences: {user_answers}, recommend a gadget. The best match found is: {recommended_product}."
    response = llm.invoke(prompt)

    return {"recommendation": response, "matched_product": recommended_product}
