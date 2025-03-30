import csv
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
import faiss
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf model
try:
    llm = Llama(
        model_path="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=0,
        verbose=True
    )
    logger.info("LLaMA model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load LLaMA model: {e}")
    raise

# Load the sentence transformer model for embeddings
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("SentenceTransformer model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer model: {e}")
    raise

# Load the tech gadgets dataset from CSV
def load_tech_gadgets_data():
    try:
        data = []
        with open("gadgets_dataset.csv", mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                row["ID"] = int(row["ID"])
                row["Price"] = int(row["Price"])
                row["Popularity Score"] = int(row["Popularity Score"])
                data.append(row)
        logger.info(f"Loaded {len(data)} gadgets from dataset.")
        return data
    except Exception as e:
        logger.error(f"Failed to load tech gadgets dataset: {e}")
        raise

# Create embeddings for the tech gadgets dataset
def embed_tech_gadgets_data(data):
    try:
        descriptions = [
            f"{gadget['Product Name']} {gadget['Category']} {gadget['Brand']} {gadget['Specifications']} {gadget['Features']}"
            for gadget in data
        ]
        embeddings = model.encode(descriptions, convert_to_numpy=True)
        logger.info("Embeddings created successfully.")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to create embeddings: {e}")
        raise

# Create a FAISS index for similarity search
def create_faiss_index(embeddings):
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        logger.info("FAISS index created successfully.")
        return index
    except Exception as e:
        logger.error(f"Failed to create FAISS index: {e}")
        raise

# Global variables
try:
    TECH_GADGETS_DATA = load_tech_gadgets_data()
    embeddings = embed_tech_gadgets_data(TECH_GADGETS_DATA)
    faiss_index = create_faiss_index(embeddings)
except Exception as e:
    logger.error(f"Failed to initialize global variables: {e}")
    raise

# Define brands for each category
category_brands = {
    "smartphone": ["apple", "samsung", "xiaomi", "oneplus"],
    "laptop": ["dell", "hp", "asus", "lenovo", "microsoft", "apple"],
    "tablet": ["apple", "samsung", "xiaomi", "lenovo"],
    "smartwatch": ["apple", "samsung", "garmin", "oneplus"],
    "headphones": ["sony", "sennheiser", "bose", "jbl"]
}

# Define budget ranges for each category
category_budget_ranges = {
    "smartphone": {
        "300-800": (300, 800),
        "801-1200": (801, 1200),
        "1201-1800": (1201, 1800),
        "1801-2500": (1801, 2500)
    },
    "laptop": {
        "500-1000": (500, 1000),
        "1001-1500": (1001, 1500),
        "1501-2000": (1501, 2000),
        "2001-3000": (2001, 3000)
    },
    "tablet": {
        "200-500": (200, 500),
        "501-800": (501, 800),
        "801-1200": (801, 1200),
        "1201-1500": (1201, 1500)
    },
    "smartwatch": {
        "100-300": (100, 300),
        "301-500": (301, 500),
        "501-800": (501, 800)
    },
    "headphones": {
        "50-150": (50, 150),
        "151-300": (151, 300),
        "301-600": (301, 600)
    }
}

# Pydantic model for chat requests
class ChatRequest(BaseModel):
    message: str
    context: dict = None

# Helper function to compare products and generate a comparison summary
def compare_products(retrieved_items):
    if not retrieved_items:
        return "No products available to compare."

    # Start with an empty comparison summary
    comparison_summary = ""

    # Display each product as a bullet point with all details inline
    for i, item in enumerate(retrieved_items, 1):
        product_details = (
            f"- Product {i}: {item['Product Name']}, "
            f"Category: {item['Category']}, "
            f"Brand: {item['Brand']}, "
            f"Specifications: {item['Specifications']}, "
            f"Price: ${item['Price']}, "
            f"Features: {item['Features']}, "
            f"User Reviews: {item['User Reviews']}, "
            f"Popularity Score: {item['Popularity Score']}"
        )
        comparison_summary += product_details
        # Add a newline after each product, but not after the last one
        if i < len(retrieved_items):
            comparison_summary += "\n"

    # Add a single trailing newline
    comparison_summary += "\n"

    return comparison_summary

# Process user messages and manage conversation state
def process_message(message, context):
    if "current_step" not in context:
        context["current_step"] = "category"
        context["preferences"] = {}
        context["recommendation_history"] = []

    current_step = context["current_step"]
    preferences = context["preferences"]

    if message == "start":
        context["current_step"] = "category"
        return "What type of gadget are you looking for? (options: Smartphone, Laptop, Tablet, Smartwatch, Headphones)", context

    if current_step == "category":
        categories = ["smartphone", "laptop", "tablet", "smartwatch", "headphones"]
        if message in categories:
            preferences["category"] = message
            context["current_step"] = "brand"
            relevant_brands = category_brands[message]
            brands_str = ", ".join([brand.capitalize() for brand in relevant_brands])
            return f"Which brand do you prefer for your {message}? (options: {brands_str})", context
        return "Please select a valid category: Smartphone, Laptop, Tablet, Smartwatch, Headphones", context

    if current_step == "brand":
        relevant_brands = category_brands[preferences["category"]]
        if message in relevant_brands:
            preferences["brand"] = message
            context["current_step"] = "budget"
            budget_ranges = category_budget_ranges[preferences["category"]]
            # Add dollar symbol to budget options
            budget_options = ", ".join([f"${key.replace('-', '-$')}" for key in budget_ranges.keys()])
            return f"What’s your budget range for your gadget? (options: {budget_options})", context
        brands_str = ", ".join([brand.capitalize() for brand in relevant_brands])
        return f"Please select a valid brand: {brands_str}", context

    if current_step == "budget":
        budget_ranges = category_budget_ranges[preferences["category"]]
        # Strip dollar symbols from user input for validation
        cleaned_message = message.replace("$", "")
        if cleaned_message in budget_ranges:
            preferences["budget"] = budget_ranges[cleaned_message]
            context["current_step"] = "sort"
            return "How would you like to sort the recommendations? (options: best seller, new arrival, price low to high, price high to low)", context
        budget_options = ", ".join([f"${key.replace('-', '-$')}" for key in budget_ranges.keys()])
        return f"Please select a valid budget range: {budget_options}", context

    if current_step == "sort":
        sort_options = ["best seller", "new arrival", "price low to high", "price high to low"]
        if message in sort_options:
            preferences["sort"] = message
            context["current_step"] = "recommend"

            logger.info(f"Preferences: {preferences}")

            filtered_gadgets = TECH_GADGETS_DATA
            logger.info(f"Total gadgets before filtering: {len(filtered_gadgets)}")

            if "category" in preferences:
                filtered_gadgets = [gadget for gadget in filtered_gadgets if gadget["Category"].lower() == preferences["category"].lower()]
                logger.info(f"Gadgets after category filter ({preferences['category']}): {len(filtered_gadgets)}")

            if "brand" in preferences:
                filtered_gadgets = [gadget for gadget in filtered_gadgets if gadget["Brand"].lower() == preferences["brand"].lower()]
                logger.info(f"Gadgets after brand filter ({preferences['brand']}): {len(filtered_gadgets)}")

            if "budget" in preferences:
                min_price, max_price = preferences["budget"]
                filtered_gadgets = [gadget for gadget in filtered_gadgets if min_price <= gadget["Price"] <= max_price]
                logger.info(f"Gadgets after budget filter ({min_price}-{max_price}): {len(filtered_gadgets)}")

            if preferences["sort"] == "best seller":
                filtered_gadgets.sort(key=lambda x: x["Popularity Score"], reverse=True)
            elif preferences["sort"] == "new arrival":
                filtered_gadgets.sort(key=lambda x: x["ID"], reverse=True)
            elif preferences["sort"] == "price low to high":
                filtered_gadgets.sort(key=lambda x: x["Price"])
            elif preferences["sort"] == "price high to low":
                filtered_gadgets.sort(key=lambda x: x["Price"], reverse=True)

            logger.info(f"Filtered gadgets: {filtered_gadgets}")

            filtered_gadgets = filtered_gadgets[:3]

            if not filtered_gadgets:
                return f"Sorry, I couldn't find any {preferences['category']}s from {preferences['brand'].capitalize()} in the price range ${preferences['budget'][0]}-${preferences['budget'][1]}. Would you like to explore more options? (options: explore more, stop)", context

            context["last_retrieved_items"] = filtered_gadgets
            context["recommendation_history"].append(filtered_gadgets)

            # Prepare the product list to ensure it's always displayed
            product_list = "Let me show you some awesome options that fit your budget and preferences!\n"
            for gadget in filtered_gadgets:
                product_list += f"- {gadget['Product Name']}: {gadget['Specifications']}, priced at ${gadget['Price']}, features: {gadget['Features']}, user reviews: {gadget['User Reviews']}, popularity score: {gadget['Popularity Score']}\n"

            # Try to generate a response with LLaMA
            prompt = f"Based on the user's preferences (category: {preferences['category']}, brand: {preferences['brand']}, budget: {preferences['budget'][0]}-{preferences['budget'][1]}), I found the following gadgets:\n"
            for gadget in filtered_gadgets:
                prompt += f"- {gadget['Product Name']}: {gadget['Specifications']}, priced at ${gadget['Price']}, features: {gadget['Features']}, user reviews: {gadget['User Reviews']}, popularity score: {gadget['Popularity Score']}\n"
            prompt += "Generate a friendly and inviting response introducing these gadgets to the user in a conversational tone. Start with a warm greeting like 'Let me show you some awesome options that fit your budget and preferences!' Mention each gadget's name, price (with a dollar symbol), features, user reviews, and popularity score. Encourage the user to engage further by asking 'Which one of these devices catches your eye? Let me know and I can provide more information!' Also, mention that these options fit within the user's budget."

            try:
                llm_response = llm(prompt, max_tokens=500, stop=["\n\n"], temperature=0.7)
                response = llm_response["choices"][0]["text"].strip()
                # Ensure the response contains the product list
                if not any(gadget["Product Name"] in response for gadget in filtered_gadgets):
                    response = product_list + "\nWhich one of these devices catches your eye? Let me know and I can provide more information!"
            except Exception as e:
                logger.error(f"Failed to generate LLM response: {e}")
                response = product_list + "\nWhich one of these devices catches your eye? Let me know and I can provide more information!"

            return f"{response}\nThese options all fit within your budget of ${preferences['budget'][0]}-${preferences['budget'][1]}. Would you like to compare these products, proceed with one of these options, or stop the process? (options: compare, proceed, stop, explore more, go back to the previous recommendations)", context

        return "Please select a valid sort option: best seller, new arrival, price low to high, price high to low", context

    if current_step == "recommend":
        if message == "compare":
            context["current_step"] = "compare_products"
            retrieved_items = context.get("last_retrieved_items", [])
            comparison_summary = compare_products(retrieved_items)
            response = f"Here’s a detailed comparison of the recommended products:\n\n{comparison_summary}\nWould you like to proceed with one of these options, stop the process, explore more options, or go back to the previous recommendations? (options: proceed, stop, explore more, go back to the previous recommendations)"
            return response, context
        elif message == "proceed":
            context["current_step"] = "select_product"
            recommended_products = context.get("last_retrieved_items", [])
            if not recommended_products:
                return "Sorry, I don't have any recommendations to proceed with. Would you like to explore more options? (options: explore more, stop)", context

            product_options = [gadget["Product Name"] for gadget in recommended_products]
            response = "Great! Let's pick a product to proceed with. Here are the options I recommended:\n\n"
            for gadget in recommended_products:
                response += f"- {gadget['Product Name']}: ${gadget['Price']}\n"
            response += f"\nWhich one would you like to choose? (options: {', '.join(product_options)}, explore more, stop)"
            return response, context
        elif message == "stop":
            context["current_step"] = "category"
            context["preferences"] = {}
            context["recommendation_history"] = []
            return "Thanks for chatting! If you'd like to start over, just say 'start'.", context
        elif message == "explore more":
            context["current_step"] = "category"
            context["preferences"] = {}
            return "Let's explore more options. What type of gadget are you looking for? (options: Smartphone, Laptop, Tablet, Smartwatch, Headphones)", context
        elif message == "go back to the previous recommendations":
            if len(context["recommendation_history"]) > 1:
                context["recommendation_history"].pop()
                context["last_retrieved_items"] = context["recommendation_history"][-1]
                prompt = "Here are the previous recommendations:\n"
                for gadget in context["last_retrieved_items"]:
                    prompt += f"- {gadget['Product Name']}: {gadget['Specifications']}, priced at ${gadget['Price']}, features: {gadget['Features']}, user reviews: {gadget['User Reviews']}, popularity score: {gadget['Popularity Score']}\n"
                prompt += "Generate a friendly and inviting response reintroducing these gadgets to the user in a conversational tone. Start with a warm greeting like 'Let’s take a look at the previous options I found for you!' Mention each gadget's name, price (with a dollar symbol), features, user reviews, and popularity score. Encourage the user to engage further by asking 'Which one of these devices catches your eye now? Let me know and I can provide more information!'"

                try:
                    llm_response = llm(prompt, max_tokens=500, stop=["\n\n"], temperature=0.7)
                    response = llm_response["choices"][0]["text"].strip()
                    # Ensure the response contains the product list
                    if not any(gadget["Product Name"] in response for gadget in context["last_retrieved_items"]):
                        response = "Let’s take a look at the previous options I found for you!\n"
                        for gadget in context["last_retrieved_items"]:
                            response += f"- {gadget['Product Name']}: {gadget['Specifications']}, priced at ${gadget['Price']}, features: {gadget['Features']}, user reviews: {gadget['User Reviews']}, popularity score: {gadget['Popularity Score']}\n"
                        response += "Which one of these devices catches your eye now? Let me know and I can provide more information!"
                except Exception as e:
                    logger.error(f"Failed to generate LLM response: {e}")
                    response = "Let’s take a look at the previous options I found for you!\n"
                    for gadget in context["last_retrieved_items"]:
                        response += f"- {gadget['Product Name']}: {gadget['Specifications']}, priced at ${gadget['Price']}, features: {gadget['Features']}, user reviews: {gadget['User Reviews']}, popularity score: {gadget['Popularity Score']}\n"
                    response += "Which one of these devices catches your eye now? Let me know and I can provide more information!"

                return f"{response}\nWould you like to compare these products, proceed with one of these options, or stop the process? (options: compare, proceed, stop, explore more, go back to the previous recommendations)", context
            return "There are no previous recommendations to go back to. Would you like to explore more options? (options: explore more, stop)", context

        return "Please select an option: compare, proceed, stop, explore more, go back to the previous recommendations", context

    if current_step == "compare_products":
        if message == "proceed":
            context["current_step"] = "select_product"
            recommended_products = context.get("last_retrieved_items", [])
            if not recommended_products:
                return "Sorry, I don't have any recommendations to proceed with. Would you like to explore more options? (options: explore more, stop)", context

            product_options = [gadget["Product Name"] for gadget in recommended_products]
            response = "Great! Let's pick a product to proceed with. Here are the options I recommended:\n\n"
            for gadget in recommended_products:
                response += f"- {gadget['Product Name']}: ${gadget['Price']}\n"
            response += f"\nWhich one would you like to choose? (options: {', '.join(product_options)}, explore more, stop)"
            return response, context
        elif message == "stop":
            context["current_step"] = "category"
            context["preferences"] = {}
            context["recommendation_history"] = []
            return "Thanks for chatting! If you'd like to start over, just say 'start'.", context
        elif message == "explore more":
            context["current_step"] = "category"
            context["preferences"] = {}
            return "Let's explore more options. What type of gadget are you looking for? (options: Smartphone, Laptop, Tablet, Smartwatch, Headphones)", context
        elif message == "go back to the previous recommendations":
            if len(context["recommendation_history"]) > 1:
                context["recommendation_history"].pop()
                context["last_retrieved_items"] = context["recommendation_history"][-1]
                context["current_step"] = "recommend"
                prompt = "Here are the previous recommendations:\n"
                for gadget in context["last_retrieved_items"]:
                    prompt += f"- {gadget['Product Name']}: {gadget['Specifications']}, priced at ${gadget['Price']}, features: {gadget['Features']}, user reviews: {gadget['User Reviews']}, popularity score: {gadget['Popularity Score']}\n"
                prompt += "Generate a friendly and inviting response reintroducing these gadgets to the user in a conversational tone. Start with a warm greeting like 'Let’s take a look at the previous options I found for you!' Mention each gadget's name, price (with a dollar symbol), features, user reviews, and popularity score. Encourage the user to engage further by asking 'Which one of these devices catches your eye now? Let me know and I can provide more information!'"

                try:
                    llm_response = llm(prompt, max_tokens=500, stop=["\n\n"], temperature=0.7)
                    response = llm_response["choices"][0]["text"].strip()
                    # Ensure the response contains the product list
                    if not any(gadget["Product Name"] in response for gadget in context["last_retrieved_items"]):
                        response = "Let’s take a look at the previous options I found for you!\n"
                        for gadget in context["last_retrieved_items"]:
                            response += f"- {gadget['Product Name']}: {gadget['Specifications']}, priced at ${gadget['Price']}, features: {gadget['Features']}, user reviews: {gadget['User Reviews']}, popularity score: {gadget['Popularity Score']}\n"
                        response += "Which one of these devices catches your eye now? Let me know and I can provide more information!"
                except Exception as e:
                    logger.error(f"Failed to generate LLM response: {e}")
                    response = "Let’s take a look at the previous options I found for you!\n"
                    for gadget in context["last_retrieved_items"]:
                        response += f"- {gadget['Product Name']}: {gadget['Specifications']}, priced at ${gadget['Price']}, features: {gadget['Features']}, user reviews: {gadget['User Reviews']}, popularity score: {gadget['Popularity Score']}\n"
                    response += "Which one of these devices catches your eye now? Let me know and I can provide more information!"

                return f"{response}\nWould you like to compare these products, proceed with one of these options, or stop the process? (options: compare, proceed, stop, explore more, go back to the previous recommendations)", context
            return "There are no previous recommendations to go back to. Would you like to explore more options? (options: explore more, stop)", context
        return "Please select an option: proceed, stop, explore more, go back to the previous recommendations", context

    if current_step == "select_product":
        recommended_products = context.get("last_retrieved_items", [])
        product_names = [gadget["Product Name"].lower() for gadget in recommended_products]

        if message in product_names:
            selected_product = next(gadget for gadget in recommended_products if gadget["Product Name"].lower() == message)
            context["selected_product"] = selected_product
            context["current_step"] = "finalize"
            return f"You've selected {selected_product['Product Name']} for ${selected_product['Price']}. Would you like to add it to your cart, explore more items, or finalize your order? (options: add to cart, explore more, finalize my order)", context

        if message == "explore more":
            context["current_step"] = "category"
            context["preferences"] = {}
            return "Let's explore more options. What type of gadget are you looking for? (options: Smartphone, Laptop, Tablet, Smartwatch, Headphones)", context

        if message == "stop":
            context["current_step"] = "category"
            context["preferences"] = {}
            context["recommendation_history"] = []
            return "Thanks for chatting! If you'd like to start over, just say 'start'.", context

        product_options = ", ".join([gadget["Product Name"] for gadget in recommended_products])
        return f"Please select a valid product: {product_options}, or choose 'explore more' or 'stop'.", context

    if current_step == "finalize":
        if message == "add to cart":
            selected_product = context.get("selected_product", {})
            if selected_product:
                context["cart"] = context.get("cart", []) + [selected_product]
                response = f"{selected_product['Product Name']} has been added to your cart! Would you like to explore more items or finalize your order? (options: explore more, finalize my order)"
            else:
                response = "No product selected to add to cart. Let's explore more options. What type of gadget are you looking for? (options: Smartphone, Laptop, Tablet, Smartwatch, Headphones)"
                context["current_step"] = "category"
                context["preferences"] = {}
            return response, context

        if message == "explore more":
            context["current_step"] = "category"
            context["preferences"] = {}
            return "Let's explore more options. What type of gadget are you looking for? (options: Smartphone, Laptop, Tablet, Smartwatch, Headphones)", context

        if message == "finalize my order":
            cart = context.get("cart", [])
            if not cart:
                response = "Your cart is empty. Let's explore more gadgets! What type of gadget are you looking for? (options: Smartphone, Laptop, Tablet, Smartwatch, Headphones)"
                context["current_step"] = "category"
            else:
                cart_items = "\n".join([f"- {item['Product Name']}: ${item['Price']}" for item in cart])
                total_price = sum(item["Price"] for item in cart)
                response = f"Thank you for your order! Here’s what you’ve selected:\n{cart_items}\nTotal: ${total_price}\nYour order has been finalized. If you'd like to explore more gadgets, just say 'start'."
                context["current_step"] = "category"
                context["preferences"] = {}
                context["recommendation_history"] = []
                context["cart"] = []
            return response, context

        return "Please select an option: add to cart, explore more, finalize my order", context

    return "I'm not sure how to proceed. Please select an option or say 'start' to begin again.", context

# FastAPI endpoint for chat
@app.post("/chat")
async def chat(request: ChatRequest):
    message = request.message.lower().strip()
    context = request.context or {}
    response, updated_context = process_message(message, context)
    return {"response": response, "context": updated_context}

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Tech Gadget Recommendation Chatbot API"}

# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)