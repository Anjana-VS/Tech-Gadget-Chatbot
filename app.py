from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# Dummy product data
products = [
    {"id": 1, "name": "iPhone 14", "category": "Smartphone", "price": 999.99, "specs": "128GB, A15 Bionic"},
    {"id": 2, "name": "Samsung Galaxy S23", "category": "Smartphone", "price": 899.99, "specs": "256GB, Snapdragon 8 Gen 2"},
    {"id": 3, "name": "MacBook Air M2", "category": "Laptop", "price": 1199.99, "specs": "256GB SSD, 8GB RAM"},
    {"id": 4, "name": "Dell XPS 13", "category": "Laptop", "price": 1299.99, "specs": "512GB SSD, 16GB RAM"},
]

# Product model
class Product(BaseModel):
    name: str
    category: str
    price: float
    specs: str

# Home route
@app.get("/")
def home():
    return {"message": "Welcome to the Electronics & Gadgets Chatbot API"}

# Get all products
@app.get("/products", response_model=List[dict])
def get_products(category: Optional[str] = None, sort_by: Optional[str] = None):
    filtered_products = products

    # Filter by category
    if category:
        filtered_products = [p for p in products if p["category"].lower() == category.lower()]

    # Sorting options
    if sort_by == "price":
        filtered_products.sort(key=lambda x: x["price"])
    elif sort_by == "name":
        filtered_products.sort(key=lambda x: x["name"])

    return filtered_products

# Add a new product
@app.post("/products", response_model=Product)
def add_product(product: Product):
    new_product = {"id": len(products) + 1, **product.dict()}
    products.append(new_product)
    return new_product

# Get a single product by ID
@app.get("/products/{product_id}")
def get_product(product_id: int):
    product = next((p for p in products if p["id"] == product_id), None)
    if not product:
        return {"error": "Product not found"}
    return product

