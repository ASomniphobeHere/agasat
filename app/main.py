from fastapi import FastAPI
from app.routes import items, chat

# Create FastAPI app
app = FastAPI(title="My FastAPI Project")

# Include item routes
app.include_router(items.router)
app.include_router(chat.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI project!"}
