from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import items, chat, prompt

# Create FastAPI app
app = FastAPI(title="My FastAPI Project")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include item routes
app.include_router(items.router)
app.include_router(chat.router)
app.include_router(prompt.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI project!"}
