from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import llm  # Import your LLM router

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the LLM router
app.include_router(llm.router, prefix="/api") 