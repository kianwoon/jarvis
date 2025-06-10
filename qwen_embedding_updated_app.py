from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Updated model name
MODEL_NAME = "/app/model/Qwen3-Embedding-0.6B"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.eval().to(device)
    
    # Log model info
    logger.info(f"Model loaded successfully: {MODEL_NAME}")
    logger.info(f"Device: {device}")
    
    # Check embedding dimension
    with torch.no_grad():
        test_input = tokenizer("test", return_tensors="pt").to(device)
        test_output = model(**test_input)
        embedding_dim = test_output.last_hidden_state.shape[-1]
        logger.info(f"Embedding dimension: {embedding_dim}")
        
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

class EmbedRequest(BaseModel):
    texts: list[str]

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    embedding_dim: int

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": str(device),
        "embedding_dim": embedding_dim if 'embedding_dim' in globals() else None
    }

@app.post("/embed", response_model=EmbedResponse)
async def get_embedding(req: EmbedRequest):
    embeddings = []
    for text in req.texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Check if model has a specific pooler output (some models do)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                # Use pooler output if available
                embed = outputs.pooler_output
            else:
                # Otherwise use CLS token (first token) embedding
                embed = outputs.last_hidden_state[:, 0]
            
            # Normalize the embedding
            normed = torch.nn.functional.normalize(embed, p=2, dim=1)
            embeddings.append(normed.cpu().squeeze().tolist())
    
    return EmbedResponse(
        embeddings=embeddings,
        model=MODEL_NAME,
        embedding_dim=len(embeddings[0]) if embeddings else 0
    )

@app.get("/")
async def root():
    return {
        "service": "Qwen Embedding Service",
        "model": MODEL_NAME,
        "endpoints": {
            "/": "Service info",
            "/health": "Health check",
            "/embed": "Generate embeddings (POST)",
            "/docs": "API documentation"
        }
    }