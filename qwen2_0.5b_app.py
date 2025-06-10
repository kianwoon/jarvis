from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Updated model name for Qwen2-0.5B (3x smaller than current 1.5B model)
MODEL_NAME = "/app/model/Qwen2-0.5B"

try:
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
    
    # Use CPU by default (change to "cuda" if GPU is available)
    device = torch.device("cpu")
    model.eval().to(device)
    
    # Log model info
    logger.info(f"Model loaded successfully: {MODEL_NAME}")
    logger.info(f"Device: {device}")
    logger.info(f"Model type: {type(model).__name__}")
    
    # Get embedding dimension
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

@app.get("/")
async def root():
    return {
        "service": "Qwen2-0.5B Embedding Service",
        "model": MODEL_NAME,
        "status": "ready",
        "embedding_dim": embedding_dim if 'embedding_dim' in globals() else None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": str(device)
    }

@app.post("/embed")
async def get_embedding(req: EmbedRequest):
    embeddings = []
    
    for text in req.texts:
        # Tokenize with truncation and padding
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            # For Qwen2 models, use mean pooling over all tokens
            # This often gives better results than just using CLS token
            hidden_states = outputs.last_hidden_state
            
            # Create attention mask for proper mean pooling
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            
            # Sum embeddings and divide by number of tokens
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            
            # Normalize the embedding
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            embeddings.append(normalized.cpu().squeeze().tolist())
    
    return EmbedResponse(
        embeddings=embeddings,
        model=MODEL_NAME,
        embedding_dim=len(embeddings[0]) if embeddings else 0
    )