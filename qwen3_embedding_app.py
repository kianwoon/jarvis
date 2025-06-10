from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Model configuration for Qwen3-Embedding-0.6B
# Update this path based on where your model is located
MODEL_NAME = "/app/model/Qwen3-Embedding-0.6B"

# Alternative: Use from Hugging Face cache or direct model ID
# MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"  # If using from HF directly
# MODEL_NAME = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/latest")  # From cache

try:
    logger.info(f"Loading model from: {MODEL_NAME}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True,
        local_files_only=True if "/app/model" in MODEL_NAME else False
    )
    
    model = AutoModel.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True,
        local_files_only=True if "/app/model" in MODEL_NAME else False
    )
    
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    
    # Log model info
    logger.info(f"✅ Model loaded successfully: {MODEL_NAME}")
    logger.info(f"Device: {device}")
    logger.info(f"Model type: {type(model).__name__}")
    
    # Get model configuration
    config = model.config
    logger.info(f"Hidden size: {config.hidden_size}")
    logger.info(f"Num layers: {config.num_hidden_layers}")
    logger.info(f"Num attention heads: {config.num_attention_heads}")
    
    # Test embedding dimension
    with torch.no_grad():
        test_input = tokenizer("test", return_tensors="pt", padding=True, truncation=True).to(device)
        test_output = model(**test_input)
        
        # Check if model has sentence embeddings attribute (some models do)
        if hasattr(model, 'sentence_embedding_dimension'):
            embedding_dim = model.sentence_embedding_dimension
        else:
            embedding_dim = test_output.last_hidden_state.shape[-1]
            
        logger.info(f"Embedding dimension: {embedding_dim}")
        
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

class EmbedRequest(BaseModel):
    texts: list[str]
    normalize: bool = True  # Option to normalize embeddings
    
class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    embedding_dim: int

@app.get("/")
async def root():
    return {
        "service": "Qwen3-Embedding-0.6B Service",
        "model": MODEL_NAME,
        "status": "ready",
        "embedding_dim": embedding_dim if 'embedding_dim' in globals() else None,
        "device": str(device) if 'device' in globals() else None
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
        # Tokenize input
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Extract embeddings based on model architecture
            # Option 1: If model has a specific embedding method
            if hasattr(model, 'encode'):
                embedding = model.encode(text)
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy()
            
            # Option 2: Use pooler output if available
            elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embedding = outputs.pooler_output
                
            # Option 3: Mean pooling over sequence
            else:
                hidden_states = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                
                # Expand mask for broadcasting
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                
                # Sum masked embeddings
                sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                
                # Calculate mean
                embedding = sum_embeddings / sum_mask
            
            # Normalize if requested
            if req.normalize and isinstance(embedding, torch.Tensor):
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
                
            # Convert to list
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().squeeze().tolist()
                
            embeddings.append(embedding)
    
    return EmbedResponse(
        embeddings=embeddings,
        model=MODEL_NAME,
        embedding_dim=len(embeddings[0]) if embeddings else 0
    )

@app.get("/model-info")
async def model_info():
    """Get detailed model information"""
    info = {
        "model_name": MODEL_NAME,
        "model_type": type(model).__name__ if 'model' in globals() else None,
        "device": str(device) if 'device' in globals() else None,
        "embedding_dim": embedding_dim if 'embedding_dim' in globals() else None,
    }
    
    if 'model' in globals() and hasattr(model, 'config'):
        info["config"] = {
            "hidden_size": model.config.hidden_size,
            "num_hidden_layers": model.config.num_hidden_layers,
            "num_attention_heads": model.config.num_attention_heads,
            "vocab_size": model.config.vocab_size,
            "model_type": model.config.model_type,
        }
    
    return info