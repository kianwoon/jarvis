from typing import List, Dict, Any
import torch
from transformers import AutoModel, AutoTokenizer
from app.core.config import get_settings
from app.llm.inference import ModelCache

class QwenEmbedding:
    """Qwen embedding service."""
    
    def __init__(self, model_name: str = "Qwen/Qwen-7B-Chat"):
        self.settings = get_settings()
        self.model_name = model_name
        self.cache = ModelCache()
        self._load_model()
    
    def _load_model(self):
        """Load embedding model and tokenizer."""
        # Use the same tokenizer as the main model
        self.tokenizer = self.cache.get_tokenizer(self.model_name)
        
        # Load embedding model
        self.model = AutoModel.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        self.model.eval()  # Set to evaluation mode
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512  # Typical max length for embeddings
            ).to(self.model.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the last hidden state's [CLS] token as the embedding
                embeddings = outputs.last_hidden_state[:, 0, :]
                # Normalize the embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            return embeddings[0].cpu().numpy().tolist()
            
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {str(e)}")
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            return embeddings.cpu().numpy().tolist()
            
        except Exception as e:
            raise RuntimeError(f"Batch embedding generation failed: {str(e)}")
    
    async def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Compute cosine similarity between two texts."""
        try:
            # Generate embeddings
            emb1 = await self.embed_text(text1)
            emb2 = await self.embed_text(text2)
            
            # Convert to tensors
            emb1_tensor = torch.tensor(emb1)
            emb2_tensor = torch.tensor(emb2)
            
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                emb1_tensor.unsqueeze(0),
                emb2_tensor.unsqueeze(0)
            )
            
            return similarity.item()
            
        except Exception as e:
            raise RuntimeError(f"Similarity computation failed: {str(e)}") 