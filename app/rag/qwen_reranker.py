"""
Qwen3-Reranker-4B integration for enhanced RAG retrieval

This module provides reranking capabilities using the Qwen3-Reranker-4B model,
which is specifically designed to work with Qwen3 embeddings.
"""

import torch
from typing import List, Tuple, Optional, Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from dataclasses import dataclass
import os


@dataclass
class RerankResult:
    """Result from reranking operation"""
    document: any  # Document object (e.g., Langchain Document)
    score: float
    original_score: float
    metadata: Dict


class QwenReranker:
    """
    Qwen3-Reranker-4B for document reranking in RAG pipelines
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the Qwen3 Reranker
        
        Args:
            model_path: Path to the model files. If None, uses the default HuggingFace cache
            device: Device to run the model on. If None, auto-detects (cuda if available)
        """
        # Set device
        if device is None:
            # Check for available accelerators
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                # Apple Silicon GPU support
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Set model path
        if model_path is None:
            # Check environment variable first, then use default path
            model_path = os.environ.get(
                "QWEN_RERANKER_MODEL_PATH",
                os.path.expanduser(
                    "~/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B/snapshots/57906229d41697e4494d50ca5859598cf86154a1"
                )
            )
        
        print(f"[QwenReranker] Loading model from: {model_path}")
        print(f"[QwenReranker] Using device: {self.device}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Set padding token if not already set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Use appropriate dtype for device
            if self.device.type == "cuda":
                torch_dtype = torch.float16
            elif self.device.type == "mps":
                # MPS works better with float32 for now
                torch_dtype = torch.float32
            else:
                torch_dtype = torch.float32
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch_dtype,
                pad_token_id=self.tokenizer.pad_token_id,
                low_cpu_mem_usage=True  # Helps with memory efficiency
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"[QwenReranker] Model loaded successfully")
            
        except Exception as e:
            print(f"[QwenReranker] Error loading model: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[Tuple[any, float]],
        top_k: Optional[int] = None,
        instruction: Optional[str] = None,
        return_scores: bool = True
    ) -> List[RerankResult]:
        """
        Rerank documents based on relevance to the query
        
        Args:
            query: The search query
            documents: List of tuples (document, original_score)
            top_k: Number of top documents to return. If None, returns all
            instruction: Custom instruction for the reranking task
            return_scores: Whether to return reranking scores
            
        Returns:
            List of RerankResult objects sorted by relevance (highest first)
        """
        if not documents:
            return []
        
        # Default instruction if none provided
        if instruction is None:
            instruction = "Given a query and a passage, predict whether the passage contains an answer to the query."
        
        # Prepare inputs for batch processing
        queries = []
        documents_text = []
        
        for doc, _ in documents:
            # Extract text content from document
            if hasattr(doc, 'page_content'):
                content = doc.page_content
            elif isinstance(doc, dict) and 'content' in doc:
                content = doc['content']
            elif isinstance(doc, str):
                content = doc
            else:
                content = str(doc)
            
            # Truncate content if too long (keeping first 512 tokens worth)
            content = content[:2048]
            
            # Add query and document to lists
            queries.append(query)
            documents_text.append(content)
        
        # Batch encode all pairs
        # The tokenizer expects text and text_pair arguments
        with torch.no_grad():
            # For MPS, we may need to process in smaller batches to avoid memory issues
            if self.device.type == "mps" and len(queries) > 10:
                # Process in batches of 10 for MPS
                all_scores = []
                batch_size = 10
                
                for i in range(0, len(queries), batch_size):
                    batch_queries = queries[i:i+batch_size]
                    batch_docs = documents_text[i:i+batch_size]
                    
                    inputs = self.tokenizer(
                        batch_queries,
                        batch_docs,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    if logits.shape[-1] == 2:
                        batch_scores = logits[:, 1].cpu().numpy()
                    else:
                        batch_scores = logits.squeeze(-1).cpu().numpy()
                    
                    all_scores.extend(batch_scores)
                
                scores = np.array(all_scores)
                # Apply sigmoid to get probabilities
                scores = 1 / (1 + np.exp(-scores))
            else:
                # Regular processing for small batches or non-MPS devices
                inputs = self.tokenizer(
                    queries,
                    documents_text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get scores from the model
                outputs = self.model(**inputs)
                # The logits might be 2D [batch_size, num_classes] for classification
                logits = outputs.logits
                
                # If binary classification, take the positive class score
                if logits.shape[-1] == 2:
                    scores = logits[:, 1].cpu().numpy()  # Take positive class
                else:
                    scores = logits.squeeze(-1).cpu().numpy()
                
                # Apply sigmoid to get probabilities
                scores = 1 / (1 + np.exp(-scores))
        
        # Create results
        results = []
        for i, (doc, original_score) in enumerate(documents):
            result = RerankResult(
                document=doc,
                score=float(scores[i]) if return_scores else 0.0,
                original_score=original_score,
                metadata={
                    "reranker_model": "Qwen3-Reranker-4B",
                    "instruction": instruction
                }
            )
            results.append(result)
        
        # Sort by reranking score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k if specified
        if top_k is not None and top_k < len(results):
            results = results[:top_k]
        
        return results
    
    def rerank_with_hybrid_score(
        self,
        query: str,
        documents: List[Tuple[any, float]],
        top_k: Optional[int] = None,
        rerank_weight: Optional[float] = None,
        instruction: Optional[str] = None
    ) -> List[RerankResult]:
        """
        Rerank documents using a hybrid score combining original and reranking scores
        
        Args:
            query: The search query
            documents: List of tuples (document, original_score)
            top_k: Number of top documents to return
            rerank_weight: Weight for reranking score (0-1). Original score weight = 1 - rerank_weight
            instruction: Custom instruction for the reranking task
            
        Returns:
            List of RerankResult objects sorted by hybrid score
        """
        # Get reranking results
        results = self.rerank(query, documents, instruction=instruction)
        
        # Get rerank weight from settings if not provided
        if rerank_weight is None:
            from app.core.rag_settings_cache import get_reranking_settings
            rerank_settings = get_reranking_settings()
            rerank_weight = rerank_settings.get('rerank_weight', 0.7)
        
        # Calculate hybrid scores
        original_weight = 1 - rerank_weight
        for result in results:
            # Normalize original score to 0-1 range if needed
            normalized_original = min(1.0, max(0.0, result.original_score))
            
            # Calculate hybrid score
            hybrid_score = (
                result.score * rerank_weight +
                normalized_original * original_weight
            )
            
            # Update metadata
            result.metadata["hybrid_score"] = hybrid_score
            result.metadata["rerank_weight"] = rerank_weight
            result.metadata["original_weight"] = original_weight
        
        # Sort by hybrid score
        results.sort(key=lambda x: x.metadata["hybrid_score"], reverse=True)
        
        # Return top_k if specified
        if top_k is not None and top_k < len(results):
            results = results[:top_k]
        
        return results
    
    def create_task_specific_instruction(self, task_type: str = "general") -> str:
        """
        Create task-specific instructions for better reranking performance
        
        Args:
            task_type: Type of task (general, technical, conversational, etc.)
            
        Returns:
            Instruction string optimized for the task
        """
        instructions = {
            "general": "Given a query and a passage, predict whether the passage contains an answer to the query.",
            "technical": "Given a technical question and a documentation passage, predict whether the passage contains relevant technical information to answer the question.",
            "conversational": "Given a conversational query and a passage, predict whether the passage provides helpful information for responding to the query.",
            "factual": "Given a factual question and a passage, predict whether the passage contains accurate facts that answer the question.",
            "code": "Given a programming-related query and a code snippet or documentation, predict whether it contains relevant information to solve the problem."
        }
        
        return instructions.get(task_type, instructions["general"])


# Singleton instance management
_reranker_instance = None
_init_lock = False


def get_qwen_reranker(model_path: Optional[str] = None, device: Optional[str] = None) -> Optional[QwenReranker]:
    """
    Get or create a singleton instance of QwenReranker
    
    Args:
        model_path: Path to model files
        device: Device to use
        
    Returns:
        QwenReranker instance or None if initialization fails
    """
    global _reranker_instance, _init_lock
    
    # If already initialized, return cached instance
    if _reranker_instance is not None:
        return _reranker_instance
    
    # Prevent concurrent initialization attempts
    if _init_lock:
        print("[QwenReranker] Model initialization already in progress")
        return None
    
    try:
        _init_lock = True
        print("[QwenReranker] Initializing singleton instance...")
        _reranker_instance = QwenReranker(model_path=model_path, device=device)
        print("[QwenReranker] Singleton instance created successfully")
        return _reranker_instance
    except Exception as e:
        print(f"[QwenReranker] Failed to initialize: {e}")
        return None
    finally:
        _init_lock = False


# Pre-initialize the model on module import (optional)
def initialize_reranker_async():
    """Initialize the reranker in the background"""
    import threading
    
    def _init():
        try:
            get_qwen_reranker()
        except Exception as e:
            print(f"[QwenReranker] Background initialization failed: {e}")
    
    # Start initialization in a separate thread
    thread = threading.Thread(target=_init, daemon=True)
    thread.start()


# Optionally pre-initialize on import (commented out by default)
# initialize_reranker_async()