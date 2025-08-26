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

# Initialize environment variables for local-only mode if configured
def _initialize_local_mode():
    """Initialize local-only mode based on configuration"""
    try:
        from app.core.reranker_config import RerankerConfig
        if RerankerConfig.force_local_only():
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            print("[QwenReranker] Module initialization: Enforcing offline mode - local files only")
    except ImportError:
        # Configuration not available during import, will be set in constructor
        pass

# Initialize local mode on module import
_initialize_local_mode()


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
        
        # Enforce local-only mode based on configuration
        from app.core.reranker_config import RerankerConfig
        if RerankerConfig.force_local_only():
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            print("[QwenReranker] Enforcing offline mode - local files only")
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
            # Use configured path from RerankerConfig
            model_path = RerankerConfig.get_model_path()
            
            if model_path is None:
                # Try different common paths (Docker vs local) - prioritize 0.6B model
                possible_paths = [
                    # 0.6B model paths (preferred - smaller, faster) with snapshot directories
                    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/6e9e69830b95c52b5fd889b7690dda3329508de3",
                    os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/6e9e69830b95c52b5fd889b7690dda3329508de3"),
                    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B",
                    os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B"),
                    # 4B model paths (fallback)
                    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B/snapshots/57906229d41697e4494d50ca5859598cf86154a1",
                    os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B/snapshots/57906229d41697e4494d50ca5859598cf86154a1"),
                    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B",
                    os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B")
                ]
                
                # Use the first path that exists
                for path in possible_paths:
                    if os.path.exists(path):
                        model_path = path
                        print(f"[QwenReranker] Found model at: {model_path}")
                        break
                
                # Fallback to first path if none exist (will trigger download)
                if model_path is None:
                    model_path = possible_paths[0]
        
        print(f"[QwenReranker] Loading model from: {model_path}")
        print(f"[QwenReranker] Using device: {self.device}")
        
        # Initialize flags
        self.use_causal_lm = False
        
        # Load tokenizer and model
        try:
            # Check if we're in Docker and use appropriate loading strategy
            is_docker = os.path.exists('/root') or os.environ.get('DOCKER_ENVIRONMENT')
            
            if is_docker:
                # Docker environment - use Docker-safe tokenizer loading
                self.tokenizer = self._load_tokenizer_docker_safe()
            else:
                # Local environment - try local path first, then Hub
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        local_files_only=True
                    )
                    print(f"[QwenReranker] Loaded tokenizer from local path: {model_path}")
                except Exception as local_error:
                    print(f"[QwenReranker] Local tokenizer loading failed: {str(local_error)[:100]}")
                    # Fallback to Hub
                    self.tokenizer = self._load_tokenizer_docker_safe()
            
            # Set padding token if not already set
            if self.tokenizer.pad_token is None:
                # For Qwen models, use eos_token as pad_token
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Ensure pad_token_id is properly set in model config
            if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                print(f"[QwenReranker] Using pad_token_id: {self.tokenizer.pad_token_id}")
            
            # Use appropriate dtype for device
            if self.device.type == "cuda":
                torch_dtype = torch.float16
            elif self.device.type == "mps":
                # MPS works better with float32 for now
                torch_dtype = torch.float32
            else:
                torch_dtype = torch.float32
            
            # Try multiple loading approaches to handle ModelWrapper and other errors
            # Check if we're in Docker environment to avoid ModelWrapper issues
            is_docker = os.path.exists('/root') or os.environ.get('DOCKER_ENVIRONMENT')
            
            if is_docker:
                # Docker environment - ModelWrapper errors are common, use simplified loading
                print(f"[QwenReranker] Docker environment detected, using simplified loading due to safetensors compatibility issues")
                loading_attempts = [
                    {
                        "name": "Docker Safe Loading (0.6B)",
                        "func": lambda: self._load_docker_safe(torch_dtype, "Qwen/Qwen3-Reranker-0.6B")
                    },
                    {
                        "name": "Docker Safe Loading (4B)", 
                        "func": lambda: self._load_docker_safe(torch_dtype, "Qwen/Qwen3-Reranker-4B")
                    },
                    {
                        "name": "HuggingFace Hub (Sequence Classification)",
                        "func": lambda: self._load_from_hub_sequence_classification(torch_dtype)
                    },
                    {
                        "name": "HuggingFace Hub (CausalLM)", 
                        "func": lambda: self._load_from_hub_causal_lm(torch_dtype)
                    }
                ]
            else:
                # Local environment - prioritize local cache first
                loading_attempts = [
                    {
                        "name": "Local Cache (Sequence Classification)",
                        "func": lambda: self._load_from_local_sequence_classification(model_path, torch_dtype)
                    },
                    {
                        "name": "Local Cache (CausalLM)",
                        "func": lambda: self._load_from_local_causal_lm(model_path, torch_dtype)
                    },
                    {
                        "name": "HuggingFace Hub (Sequence Classification)",
                        "func": lambda: self._load_from_hub_sequence_classification(torch_dtype)
                    },
                    {
                        "name": "HuggingFace Hub (CausalLM)", 
                        "func": lambda: self._load_from_hub_causal_lm(torch_dtype)
                    }
                ]
            
            model_loaded = False
            for attempt in loading_attempts:
                try:
                    print(f"[QwenReranker] Trying: {attempt['name']}")
                    self.model = attempt['func']()
                    model_loaded = True
                    print(f"[QwenReranker] Successfully loaded using: {attempt['name']}")
                    break
                except Exception as e:
                    print(f"[QwenReranker] {attempt['name']} failed: {str(e)[:100]}...")
                    continue
            
            if not model_loaded:
                raise Exception("All model loading attempts failed")
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"[QwenReranker] Model loaded successfully")
            
        except Exception as e:
            print(f"[QwenReranker] Error loading model: {e}")
            raise
    
    def _load_from_hub_sequence_classification(self, torch_dtype):
        """Load model from HuggingFace Hub as sequence classification"""
        from transformers import AutoConfig
        from app.core.reranker_config import RerankerConfig
        
        # Check if we should force local-only mode
        force_local_only = RerankerConfig.force_local_only()
        
        # Try 0.6B model first, then fallback to 4B
        model_names = ["Qwen/Qwen3-Reranker-0.6B", "Qwen/Qwen3-Reranker-4B"]
        
        for model_name in model_names:
            try:
                print(f"[QwenReranker] Attempting to load {model_name} from Hub")
                
                # Load config from hub and configure for sequence classification
                config = AutoConfig.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    local_files_only=force_local_only
                )
                config.num_labels = 1
                config.problem_type = "single_label_classification"
                config.pad_token_id = self.tokenizer.pad_token_id
                
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    config=config,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    ignore_mismatched_sizes=True,
                    low_cpu_mem_usage=True,
                    local_files_only=force_local_only
                )
                self.use_causal_lm = False
                print(f"[QwenReranker] Successfully loaded {model_name}")
                return model
            except Exception as e:
                print(f"[QwenReranker] Failed to load {model_name}: {str(e)[:100]}")
                continue
        
        raise Exception("Failed to load any Qwen reranker model from Hub")
    
    def _load_from_hub_causal_lm(self, torch_dtype):
        """Load model from HuggingFace Hub as causal LM"""
        from transformers import AutoModelForCausalLM
        from app.core.reranker_config import RerankerConfig
        
        # Check if we should force local-only mode
        force_local_only = RerankerConfig.force_local_only()
        
        # Try 0.6B model first, then fallback to 4B
        model_names = ["Qwen/Qwen3-Reranker-0.6B", "Qwen/Qwen3-Reranker-4B"]
        
        for model_name in model_names:
            try:
                print(f"[QwenReranker] Attempting to load {model_name} as CausalLM from Hub")
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=force_local_only
                )
                self.use_causal_lm = True
                print(f"[QwenReranker] Successfully loaded {model_name} as CausalLM")
                return model
            except Exception as e:
                print(f"[QwenReranker] Failed to load {model_name} as CausalLM: {str(e)[:100]}")
                continue
        
        raise Exception("Failed to load any Qwen reranker model as CausalLM from Hub")
    
    def _load_docker_safe(self, torch_dtype, model_name):
        """Docker-safe loading that properly handles local safetensors files"""
        from transformers import AutoConfig, AutoModelForSequenceClassification
        import torch
        
        try:
            print(f"[QwenReranker] Attempting Docker-safe loading for {model_name}")
            
            # For 0.6B model, try to load from local path first
            if "0.6B" in model_name:
                local_path = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/6e9e69830b95c52b5fd889b7690dda3329508de3"
                if os.path.exists(local_path):
                    print(f"[QwenReranker] Loading from local path: {local_path}")
                    
                    # Load config from local path
                    config = AutoConfig.from_pretrained(
                        local_path,
                        trust_remote_code=True,
                        local_files_only=True
                    )
                    config.num_labels = 1
                    config.problem_type = "single_label_classification"
                    config.pad_token_id = self.tokenizer.pad_token_id
                    
                    # Try to load with safetensors support
                    model = AutoModelForSequenceClassification.from_pretrained(
                        local_path,
                        config=config,
                        trust_remote_code=True,
                        torch_dtype=torch_dtype,
                        local_files_only=True,
                        low_cpu_mem_usage=True,
                        ignore_mismatched_sizes=True
                    )
                    
                    self.use_causal_lm = False
                    print(f"[QwenReranker] Successfully loaded from local safetensors: {local_path}")
                    return model
            
            # Fallback to Hub loading
            print(f"[QwenReranker] Local loading failed, falling back to Hub loading")
            raise Exception("Local loading failed, will try Hub")
            
        except Exception as e:
            print(f"[QwenReranker] Docker-safe local loading failed: {str(e)[:100]}")
            raise e
    
    def _load_tokenizer_docker_safe(self):
        """Docker-safe tokenizer loading that avoids ModelWrapper issues"""
        tokenizer_names = ["Qwen/Qwen3-Reranker-0.6B", "Qwen/Qwen3-Reranker-4B"]
        
        # Check if we should force local-only mode
        from app.core.reranker_config import RerankerConfig
        force_local_only = RerankerConfig.force_local_only()
        
        print(f"[QwenReranker] Local-only mode: {force_local_only}")
        
        for tokenizer_name in tokenizer_names:
            try:
                print(f"[QwenReranker] Attempting Docker-safe tokenizer loading: {tokenizer_name}")
                
                # Try loading with minimal options to avoid ModelWrapper issues
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    trust_remote_code=True,
                    use_fast=False,  # Use slow tokenizer to avoid issues
                    use_auth_token=False,
                    local_files_only=force_local_only
                )
                
                print(f"[QwenReranker] Successfully loaded tokenizer with Docker-safe method: {tokenizer_name}")
                return tokenizer
                
            except Exception as e:
                print(f"[QwenReranker] Docker-safe tokenizer loading failed for {tokenizer_name}: {str(e)[:100]}")
                
                # Try with even more basic settings
                try:
                    print(f"[QwenReranker] Trying basic tokenizer loading for {tokenizer_name}")
                    from transformers import AutoTokenizer
                    
                    # Load with minimal settings
                    tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_name,
                        use_fast=False,
                        trust_remote_code=False,  # Disable trust_remote_code
                        local_files_only=force_local_only
                    )
                    
                    print(f"[QwenReranker] Successfully loaded basic tokenizer: {tokenizer_name}")
                    return tokenizer
                    
                except Exception as basic_error:
                    print(f"[QwenReranker] Basic tokenizer loading also failed: {str(basic_error)[:100]}")
                    continue
        
        raise Exception("Failed to load tokenizer with Docker-safe methods")
    
    def _load_from_local_sequence_classification(self, model_path, torch_dtype):
        """Load model from local cache as sequence classification"""
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        config.num_labels = 1
        config.problem_type = "single_label_classification" 
        config.pad_token_id = self.tokenizer.pad_token_id
        
        # Try with different loading parameters to handle ModelWrapper error
        loading_kwargs = [
            # First attempt: Standard loading
            {
                "config": config,
                "trust_remote_code": True,
                "local_files_only": True,
                "torch_dtype": torch_dtype,
                "ignore_mismatched_sizes": True,
                "low_cpu_mem_usage": True
            },
            # Second attempt: Force PyTorch format (avoid safetensors)
            {
                "config": config,
                "trust_remote_code": True,
                "local_files_only": True,
                "torch_dtype": torch_dtype,
                "ignore_mismatched_sizes": True,
                "low_cpu_mem_usage": True,
                "use_safetensors": False
            },
            # Third attempt: No specific dtype
            {
                "config": config,
                "trust_remote_code": True,
                "local_files_only": True,
                "ignore_mismatched_sizes": True,
                "low_cpu_mem_usage": True,
                "use_safetensors": False
            }
        ]
        
        for i, kwargs in enumerate(loading_kwargs):
            try:
                print(f"[QwenReranker] Local sequence classification attempt {i+1}")
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, **kwargs
                )
                self.use_causal_lm = False
                return model
            except Exception as e:
                print(f"[QwenReranker] Local sequence classification attempt {i+1} failed: {str(e)[:100]}")
                continue
        
        raise Exception("All local sequence classification loading attempts failed")
    
    def _load_from_local_causal_lm(self, model_path, torch_dtype):
        """Load model from local cache as causal LM"""
        from transformers import AutoModelForCausalLM
        
        # Try with different loading parameters to handle ModelWrapper error
        loading_kwargs = [
            # First attempt: Standard loading
            {
                "trust_remote_code": True,
                "local_files_only": True,
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": True
            },
            # Second attempt: Force PyTorch format (avoid safetensors)
            {
                "trust_remote_code": True,
                "local_files_only": True,
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": True,
                "use_safetensors": False
            },
            # Third attempt: No specific dtype
            {
                "trust_remote_code": True,
                "local_files_only": True,
                "low_cpu_mem_usage": True,
                "use_safetensors": False
            }
        ]
        
        for i, kwargs in enumerate(loading_kwargs):
            try:
                print(f"[QwenReranker] Local CausalLM attempt {i+1}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, **kwargs
                )
                self.use_causal_lm = True
                return model
            except Exception as e:
                print(f"[QwenReranker] Local CausalLM attempt {i+1} failed: {str(e)[:100]}")
                continue
        
        raise Exception("All local CausalLM loading attempts failed")
    
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
                        padding="longest",
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                        pad_to_multiple_of=8
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
                    padding="longest",  # Explicit padding strategy
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                    pad_to_multiple_of=8  # Helps with efficiency
                ).to(self.device)
                
                # Get scores from the model
                if self.use_causal_lm:
                    # For CausalLM approach, use logits of "yes"/"no" tokens
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # Get "yes" and "no" token IDs (typical Qwen3 reranker approach)
                    # Token IDs for "yes" and "no" in Qwen tokenizer
                    try:
                        yes_token_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
                        no_token_id = self.tokenizer.encode("no", add_special_tokens=False)[0]
                        
                        # Extract logits for last token position for each sample
                        last_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
                        
                        # Get "yes" and "no" probabilities 
                        yes_logits = last_token_logits[:, yes_token_id]
                        no_logits = last_token_logits[:, no_token_id]
                        
                        # Calculate relevance scores (higher yes vs no ratio = more relevant)
                        scores = torch.softmax(torch.stack([no_logits, yes_logits], dim=1), dim=1)[:, 1]
                        scores = scores.cpu().numpy()
                        
                    except Exception as token_error:
                        print(f"[QwenReranker] Token-based scoring failed: {token_error}")
                        # Fallback: use simple hidden state similarity
                        hidden_states = outputs.last_hidden_state
                        # Use mean pooling of last hidden state as relevance score
                        scores = torch.mean(hidden_states, dim=1).norm(dim=1).cpu().numpy()
                        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                        
                else:
                    # Standard sequence classification approach
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # If binary classification, take the positive class score
                    if logits.shape[-1] == 2:
                        scores = logits[:, 1].cpu().numpy()  # Take positive class
                    else:
                        scores = logits.squeeze(-1).cpu().numpy()  # Single output
                    
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