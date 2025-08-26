"""
Reranker configuration based on environment
"""

import os
from typing import Optional


class RerankerConfig:
    """Configuration for Qwen3-Reranker-4B based on environment"""
    
    @staticmethod
    def is_enabled() -> bool:
        """Check if reranker should be enabled"""
        # Check environment variable
        enabled = os.environ.get("ENABLE_QWEN_RERANKER", "auto").lower()
        
        if enabled == "false":
            return False
        elif enabled == "true":
            return True
        else:  # auto mode
            # Disable in Docker by default (CPU only)
            if os.path.exists("/.dockerenv"):
                return False
            # Enable locally (has M3 GPU)
            return True
    
    @staticmethod
    def should_preload() -> bool:
        """Check if reranker should be preloaded at startup"""
        # Only preload if enabled and explicitly requested
        if not RerankerConfig.is_enabled():
            return False
        
        preload = os.environ.get("PRELOAD_QWEN_RERANKER", "false").lower()
        return preload == "true"
    
    @staticmethod
    def get_device() -> Optional[str]:
        """Get the device to use for reranker"""
        device = os.environ.get("QWEN_RERANKER_DEVICE", None)
        return device
    
    @staticmethod
    def get_batch_size() -> int:
        """Get batch size for reranking"""
        try:
            return int(os.environ.get("QWEN_RERANKER_BATCH_SIZE", "10"))
        except ValueError:
            return 10
    
    @staticmethod
    def force_local_only() -> bool:
        """Check if reranker should be forced to use local files only"""
        # Environment variables to force local-only mode
        transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE", "0").lower() in ["1", "true"]
        hf_hub_offline = os.environ.get("HF_HUB_OFFLINE", "0").lower() in ["1", "true"]
        force_local = os.environ.get("QWEN_RERANKER_LOCAL_ONLY", "true").lower() in ["1", "true"]
        
        # Default to local-only for security and consistency
        return transformers_offline or hf_hub_offline or force_local
    
    @staticmethod
    def get_model_path() -> Optional[str]:
        """Get the configured model path for reranker"""
        # Check environment variable first
        model_path = os.environ.get("QWEN_RERANKER_MODEL_PATH")
        
        if model_path is None:
            # Try to detect local model paths for the user's environment
            possible_paths = [
                # User's specific 0.6B model path (preferred - smaller, faster)
                os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/6e9e69830b95c52b5fd889b7690dda3329508de3"),
                os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B"),
                # User's specific 4B model path (fallback)
                os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B"),
                # Docker paths (fallback)
                "/root/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/6e9e69830b95c52b5fd889b7690dda3329508de3",
                "/root/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B",
                "/root/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B",
            ]
            
            # Use the first path that exists
            for path in possible_paths:
                if os.path.exists(path):
                    return path
        
        return model_path