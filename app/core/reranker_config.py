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