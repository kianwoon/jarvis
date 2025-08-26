#!/usr/bin/env python3
"""
Test script to verify reranker local-only configuration is working correctly.
This script tests the configuration without actually loading models.
"""

import os
import sys

# Add the app directory to Python path
sys.path.insert(0, 'app')

def test_reranker_config():
    """Test the reranker configuration"""
    
    print("=== Testing Reranker Configuration ===")
    
    try:
        from app.core.reranker_config import RerankerConfig
        
        # Test local-only mode
        force_local = RerankerConfig.force_local_only()
        print(f"Force local-only mode: {force_local}")
        
        # Test model path detection
        model_path = RerankerConfig.get_model_path()
        print(f"Detected model path: {model_path}")
        
        if model_path and os.path.exists(model_path):
            print(f"✅ Model path exists: {model_path}")
        elif model_path:
            print(f"❌ Model path does not exist: {model_path}")
        else:
            print("❌ No model path detected")
        
        # Test device configuration
        device = RerankerConfig.get_device()
        print(f"Configured device: {device}")
        
        # Test other settings
        enabled = RerankerConfig.is_enabled()
        should_preload = RerankerConfig.should_preload()
        batch_size = RerankerConfig.get_batch_size()
        
        print(f"Reranker enabled: {enabled}")
        print(f"Should preload: {should_preload}")
        print(f"Batch size: {batch_size}")
        
        # Test environment variable setting
        print("\n=== Testing Environment Variables ===")
        original_transformers = os.environ.get("TRANSFORMERS_OFFLINE")
        original_hf_hub = os.environ.get("HF_HUB_OFFLINE")
        
        print(f"TRANSFORMERS_OFFLINE before: {original_transformers}")
        print(f"HF_HUB_OFFLINE before: {original_hf_hub}")
        
        # Test module import (this should set environment variables)
        print("\n=== Testing Module Import ===")
        from app.rag import qwen_reranker
        
        transformers_after = os.environ.get("TRANSFORMERS_OFFLINE")
        hf_hub_after = os.environ.get("HF_HUB_OFFLINE")
        
        print(f"TRANSFORMERS_OFFLINE after: {transformers_after}")
        print(f"HF_HUB_OFFLINE after: {hf_hub_after}")
        
        if force_local:
            if transformers_after == "1" and hf_hub_after == "1":
                print("✅ Environment variables set correctly for local-only mode")
            else:
                print("❌ Environment variables not set correctly")
        else:
            print("ℹ️  Local-only mode disabled, environment variables not changed")
        
        print("\n=== Configuration Summary ===")
        print(f"Status: {'✅ Ready for local-only operation' if force_local and model_path and os.path.exists(model_path) else '❌ Configuration incomplete'}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


if __name__ == "__main__":
    print("Testing Reranker Local Configuration...")
    success = test_reranker_config()
    sys.exit(0 if success else 1)