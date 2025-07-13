#!/usr/bin/env python3
"""
Debug script to investigate Qwen3-Reranker-4B loading issues
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def debug_reranker_model():
    """Debug the reranker model loading issue"""
    try:
        print("🔍 Debugging Qwen3-Reranker-4B Loading Issue")
        print("=" * 60)
        
        # Check model path
        model_path = os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B/snapshots/57906229d41697e4494d50ca5859598cf86154a1"
        )
        
        print(f"Model path: {model_path}")
        print(f"Path exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            files = os.listdir(model_path)
            print(f"Files in model directory: {files}")
            
            # Check for key files
            key_files = ['config.json', 'pytorch_model.bin', 'model.safetensors', 'tokenizer_config.json']
            for file in key_files:
                file_path = os.path.join(model_path, file)
                exists = os.path.exists(file_path)
                print(f"  {file}: {'✅' if exists else '❌'}")
                
            # Try to read config.json to understand the model structure
            config_path = os.path.join(model_path, 'config.json')
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                print(f"\n📄 Model config preview:")
                for key, value in list(config.items())[:10]:
                    print(f"  {key}: {value}")
        
        # Test basic transformers import
        print(f"\n🔍 Testing transformers import...")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print("✅ Transformers import successful")
        
        # Test torch
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        print(f"✅ MPS available: {torch.backends.mps.is_available()}")
        
        # Try alternative model loading approach
        print(f"\n🔍 Attempting direct model loading...")
        try:
            # Try with different parameters
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-Reranker-4B",
                trust_remote_code=True
            )
            print("✅ Tokenizer loaded from HuggingFace Hub")
            
            # Try loading model with different settings
            model = AutoModelForSequenceClassification.from_pretrained(
                "Qwen/Qwen3-Reranker-4B",
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="auto"
            )
            print("✅ Model loaded from HuggingFace Hub")
            
        except Exception as e:
            print(f"❌ Direct loading failed: {e}")
            
            # Try without trust_remote_code
            try:
                print(f"\n🔍 Trying without trust_remote_code...")
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-4B")
                model = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen3-Reranker-4B")
                print("✅ Model loaded without trust_remote_code")
            except Exception as e2:
                print(f"❌ Loading without trust_remote_code failed: {e2}")
        
        # Check current reranker implementation
        print(f"\n🔍 Testing current QwenReranker implementation...")
        try:
            from app.rag.qwen_reranker import get_qwen_reranker
            reranker = get_qwen_reranker()
            if reranker:
                print("✅ QwenReranker loaded successfully")
            else:
                print("❌ QwenReranker returned None")
        except Exception as e:
            print(f"❌ QwenReranker failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Debug script error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_reranker_model()