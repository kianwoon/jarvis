#!/usr/bin/env python3
"""
Test loading the 0.6B model from local snapshot directory
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_local_0_6b():
    """Test loading 0.6B model from local snapshot"""
    try:
        print("🔧 Testing Local Qwen3-Reranker-0.6B Loading")
        print("=" * 60)
        
        # Clear cached instance
        from app.rag import qwen_reranker
        qwen_reranker._reranker_instance = None
        qwen_reranker._init_lock = False
        
        # Check if snapshot path exists
        snapshot_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/6e9e69830b95c52b5fd889b7690dda3329508de3")
        print(f"Checking snapshot path: {snapshot_path}")
        print(f"Path exists: {os.path.exists(snapshot_path)}")
        
        if os.path.exists(snapshot_path):
            # List files in snapshot
            files = os.listdir(snapshot_path)
            print(f"Files in snapshot: {files}")
            
            # Check for required model files
            required_files = ['config.json', 'tokenizer.json', 'model.safetensors']
            for file in required_files:
                exists = file in files
                print(f"  {file}: {'✅' if exists else '❌'}")
        
        print("\n🔄 Loading reranker from local snapshot...")
        
        from app.rag.qwen_reranker import get_qwen_reranker
        reranker = get_qwen_reranker()
        
        if reranker:
            print("✅ Successfully loaded reranker from local files")
            print(f"   Device: {reranker.device}")
            print(f"   Using CausalLM: {reranker.use_causal_lm}")
            
            # Quick test
            class MockDoc:
                def __init__(self, content):
                    self.page_content = content
            
            test_docs = [
                (MockDoc("Tencent partnership collaboration"), 0.8),
                (MockDoc("Weather forecast today"), 0.6)
            ]
            
            results = reranker.rerank("tencent partnership", test_docs, top_k=2)
            print(f"\nQuick test results:")
            for i, result in enumerate(results):
                print(f"  {i+1}. Score: {result.score:.4f} - {result.document.page_content}")
            
            print("✅ Local 0.6B model working correctly")
        else:
            print("❌ Failed to load reranker")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_local_0_6b()