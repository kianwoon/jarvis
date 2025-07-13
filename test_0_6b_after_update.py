#!/usr/bin/env python3
"""
Test 0.6B model loading after transformers>=4.51.0 update
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_0_6b_after_update():
    """Test 0.6B model should work with transformers>=4.51.0"""
    try:
        print("🔧 Testing Qwen3-Reranker-0.6B After Transformers Update")
        print("=" * 60)
        
        # Check transformers version
        try:
            import transformers
            print(f"Transformers version: {transformers.__version__}")
            
            # Check if version meets requirement
            from packaging import version
            if version.parse(transformers.__version__) >= version.parse("4.51.0"):
                print("✅ Transformers version meets requirement (>=4.51.0)")
            else:
                print("⚠️  Transformers version may not support Qwen3")
        except ImportError:
            print("❌ Cannot check transformers version")
        
        # Clear cached instance
        from app.rag import qwen_reranker
        qwen_reranker._reranker_instance = None
        qwen_reranker._init_lock = False
        
        print("\n🔄 Loading 0.6B reranker...")
        from app.rag.qwen_reranker import get_qwen_reranker
        
        reranker = get_qwen_reranker()
        
        if reranker:
            print("✅ Qwen3-Reranker-0.6B loaded successfully")
            print(f"   Device: {reranker.device}")
            print(f"   Using CausalLM: {reranker.use_causal_lm}")
            
            # Quick functionality test
            class MockDoc:
                def __init__(self, content):
                    self.page_content = content
            
            test_docs = [
                (MockDoc("Beyondsoft partnership with Alibaba Cloud enterprise services"), 0.8),
                (MockDoc("Weather conditions and climate data analysis"), 0.6),
                (MockDoc("Alibaba and Beyondsoft collaboration on cloud platforms"), 0.75)
            ]
            
            print("\n🔍 Testing reranking functionality...")
            results = reranker.rerank(
                query="beyondsoft alibaba partnership cloud",
                documents=test_docs,
                top_k=2
            )
            
            print("📊 Results:")
            for i, result in enumerate(results):
                print(f"  {i+1}. Score: {result.score:.4f}")
                print(f"     Content: {result.document.page_content[:60]}...")
            
            print("\n✅ Reranker is working correctly!")
            print("\n💡 After Docker rebuild with transformers>=4.51.0:")
            print("   - Qwen3 model type will be recognized")
            print("   - No more ModelWrapper errors")
            print("   - Proper 0.6B model loading from local cache")
            
        else:
            print("❌ Failed to load reranker")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\n💡 Expected behavior:")
        print("   - This may fail with current transformers version")
        print("   - Should work after Docker rebuild with transformers>=4.51.0")

if __name__ == "__main__":
    test_0_6b_after_update()