#!/usr/bin/env python3
"""
Test the fixed Qwen3-Reranker-4B implementation
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_fixed_reranker():
    """Test the fixed reranker implementation"""
    try:
        print("🔧 Testing Fixed Qwen3-Reranker-4B Implementation")
        print("=" * 60)
        
        from app.rag.qwen_reranker import get_qwen_reranker
        
        # Get reranker instance
        print("🔄 Loading reranker...")
        reranker = get_qwen_reranker()
        
        if reranker is None:
            print("❌ Failed to load reranker")
            return
            
        print("✅ Reranker loaded successfully")
        print(f"   Using CausalLM approach: {reranker.use_causal_lm}")
        print(f"   Device: {reranker.device}")
        
        # Create test documents
        test_query = "partnership between beyondsoft and tencent"
        
        # Mock document objects that match the expected structure
        class MockDoc:
            def __init__(self, content):
                self.page_content = content
        
        test_docs = [
            (MockDoc("Beyondsoft's Partnership with Tencent started in 2012 with cloud infrastructure support"), 0.8),
            (MockDoc("Tencent Cloud development teams work with Beyondsoft on container platforms"), 0.75),
            (MockDoc("Random document about weather patterns in Asia"), 0.6),
            (MockDoc("Corporate Development Group manages strategic partnerships"), 0.7),
            (MockDoc("Alibaba Cloud collaboration with various partners"), 0.65)
        ]
        
        print(f"\n🔍 Testing reranking with {len(test_docs)} documents...")
        print(f"Query: {test_query}")
        
        # Perform reranking
        results = reranker.rerank(
            query=test_query,
            documents=test_docs,
            top_k=3
        )
        
        print(f"\n📊 Reranking Results:")
        for i, result in enumerate(results):
            print(f"  {i+1}. Score: {result.score:.4f} (original: {result.original_score:.2f})")
            print(f"     Content: {result.document.page_content[:100]}...")
            print()
        
        # Check if relevance scores make sense
        if len(results) >= 2:
            tencent_results = [r.score for r in results if 'tencent' in r.document.page_content.lower()]
            weather_results = [r.score for r in results if 'weather' in r.document.page_content.lower()]
            
            if tencent_results and weather_results:
                tencent_score = max(tencent_results)
                weather_score = max(weather_results)
                
                if tencent_score > weather_score:
                    print("✅ Relevance scoring works correctly (Tencent content scored higher than weather)")
                else:
                    print("⚠️  Relevance scoring may need adjustment")
            elif tencent_results:
                print("✅ Found Tencent content in results")
            else:
                print("ℹ️  No specific content types found for comparison")
        
        print("✅ Reranker test completed successfully")
        
    except Exception as e:
        print(f"❌ Reranker test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_reranker()