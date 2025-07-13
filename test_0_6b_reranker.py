#!/usr/bin/env python3
"""
Test the updated Qwen3-Reranker with 0.6B model support
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_0_6b_reranker():
    """Test the 0.6B reranker implementation"""
    try:
        print("🔧 Testing Qwen3-Reranker-0.6B Implementation")
        print("=" * 60)
        
        # Clear any cached instance to test fresh loading
        from app.rag import qwen_reranker
        qwen_reranker._reranker_instance = None
        qwen_reranker._init_lock = False
        
        from app.rag.qwen_reranker import get_qwen_reranker
        
        # Get reranker instance
        print("🔄 Loading reranker (should prioritize 0.6B model)...")
        reranker = get_qwen_reranker()
        
        if reranker is None:
            print("❌ Failed to load reranker")
            return
            
        print("✅ Reranker loaded successfully")
        print(f"   Using CausalLM approach: {reranker.use_causal_lm}")
        print(f"   Device: {reranker.device}")
        
        # Create test documents
        test_query = "partnership between beyondsoft and tencent cloud computing"
        
        # Mock document objects
        class MockDoc:
            def __init__(self, content, metadata=None):
                self.page_content = content
                self.metadata = metadata or {}
        
        test_docs = [
            (MockDoc("Beyondsoft strategic partnership with Tencent Cloud for enterprise solutions"), 0.85),
            (MockDoc("Weather patterns and climate change in tropical regions"), 0.6),
            (MockDoc("Tencent Beyondsoft collaboration on AI and machine learning platforms"), 0.8),
            (MockDoc("Random content about cooking and recipe management systems"), 0.55),
            (MockDoc("Corporate development partnerships in technology sector"), 0.7)
        ]
        
        print(f"\n🔍 Testing reranking with {len(test_docs)} documents...")
        print(f"Query: '{test_query}'")
        
        # Perform reranking
        results = reranker.rerank(
            query=test_query,
            documents=test_docs,
            top_k=3
        )
        
        print(f"\n📊 Reranking Results (0.6B Model):")
        for i, result in enumerate(results):
            relevance_keywords = ['tencent', 'beyondsoft', 'partnership', 'collaboration']
            content_lower = result.document.page_content.lower()
            relevance_count = sum(1 for keyword in relevance_keywords if keyword in content_lower)
            relevance = "high" if relevance_count >= 2 else "medium" if relevance_count >= 1 else "low"
            
            print(f"  {i+1}. Score: {result.score:.4f} (original: {result.original_score:.2f}) | Relevance: {relevance}")
            print(f"     Content: {result.document.page_content[:100]}...")
            print()
        
        # Check if relevance ranking is correct
        if len(results) >= 2:
            top_result = results[0]
            top_content = top_result.document.page_content.lower()
            is_top_relevant = any(word in top_content for word in ['tencent', 'beyondsoft', 'partnership'])
            
            if is_top_relevant:
                print("✅ 0.6B model correctly prioritized relevant content")
                
                # Check if score distribution looks reasonable
                scores = [r.score for r in results]
                score_range = max(scores) - min(scores)
                print(f"📈 Score distribution:")
                print(f"   Highest: {max(scores):.4f}")
                print(f"   Lowest: {min(scores):.4f}")
                print(f"   Range: {score_range:.4f}")
                
                if score_range > 0.1:
                    print("✅ Good score differentiation between documents")
                else:
                    print("⚠️  Limited score differentiation - may need tuning")
            else:
                print("⚠️  0.6B model ranked irrelevant content highest - may need instruction tuning")
        
        # Test with custom instruction
        print(f"\n🎯 Testing with custom instruction...")
        custom_instruction = "Given a query about business partnerships and a document, predict whether the document contains relevant information about the specific partnership mentioned in the query."
        
        custom_results = reranker.rerank(
            query=test_query,
            documents=test_docs[:3],  # Test with fewer docs for speed
            instruction=custom_instruction,
            top_k=2
        )
        
        print(f"Custom instruction results:")
        for i, result in enumerate(custom_results):
            print(f"  {i+1}. Score: {result.score:.4f}")
            print(f"     Content: {result.document.page_content[:80]}...")
        
        print("\n✅ 0.6B reranker test completed successfully")
        print("\n💡 Benefits of 0.6B model:")
        print("   - Faster loading and inference")
        print("   - Lower memory usage") 
        print("   - Good performance for most reranking tasks")
        print("   - Better compatibility with various environments")
        
    except Exception as e:
        print(f"❌ 0.6B reranker test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_0_6b_reranker()