#!/usr/bin/env python3
"""
Test the fallback reranker when model loading fails
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_fallback_reranker():
    """Test fallback reranker functionality"""
    try:
        print("🔧 Testing Fallback Reranker")
        print("=" * 60)
        
        # Clear cached instance to test fresh loading
        from app.rag import qwen_reranker
        qwen_reranker._reranker_instance = None
        qwen_reranker._init_lock = False
        
        # Simulate Docker environment to trigger fallback
        os.environ['DOCKER_ENVIRONMENT'] = '1'
        
        from app.rag.qwen_reranker import get_qwen_reranker
        
        print("🔄 Loading reranker (should fallback when model fails)...")
        reranker = get_qwen_reranker()
        
        if reranker is None:
            print("❌ Failed to load reranker")
            return
            
        print("✅ Reranker loaded successfully")
        print(f"   Using fallback: {getattr(reranker, 'use_fallback', False)}")
        print(f"   Device: {reranker.device}")
        
        # Create test documents
        test_query = "partnership between beyondsoft and alibaba cloud services"
        
        # Mock document objects
        class MockDoc:
            def __init__(self, content, metadata=None):
                self.page_content = content
                self.metadata = metadata or {}
        
        test_docs = [
            (MockDoc("Beyondsoft strategic partnership with Alibaba Cloud for enterprise solutions and services"), 0.8),
            (MockDoc("Weather forecast sunny cloudy conditions today"), 0.6),
            (MockDoc("Random cooking recipe with ingredients and preparation steps"), 0.55),
            (MockDoc("Alibaba partnership development with Beyondsoft teams"), 0.75),
            (MockDoc("Technical documentation for cloud services infrastructure"), 0.7)
        ]
        
        print(f"\n🔍 Testing fallback reranking with {len(test_docs)} documents...")
        print(f"Query: '{test_query}'")
        
        # Perform reranking
        results = reranker.rerank(
            query=test_query,
            documents=test_docs,
            top_k=3
        )
        
        print(f"\n📊 Fallback Reranking Results:")
        for i, result in enumerate(results):
            relevance_keywords = ['beyondsoft', 'alibaba', 'partnership', 'cloud', 'services']
            content_lower = result.document.page_content.lower()
            keyword_matches = [kw for kw in relevance_keywords if kw in content_lower]
            
            print(f"  {i+1}. Score: {result.score:.4f} (original: {result.original_score:.2f})")
            print(f"     Keyword matches: {keyword_matches}")
            print(f"     Metadata: {result.metadata}")
            print(f"     Content: {result.document.page_content[:80]}...")
            print()
        
        # Check if relevance ranking makes sense
        if len(results) >= 2:
            top_result = results[0]
            top_content = top_result.document.page_content.lower()
            
            # Check if top result contains relevant keywords
            relevant_keywords = ['beyondsoft', 'alibaba', 'partnership']
            matches = sum(1 for kw in relevant_keywords if kw in top_content)
            
            if matches >= 2:
                print("✅ Fallback reranker correctly prioritized relevant content")
                print(f"   Top result has {matches} relevant keyword matches")
            else:
                print("⚠️  Fallback reranker ranking could be improved")
                print(f"   Top result has only {matches} relevant keyword matches")
        
        print(f"\n🎯 Fallback System Benefits:")
        print("   - Works when transformers library doesn't support qwen3")
        print("   - Provides basic keyword-based reranking")
        print("   - Better than no reranking at all")
        print("   - Fast and lightweight")
        
        # Clean up environment variable
        del os.environ['DOCKER_ENVIRONMENT']
        
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fallback_reranker()