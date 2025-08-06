#!/usr/bin/env python3
"""
Direct test of RAG enhanced fallback to identify issues
"""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_fallback():
    """Test the enhanced RAG fallback directly"""
    
    try:
        from app.core.rag_enhanced_fallback import enhanced_rag_search
        
        query = "partnership between BeyondSoft and Alibaba"
        collections = ["partnership", "default_knowledge"]
        
        print(f"Testing RAG fallback with:")
        print(f"  Query: {query}")
        print(f"  Collections: {collections}")
        
        result = enhanced_rag_search(
            query=query,
            collections=collections,
            max_documents=5,
            include_content=True
        )
        
        print(f"\nRAG Fallback Result:")
        print(f"  Success: {result.get('success', False)}")
        print(f"  Error: {result.get('error', 'None')}")
        print(f"  Documents returned: {result.get('documents_returned', 0)}")
        print(f"  Total documents found: {result.get('total_documents_found', 0)}")
        print(f"  Collections searched: {result.get('collections_searched', [])}")
        print(f"  Execution time: {result.get('execution_time_ms', 0)}ms")
        
        if result.get('documents'):
            print(f"\nFirst document preview:")
            doc = result['documents'][0]
            print(f"  Title: {doc.get('title', 'No title')}")
            print(f"  Content: {doc.get('content', 'No content')[:200]}...")
            print(f"  Score: {doc.get('score', 'No score')}")
        
        return result
        
    except Exception as e:
        print(f"ERROR testing RAG fallback: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_rag_fallback()