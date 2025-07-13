#!/usr/bin/env python3
"""
Debug script to understand why n8n documents score highly for Tencent partnership queries
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def debug_relevance_scoring():
    """Debug the relevance scoring mechanism"""
    try:
        print("🔍 Debugging Relevance Scoring for Tencent vs n8n Content")
        print("=" * 70)
        
        from app.langchain.service import calculate_relevance_score
        
        # Test query
        query = "partnership between beyondsoft and tencent"
        
        # Mock document contents to test scoring
        test_docs = {
            "Tencent Partnership": """
            Beyondsoft's Partnership with Tencent started in 2012 with cloud infrastructure support.
            The collaboration focuses on cloud computing, AI development, and enterprise solutions.
            Tencent Cloud development teams work with Beyondsoft on container platforms and DevOps.
            """,
            "n8n AI Agent": """
            n8n ai agent messages workflow automation platform for connecting different services.
            The n8n platform enables users to build automated workflows with visual interface.
            AI agents can be configured to handle various tasks and messaging integrations.
            """,
            "Random Content": """
            This is completely unrelated content about weather patterns and climate change.
            No mention of partnerships, companies, or technology platforms here.
            Just general information about environmental topics and sustainability.
            """
        }
        
        print(f"Query: '{query}'")
        print(f"Query terms after processing: {query.lower().split()}")
        print()
        
        # Test relevance scoring for each document
        for doc_name, content in test_docs.items():
            print(f"📄 Testing: {doc_name}")
            print(f"Content preview: {content.strip()[:100]}...")
            
            # Calculate relevance score
            score = calculate_relevance_score(query, content)
            print(f"Relevance Score: {score:.4f}")
            
            # Analyze token overlap
            query_tokens = set(query.lower().split())
            content_tokens = set(content.lower().split())
            overlap = query_tokens.intersection(content_tokens)
            print(f"Token overlap: {overlap}")
            print(f"Overlap ratio: {len(overlap)} / {len(query_tokens)} = {len(overlap)/len(query_tokens):.2f}")
            print("-" * 50)
        
        # Test BM25 processor directly
        print("\n🧮 Testing BM25 Processor Directly")
        print("=" * 50)
        
        from app.rag.bm25_processor import BM25Processor
        processor = BM25Processor()
        
        for doc_name, content in test_docs.items():
            print(f"\n📄 BM25 Analysis: {doc_name}")
            
            # Get cleaned tokens
            query_tokens = processor.tokenize_and_clean(query)
            doc_tokens = processor.tokenize_and_clean(content)
            
            print(f"Query tokens: {query_tokens}")
            print(f"Doc tokens: {doc_tokens[:10]}...")  # First 10 tokens
            
            # Calculate term frequencies
            doc_term_freq = processor.calculate_term_frequencies(content)
            matching_terms = {term: freq for term, freq in doc_term_freq.items() if term in query_tokens}
            
            print(f"Matching terms: {matching_terms}")
            
            # Calculate enhanced score
            enhanced_score = processor.enhance_existing_relevance_score(
                query=query,
                content=content,
                corpus_stats=None,  # No corpus stats available
                existing_score=0.0
            )
            print(f"Enhanced BM25 score: {enhanced_score:.4f}")
            print("-" * 30)
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_relevance_scoring()