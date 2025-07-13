#!/usr/bin/env python3
"""
Debug script to investigate partnership query specificity issues
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def debug_partnership_specificity():
    """Debug why Tencent queries are returning both Tencent and Alibaba documents"""
    
    try:
        from app.core.vector_db_settings_cache import get_vector_db_settings
        from app.core.rag_settings_cache import get_rag_settings
        from pymilvus import Collection, connections
        import re
        
        print("=== PARTNERSHIP QUERY SPECIFICITY DEBUG ===")
        print("Investigating why both Tencent and Alibaba docs are retrieved for Tencent queries")
        print()
        
        # Get settings
        vector_db_settings = get_vector_db_settings()
        rag_settings = get_rag_settings()
        
        # Get Milvus config
        milvus_config = None
        if "milvus" in vector_db_settings:
            milvus_config = vector_db_settings["milvus"]
            uri = milvus_config.get("MILVUS_URI")
            token = milvus_config.get("MILVUS_TOKEN")
        else:
            for db in vector_db_settings.get("databases", []):
                if db.get("id") == "milvus" and db.get("enabled", False):
                    milvus_config = db.get("config", {})
                    break
            
            uri = milvus_config.get("MILVUS_URI") if milvus_config else None
            token = milvus_config.get("MILVUS_TOKEN") if milvus_config else None
        
        if not uri:
            print("No Milvus URI found")
            return
            
        connections.connect(uri=uri, token=token, alias="debug_connection")
        collection = Collection("partnership", using="debug_connection")
        collection.load()
        
        # Test query that should be specific to Tencent
        test_query = "relationship between beyondsoft and tencent in details"
        print(f"Testing query: '{test_query}'")
        print()
        
        # Step 1: Analyze keyword content in both PDFs
        print("1. KEYWORD ANALYSIS:")
        
        # Get all Tencent documents
        tencent_docs = collection.query(
            expr='source like "%tencent%"',
            output_fields=["content", "source", "doc_id"],
            limit=20
        )
        
        # Get all Alibaba documents 
        alibaba_docs = collection.query(
            expr='source like "%alibaba%"',
            output_fields=["content", "source", "doc_id"],
            limit=20
        )
        
        print(f"   Tencent document chunks: {len(tencent_docs)}")
        print(f"   Alibaba document chunks: {len(alibaba_docs)}")
        
        # Analyze keyword overlap
        def extract_keywords(docs):
            all_text = " ".join([doc.get('content', '') for doc in docs]).lower()
            # Extract common business/partnership terms
            words = re.findall(r'\b\w+\b', all_text)
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        tencent_keywords = extract_keywords(tencent_docs)
        alibaba_keywords = extract_keywords(alibaba_docs)
        
        print("\n   Top Tencent keywords:")
        for word, freq in tencent_keywords[:10]:
            print(f"     {word}: {freq}")
            
        print("\n   Top Alibaba keywords:")
        for word, freq in alibaba_keywords[:10]:
            print(f"     {word}: {freq}")
        
        # Find common keywords that might cause confusion
        tencent_words = {word for word, _ in tencent_keywords}
        alibaba_words = {word for word, _ in alibaba_keywords}
        common_words = tencent_words.intersection(alibaba_words)
        
        print(f"\n   Common keywords causing confusion: {list(common_words)[:10]}")
        
        # Step 2: Test simple keyword filtering
        print("\n2. KEYWORD FILTERING TEST:")
        
        # Count mentions of specific company names
        def count_company_mentions(docs, companies):
            counts = {}
            for company in companies:
                counts[company] = 0
                for doc in docs:
                    content = doc.get('content', '').lower()
                    counts[company] += content.count(company.lower())
            return counts
        
        tencent_mentions = count_company_mentions(tencent_docs, ['tencent', 'alibaba', 'beyondsoft'])
        alibaba_mentions = count_company_mentions(alibaba_docs, ['tencent', 'alibaba', 'beyondsoft'])
        
        print(f"   In Tencent docs - Tencent: {tencent_mentions['tencent']}, Alibaba: {tencent_mentions['alibaba']}")
        print(f"   In Alibaba docs - Tencent: {alibaba_mentions['tencent']}, Alibaba: {alibaba_mentions['alibaba']}")
        
        # Step 3: Test query-specific filtering
        print("\n3. QUERY-SPECIFIC CONTENT ANALYSIS:")
        
        query_terms = ['tencent', 'relationship', 'beyondsoft', 'details']
        
        def score_document_relevance(doc, query_terms):
            content = doc.get('content', '').lower()
            score = 0
            for term in query_terms:
                score += content.count(term.lower()) * (2 if term in ['tencent'] else 1)
            return score
        
        print("\n   Tencent document relevance scores:")
        for i, doc in enumerate(tencent_docs[:5]):
            score = score_document_relevance(doc, query_terms)
            print(f"     Doc {i+1}: Score {score}, Content: {doc.get('content', '')[:150]}...")
        
        print("\n   Alibaba document relevance scores:")
        for i, doc in enumerate(alibaba_docs[:5]):
            score = score_document_relevance(doc, query_terms)
            print(f"     Doc {i+1}: Score {score}, Content: {doc.get('content', '')[:150]}...")
        
        # Step 4: Check current similarity threshold
        print(f"\n4. CURRENT RAG SETTINGS:")
        doc_retrieval = rag_settings.get('document_retrieval', {})
        print(f"   Similarity threshold: {doc_retrieval.get('similarity_threshold', 'NOT SET')}")
        print(f"   Max documents: {doc_retrieval.get('max_documents_mcp', 'NOT SET')}")
        print(f"   Num docs retrieve: {doc_retrieval.get('num_docs_retrieve', 'NOT SET')}")
        
        # Step 5: Suggest improvements
        print(f"\n5. RECOMMENDATIONS:")
        print(f"   1. Lower similarity threshold to be more restrictive")
        print(f"   2. Add query-specific term weighting (boost 'tencent' matches)")
        print(f"   3. Add cross-document penalty (penalize docs from different partnerships)")
        print(f"   4. Implement query term filtering before context building")
        print(f"   5. Add semantic filtering to ensure query intent matching")
        
        connections.disconnect(alias="debug_connection")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_partnership_specificity())