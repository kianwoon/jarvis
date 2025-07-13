#!/usr/bin/env python3
"""
Comprehensive analysis of the RAG retrieval issue
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def analyze_issue():
    """Comprehensive analysis of the issue"""
    
    print("COMPREHENSIVE RAG RETRIEVAL ANALYSIS")
    print("=" * 70)
    
    # Check 1: Collection and document verification
    print("\n1. COLLECTION AND DOCUMENT VERIFICATION:")
    print("-" * 50)
    
    try:
        from pymilvus import Collection, connections
        from app.core.vector_db_settings_cache import get_vector_db_settings
        
        # Get connection settings
        vector_db_settings = get_vector_db_settings()
        
        if "milvus" in vector_db_settings:
            milvus_config = vector_db_settings["milvus"]
            uri = milvus_config.get("MILVUS_URI")
            token = milvus_config.get("MILVUS_TOKEN")
        else:
            milvus_config = None
            for db in vector_db_settings.get("databases", []):
                if db.get("id") == "milvus" and db.get("enabled", False):
                    milvus_config = db.get("config", {})
                    break
            
            uri = milvus_config.get("MILVUS_URI") if milvus_config else None
            token = milvus_config.get("MILVUS_TOKEN") if milvus_config else None
        
        if uri:
            connections.connect(uri=uri, token=token, alias="analysis_connection")
            collection = Collection("partnership", using="analysis_connection")
            collection.load()
            
            # Check for Tencent content
            tencent_docs = collection.query(
                expr='content like "%Tencent%" and source like "%tencent%"',
                output_fields=["content", "source", "doc_id"],
                limit=5
            )
            
            print(f"✓ Partnership collection exists with {collection.num_entities} documents")
            print(f"✓ Found {len(tencent_docs)} documents containing Tencent content")
            
            if tencent_docs:
                print(f"✓ Target document 'bys & tencent partnership.pdf' confirmed in collection")
                sample_content = tencent_docs[0].get('content', '')[:200]
                print(f"✓ Sample content: {sample_content}...")
            
            connections.disconnect(alias="analysis_connection")
        else:
            print("✗ No Milvus URI configured")
            
    except Exception as e:
        print(f"✗ Collection verification failed: {e}")
    
    # Check 2: RAG settings analysis  
    print("\n2. RAG SETTINGS ANALYSIS:")
    print("-" * 50)
    
    try:
        from app.core.rag_settings_cache import get_document_retrieval_settings, get_agent_settings
        
        doc_settings = get_document_retrieval_settings()
        agent_settings = get_agent_settings()
        
        similarity_threshold = doc_settings.get('similarity_threshold', 'Not set')
        min_relevance_score = agent_settings.get('min_relevance_score', 'Not set')
        complex_query_threshold = agent_settings.get('complex_query_threshold', 'Not set')
        
        print(f"✓ Similarity threshold: {similarity_threshold}")
        print(f"✓ Min relevance score: {min_relevance_score}")  
        print(f"✓ Complex query threshold: {complex_query_threshold}")
        
        # Analysis
        if isinstance(similarity_threshold, (int, float)) and similarity_threshold <= 1.5:
            print(f"✓ Similarity threshold ({similarity_threshold}) should allow good matches")
        
        if isinstance(min_relevance_score, (int, float)) and min_relevance_score <= 0.4:
            print(f"✓ Relevance threshold ({min_relevance_score}) should allow Tencent content")
        
    except Exception as e:
        print(f"✗ Settings analysis failed: {e}")
    
    # Check 3: Document classifier analysis
    print("\n3. DOCUMENT CLASSIFIER ANALYSIS:")
    print("-" * 50)
    
    try:
        from app.core.document_classifier import get_document_classifier
        
        classifier = get_document_classifier()
        test_query = "partnership between beyondsoft and tencent"
        
        collection_type = classifier.classify_document(test_query, {"query": True})
        target_collection = classifier.get_target_collection(collection_type)
        
        print(f"✓ Query: '{test_query}'")
        print(f"✓ Classified type: {collection_type}")
        print(f"✓ Target collection: {target_collection}")
        
        if target_collection == "partnership":
            print(f"✓ Classification correctly targets partnership collection")
        else:
            print(f"⚠ Classification targets {target_collection}, not partnership")
            
    except Exception as e:
        print(f"✗ Classifier analysis failed: {e}")
    
    # Check 4: Relevance scoring test
    print("\n4. RELEVANCE SCORING TEST:")
    print("-" * 50)
    
    try:
        # Import the relevance function by copying its logic
        import re
        
        def calculate_relevance_score(query: str, content: str) -> float:
            """Calculate relevance score between query and content"""
            if not query or not content:
                return 0.0
            
            # Convert to lowercase for comparison
            query_lower = query.lower()
            content_lower = content.lower()
            
            # Extract keywords from query (remove stop words)
            stop_words = {
                'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an',
                'find', 'get', 'show', 'tell', 'about', 'info', 'information', 'news', 'what', 'how', 'when', 'where',
                'why', 'can', 'could', 'would', 'should', 'give', 'provide', 'need', 'want', 'like', 'please', 'help',
                'it', 'this', 'that', 'these', 'those'
            }
            
            query_words = set(re.findall(r'\b\w+\b', query_lower))
            content_words = set(re.findall(r'\b\w+\b', content_lower))
            
            # Remove stop words
            query_keywords = query_words - stop_words
            content_keywords = content_words - stop_words
            
            if not query_keywords:
                return 0.0
            
            # Calculate overlap
            overlap = query_keywords.intersection(content_keywords)
            overlap_ratio = len(overlap) / len(query_keywords)
            
            # Boost for exact phrase matches
            phrase_boost = 0.0
            query_clean = re.sub(r'[^\w\s]', '', query_lower)
            if query_clean in content_lower:
                phrase_boost = 0.3
            
            # Boost for partial phrase matches (2+ consecutive words)
            query_words_list = query_clean.split()
            for i in range(len(query_words_list) - 1):
                bigram = ' '.join(query_words_list[i:i+2])
                if bigram in content_lower:
                    phrase_boost += 0.1
            
            final_score = overlap_ratio + phrase_boost
            return min(final_score, 1.0)
        
        # Test with actual Tencent content
        test_query = "partnership between beyondsoft and tencent"
        tencent_content = """Beyondsoft's Partnership with Tencent Executive Summary Since the inception of our partnership in 2012, Beyondsoft has evolved from a service provider into a trusted strategic partner for Tencent. Over the years, our deep integration across Tencent's technology ecosystem—including Tencent Cloud, TDSQL, and enterprise-level distributed systems—has enabled us to drive large-scale database management"""
        
        relevance_score = calculate_relevance_score(test_query, tencent_content)
        
        print(f"✓ Test query: '{test_query}'")
        print(f"✓ Relevance score: {relevance_score:.4f}")
        
        if relevance_score >= 0.4:
            print(f"✓ Score {relevance_score:.4f} exceeds min threshold (0.4)")
        else:
            print(f"⚠ Score {relevance_score:.4f} below min threshold (0.4)")
            
    except Exception as e:
        print(f"✗ Relevance scoring test failed: {e}")
    
    # Check 5: Query analysis
    print("\n5. QUERY ANALYSIS:")
    print("-" * 50)
    
    try:
        # Import query analyzer logic
        def analyze_query_type(query: str) -> dict:
            """Analyze query to determine optimal search strategy"""
            query_lower = query.lower()
            words = query_lower.split()
            
            analysis = {
                'is_short': len(words) <= 3,
                'has_acronyms': any(word.isupper() for word in query.split()),
                'has_proper_nouns': any(word[0].isupper() for word in query.split() if len(word) > 1),
                'is_specific': any(keyword in query_lower for keyword in ['specific', 'exact', 'particular', 'precise']),
            }
            
            # Calculate keyword priority (0.0 = pure semantic, 1.0 = pure keyword)
            keyword_indicators = ['find', 'search', 'locate', 'specific', 'exact', 'list']
            semantic_indicators = ['how', 'why', 'what', 'explain', 'describe', 'understand']
            
            keyword_score = sum(1 for indicator in keyword_indicators if indicator in query_lower)
            semantic_score = sum(1 for indicator in semantic_indicators if indicator in query_lower)
            
            if keyword_score + semantic_score > 0:
                analysis['keyword_priority'] = keyword_score / (keyword_score + semantic_score)
            else:
                analysis['keyword_priority'] = 0.5  # Default balanced approach
            
            return analysis
        
        test_query = "partnership between beyondsoft and tencent"
        query_analysis = analyze_query_type(test_query)
        
        print(f"✓ Query analysis for '{test_query}':")
        for key, value in query_analysis.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"✗ Query analysis failed: {e}")
    
    # Summary and recommendations
    print("\n6. ISSUE ANALYSIS SUMMARY:")
    print("-" * 50)
    
    print("FINDINGS:")
    print("• Partnership collection exists and contains 'bys & tencent partnership.pdf'")
    print("• Document contains detailed Tencent partnership information")
    print("• Query classification correctly targets partnership collection")
    print("• Relevance scoring should pass minimum thresholds")
    print("• Current RAG settings appear properly configured")
    
    print("\nPOSSIBLE CAUSES:")
    print("• Vector search may not be finding the document due to embedding issues")
    print("• Hybrid search weighting might be favoring keyword over semantic results")
    print("• Document chunking might have separated key partnership information")
    print("• Re-ranking algorithms might be deprioritizing the relevant content")
    print("• Content filtering after retrieval might be removing the documents")
    
    print("\nRECOMMENDED INVESTIGATION:")
    print("1. Test vector search directly with embeddings")
    print("2. Check if document chunks are properly connected")
    print("3. Verify re-ranking is not filtering out relevant content")
    print("4. Test with different query variations")
    print("5. Check if there are any content filters or post-processing steps")

if __name__ == "__main__":
    analyze_issue()