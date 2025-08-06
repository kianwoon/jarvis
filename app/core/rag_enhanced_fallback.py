"""
Enhanced RAG Fallback with Hybrid Search
========================================

This module provides an enhanced RAG fallback that includes:
- Hybrid search (vector + keyword)
- Basic query expansion
- Smart collection matching
- Better scoring and ranking

This is used when the full service layer cannot be accessed due to circular imports.
"""

import logging
import time
import re
from typing import List, Dict, Any, Optional
from pymilvus import Collection, connections

logger = logging.getLogger(__name__)

def enhanced_rag_search(
    query: str,
    collections: Optional[List[str]] = None,
    max_documents: Optional[int] = None,
    include_content: bool = True,
    trace=None
) -> Dict[str, Any]:
    """
    Enhanced RAG search with hybrid vector + keyword search
    
    This combines the best of both the service layer and fallback implementations
    without circular import issues.
    """
    start_time = time.time()
    
    try:
        logger.info(f"[ENHANCED RAG] Starting hybrid search for query: {query[:100]}...")
        
        # Validate query
        if not query or not isinstance(query, str):
            return {
                "success": False,
                "error": "Invalid query provided",
                "query": str(query),
                "collections_searched": [],
                "total_documents_found": 0,
                "documents_returned": 0,
                "execution_time_ms": int((time.time() - start_time) * 1000),
                "documents": [],
                "search_metadata": {"method": "enhanced_fallback", "error": True}
            }
        
        # Get collections to search with smart matching
        from app.core.rag_fallback import smart_collection_matching
        collections = smart_collection_matching(collections)
        
        logger.info(f"[ENHANCED RAG] Searching collections: {collections}")
        
        # Get settings
        from app.core.vector_db_settings_cache import get_vector_db_settings
        from app.core.embedding_settings_cache import get_embedding_settings
        from app.core.rag_settings_cache import get_document_retrieval_settings
        
        vector_settings = get_vector_db_settings()
        embedding_settings = get_embedding_settings()
        
        # Use same max_documents as standard chat for consistency
        if max_documents is None:
            doc_settings = get_document_retrieval_settings()
            max_documents = doc_settings.get('max_documents_mcp', 8)
        
        # Initialize embeddings
        from app.core.rag_fallback import get_simple_embedder
        embedder = get_simple_embedder(embedding_settings)
        
        # Generate query variations for better recall
        query_variations = expand_query_simple(query)
        logger.info(f"[ENHANCED RAG] Query variations: {query_variations}")
        
        # Extract search terms for keyword search
        search_terms = extract_search_terms(query)
        logger.info(f"[ENHANCED RAG] Search terms: {search_terms}")
        
        # Get Milvus config
        milvus_config = {}
        if 'databases' in vector_settings:
            for db in vector_settings['databases']:
                if db.get('id') == 'milvus' and db.get('enabled'):
                    milvus_config = db.get('config', {})
                    break
        
        uri = milvus_config.get('MILVUS_URI') or milvus_config.get('uri')
        token = milvus_config.get('MILVUS_TOKEN') or milvus_config.get('token')
        
        if not uri:
            logger.warning("[ENHANCED RAG] No Milvus URI found, falling back to simple search")
            from app.core.rag_fallback import simple_rag_search
            return simple_rag_search(query, collections, max_documents, include_content)
        
        # Search each collection with hybrid approach
        all_documents = []
        seen_ids = set()
        
        for collection_name in collections:
            try:
                logger.info(f"[ENHANCED RAG] Searching collection: {collection_name}")
                
                # 1. Vector search with query variations
                vector_docs = []
                for query_var in query_variations[:3]:  # Use top 3 variations
                    try:
                        query_embedding = embedder.encode([query_var])[0]
                        docs = search_milvus_vector(
                            collection_name, query_embedding, uri, token, max_docs=max_documents * 2
                        )
                        vector_docs.extend(docs)
                    except Exception as e:
                        logger.debug(f"[ENHANCED RAG] Vector search error for '{query_var}': {e}")
                
                logger.info(f"[ENHANCED RAG] Vector search found {len(vector_docs)} documents")
                
                # 2. Keyword search
                keyword_docs = []
                if search_terms:
                    try:
                        keyword_docs = search_milvus_keyword(
                            collection_name, search_terms, uri, token, max_docs=max_documents
                        )
                        logger.info(f"[ENHANCED RAG] Keyword search found {len(keyword_docs)} documents")
                    except Exception as e:
                        logger.debug(f"[ENHANCED RAG] Keyword search error: {e}")
                
                # 3. Merge and score documents
                doc_scores = {}
                
                # Add vector search results
                for doc in vector_docs:
                    doc_id = doc.get('id', hash(doc.get('content', '')))
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = {
                            'doc': doc,
                            'vector_score': doc.get('score', 0.0),
                            'keyword_score': 0.0,
                            'combined_score': 0.0
                        }
                
                # Boost scores for keyword matches
                for doc in keyword_docs:
                    doc_id = doc.get('id', hash(doc.get('content', '')))
                    if doc_id in doc_scores:
                        # Document found by both methods - boost score
                        doc_scores[doc_id]['keyword_score'] = 0.3
                        doc_scores[doc_id]['combined_score'] = (
                            doc_scores[doc_id]['vector_score'] * 0.7 +
                            doc_scores[doc_id]['keyword_score'] * 0.3
                        )
                    else:
                        # Keyword-only match
                        doc_scores[doc_id] = {
                            'doc': doc,
                            'vector_score': 0.0,
                            'keyword_score': 0.5,
                            'combined_score': 0.5
                        }
                
                # Add unique documents with metadata
                for doc_id, score_data in doc_scores.items():
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        doc = score_data['doc']
                        doc['final_score'] = score_data['combined_score']
                        doc['metadata']['source_collection'] = collection_name
                        doc['metadata']['search_method'] = 'hybrid'
                        all_documents.append(doc)
                
            except Exception as e:
                logger.warning(f"[ENHANCED RAG] Error searching collection {collection_name}: {e}")
                continue
        
        # Sort by combined score and limit results
        all_documents = sorted(all_documents, key=lambda x: x.get('final_score', 0), reverse=True)
        limited_documents = all_documents[:max_documents]
        
        # Format documents
        formatted_documents = []
        for doc in limited_documents:
            formatted_doc = {
                'title': doc.get('title', 'Unknown'),
                'score': doc.get('final_score', 0.0),
                'metadata': doc.get('metadata', {})
            }
            
            if include_content:
                formatted_doc['content'] = doc.get('content', '')
            else:
                formatted_doc['content_preview'] = (doc.get('content', '') or '')[:200] + '...'
            
            formatted_documents.append(formatted_doc)
        
        execution_time = int((time.time() - start_time) * 1000)
        
        # Build text summary for agents
        text_summary = f"✅ SUCCESS: Found {len(formatted_documents)} relevant documents from {len(collections)} collections.\n\n"
        
        if formatted_documents:
            text_summary += "📄 DOCUMENTS RETRIEVED:\n"
            for i, doc in enumerate(formatted_documents[:5], 1):
                title = doc.get('title', 'Untitled')
                content_preview = doc.get('content', doc.get('content_preview', ''))[:200]
                text_summary += f"\n{i}. {title}\n   Content: {content_preview}...\n"
        
        result = {
            "success": True,
            "query": query,
            "collections_searched": collections,
            "total_documents_found": len(all_documents),
            "documents_returned": len(formatted_documents),
            "execution_time_ms": execution_time,
            "documents": formatted_documents,
            "summary": f"Found {len(formatted_documents)} relevant documents using enhanced hybrid search",
            "text_summary": text_summary,
            "has_results": len(formatted_documents) > 0,
            "search_metadata": {
                "method": "enhanced_fallback",
                "hybrid_search": True,
                "query_expansion": True,
                "keyword_search": True,
                "vector_search": True,
                "query_variations": len(query_variations),
                "search_terms": len(search_terms)
            }
        }
        
        logger.info(f"[ENHANCED RAG] Completed hybrid search in {execution_time}ms, found {len(formatted_documents)} documents")
        return result
        
    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        logger.error(f"[ENHANCED RAG] Search failed: {str(e)}")
        
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "collections_searched": collections or [],
            "total_documents_found": 0,
            "documents_returned": 0,
            "execution_time_ms": execution_time,
            "documents": [],
            "search_metadata": {"method": "enhanced_fallback", "error": True}
        }

def expand_query_simple(query: str) -> List[str]:
    """Simple query expansion without LLM"""
    variations = [query]  # Always include original
    
    # Extract important terms (capitalized words, acronyms)
    words = query.split()
    important_terms = []
    
    for word in words:
        cleaned = re.sub(r'[^\w]', '', word)
        if len(cleaned) > 1:
            if cleaned.isupper() or (cleaned[0].isupper() and 
                                    cleaned.lower() not in ['find', 'get', 'show', 'tell']):
                important_terms.append(cleaned)
    
    # Add focused query with just important terms
    if important_terms:
        variations.append(" ".join(important_terms))
    
    # Add lowercase version for better matching
    if query != query.lower():
        variations.append(query.lower())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for v in variations:
        if v not in seen:
            seen.add(v)
            unique_variations.append(v)
    
    return unique_variations[:5]  # Limit to 5 variations

def extract_search_terms(query: str) -> List[str]:
    """Extract key search terms from query"""
    # Common stop words to filter out
    stop_words = {
        'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'a', 'an', 'what', 'how', 'when', 'where', 'why',
        'find', 'get', 'show', 'tell', 'about', 'me', 'please', 'can', 'could'
    }
    
    words = query.lower().split()
    search_terms = []
    
    for word in words:
        cleaned = re.sub(r'[^\w]', '', word)
        if len(cleaned) > 2 and cleaned not in stop_words:
            search_terms.append(cleaned)
    
    # Also extract any phrases in quotes
    quoted = re.findall(r'"([^"]*)"', query)
    search_terms.extend(quoted)
    
    return search_terms[:10]  # Limit to 10 terms

def search_milvus_vector(collection_name: str, query_embedding: List[float], 
                         uri: str, token: str, max_docs: int = 10) -> List[Dict[str, Any]]:
    """Search Milvus using vector similarity"""
    try:
        # Connect to Milvus
        connections.connect(uri=uri, token=token, alias="vector_search")
        collection = Collection(collection_name, using="vector_search")
        collection.load()
        
        # Search with vector
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="vector",
            param=search_params,
            limit=max_docs,
            output_fields=["content", "source"]
        )
        
        # Format results
        documents = []
        for hits in results:
            for hit in hits:
                entity = hit.entity
                documents.append({
                    'id': hit.id if hasattr(hit, 'id') else hash(entity.get('content', '')),
                    'title': entity.get('source', 'Unknown'),
                    'content': entity.get('content', ''),
                    'score': float(hit.distance),
                    'metadata': {
                        'source': entity.get('source', ''),
                        'collection': collection_name
                    }
                })
        
        connections.disconnect(alias="vector_search")
        return documents
        
    except Exception as e:
        logger.debug(f"[ENHANCED RAG] Vector search error: {e}")
        if connections.has_connection("vector_search"):
            connections.disconnect(alias="vector_search")
        return []

def search_milvus_keyword(collection_name: str, search_terms: List[str],
                          uri: str, token: str, max_docs: int = 10) -> List[Dict[str, Any]]:
    """Search Milvus using keyword matching"""
    try:
        # Connect to Milvus
        connections.connect(uri=uri, token=token, alias="keyword_search")
        collection = Collection(collection_name, using="keyword_search")
        collection.load()
        
        # Build search expression
        conditions = []
        for term in search_terms[:5]:  # Limit to 5 terms
            conditions.append(f'content like "%{term}%"')
        
        if not conditions:
            connections.disconnect(alias="keyword_search")
            return []
        
        expr = " or ".join(conditions)
        
        # Query with expression
        results = collection.query(
            expr=expr,
            output_fields=["content", "source"],
            limit=max_docs
        )
        
        # Format results
        documents = []
        for r in results:
            documents.append({
                'id': r.get('id', hash(r.get('content', ''))),
                'title': r.get('source', 'Unknown'),
                'content': r.get('content', ''),
                'score': 0.8,  # Default score for keyword matches
                'metadata': {
                    'source': r.get('source', ''),
                    'collection': collection_name
                }
            })
        
        connections.disconnect(alias="keyword_search")
        return documents
        
    except Exception as e:
        logger.debug(f"[ENHANCED RAG] Keyword search error: {e}")
        if connections.has_connection("keyword_search"):
            connections.disconnect(alias="keyword_search")
        return []