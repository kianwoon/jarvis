"""
RAG Service Wrapper for Workflow Integration
============================================

Provides a clean interface to the service layer RAG functionality
without circular import issues. This wrapper is specifically designed
for workflow contexts where the full service layer features are needed.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def execute_service_layer_rag(
    query: str,
    collections: Optional[List[str]] = None,
    max_documents: Optional[int] = None,
    include_content: bool = True,
    trace=None
) -> Dict[str, Any]:
    """
    Execute RAG search using the proven service layer
    
    This function wraps the handle_rag_query from langchain.service
    and formats the results for workflow/agent consumption.
    
    Args:
        query: The search query
        collections: Optional list of collections to search
        max_documents: Maximum number of documents to return
        include_content: Whether to include full content or just previews
        trace: Optional trace object for monitoring
    
    Returns:
        Dict with success status and results or error message
    """
    try:
        # Use same max_documents logic as standard chat for consistency
        if max_documents is None:
            from app.core.rag_settings_cache import get_document_retrieval_settings
            doc_settings = get_document_retrieval_settings()
            # Use same default as standard chat (langchain/service.py:2640)
            max_documents = doc_settings.get('max_documents_mcp', 10)
        
        # Use importlib to delay import and break circular dependencies
        import importlib
        import sys
        
        # Force reload of the module if it's partially initialized
        if 'app.langchain.service' in sys.modules:
            # Module is already imported (possibly partially)
            service_module = sys.modules['app.langchain.service']
            if hasattr(service_module, 'handle_rag_query'):
                handle_rag_query = service_module.handle_rag_query
            else:
                # Try to complete the import
                try:
                    service_module = importlib.reload(service_module)
                    handle_rag_query = service_module.handle_rag_query
                except:
                    # Fall back to direct import
                    from app.langchain.service import handle_rag_query
        else:
            # Fresh import
            service_module = importlib.import_module('app.langchain.service')
            handle_rag_query = service_module.handle_rag_query
        
        # Determine collection strategy
        collection_strategy = "specific" if collections else "auto"
        
        logger.info(f"[RAG SERVICE WRAPPER] Executing service layer RAG")
        logger.info(f"  Query: {query[:100]}...")
        logger.info(f"  Collections: {collections}")
        logger.info(f"  Strategy: {collection_strategy}")
        
        start_time = time.time()
        
        # Call the service layer function
        context, sources = handle_rag_query(
            question=query,
            thinking=False,
            collections=collections,
            collection_strategy=collection_strategy,
            trace=trace
        )
        
        execution_time = int((time.time() - start_time) * 1000)
        
        # Process the results into document format
        documents = []
        
        # Context might be a string or list of documents
        if isinstance(context, str) and context:
            # Parse sources for document metadata
            if sources and isinstance(sources, list):
                for i, source in enumerate(sources):
                    documents.append({
                        'title': source.get('title', f'Document {i+1}'),
                        'content': source.get('content', ''),
                        'score': source.get('score', 0.0),
                        'metadata': source.get('metadata', {})
                    })
            else:
                # Create a single document from context
                documents.append({
                    'title': 'Context',
                    'content': context,
                    'score': 1.0,
                    'metadata': {}
                })
        elif isinstance(context, list):
            # Context is already a list of documents
            for doc in context:
                if isinstance(doc, dict):
                    documents.append(doc)
                else:
                    # Handle Document objects
                    documents.append({
                        'title': getattr(doc, 'title', 'Unknown'),
                        'content': getattr(doc, 'page_content', str(doc)),
                        'score': getattr(doc, 'score', 0.0),
                        'metadata': getattr(doc, 'metadata', {})
                    })
        
        # Limit documents to max_documents
        if documents and max_documents:
            documents = documents[:max_documents]
        
        # Format content based on include_content flag
        if not include_content:
            for doc in documents:
                if 'content' in doc:
                    doc['content_preview'] = doc['content'][:200] + '...'
                    del doc['content']
        
        docs_returned = len(documents)
        
        # Build text summary for agent
        text_summary = f"✅ SUCCESS: Found {docs_returned} relevant documents.\n\n"
        
        if docs_returned > 0:
            text_summary += "📄 DOCUMENTS RETRIEVED:\n"
            for i, doc in enumerate(documents[:5], 1):
                title = doc.get('title', 'Untitled')
                content_preview = doc.get('content', doc.get('content_preview', ''))[:200]
                text_summary += f"\n{i}. {title}\n   Content: {content_preview}...\n"
        
        result = {
            "success": True,
            "query": query,
            "collections_searched": collections or ["auto-detected"],
            "total_documents_found": docs_returned,
            "documents_returned": docs_returned,
            "execution_time_ms": execution_time,
            "documents": documents,
            "summary": f"Found {docs_returned} relevant documents using service layer",
            "text_summary": text_summary,
            "has_results": docs_returned > 0,
            "search_metadata": {
                "method": "service_layer",
                "hybrid_search": True,
                "query_expansion": True,
                "keyword_search": True,
                "vector_search": True
            }
        }
        
        logger.info(f"[RAG SERVICE WRAPPER] Success: {docs_returned} documents found in {execution_time}ms")
        
        return result
        
    except Exception as e:
        logger.error(f"[RAG SERVICE WRAPPER] Failed to execute service layer RAG: {e}")
        
        # Return error result
        return {
            "success": False,
            "error": f"Service layer error: {str(e)}",
            "query": query,
            "collections_searched": collections or [],
            "total_documents_found": 0,
            "documents_returned": 0,
            "execution_time_ms": 0,
            "documents": [],
            "search_metadata": {
                "method": "service_layer",
                "error": True
            }
        }