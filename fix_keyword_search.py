"""
Proposed fix for keyword_search_milvus function to also search in source field
"""

def keyword_search_milvus(question: str, collection_name: str, uri: str, token: str) -> list:
    """Perform direct keyword search in Milvus using expressions"""
    from pymilvus import Collection, connections
    import re
    
    try:
        connections.connect(uri=uri, token=token, alias="keyword_search")
        collection = Collection(collection_name, using="keyword_search")
        collection.load()
        
        # Generic stop words for any query
        stop_words = {
            'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an',
            'find', 'get', 'show', 'tell', 'about', 'info', 'information', 'news', 'what', 'how', 'when', 'where',
            'why', 'can', 'could', 'would', 'should', 'give', 'provide', 'need', 'want', 'like', 'please', 'help',
            'me', 'out', 'up', 'do', 'does', 'did', 'has', 'have', 'had', 'will', 'was', 'were', 'been', 'being'
        }
        
        # Intelligently extract search terms
        words = question.lower().split()
        search_terms = []
        
        # First pass: identify and prioritize important terms
        important_terms = []
        content_terms = []
        
        for word in words:
            # Clean the word
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) < 2:
                continue
            
            # Check if it's an important term by looking at original casing
            original_words = question.split()
            for orig in original_words:
                orig_clean = re.sub(r'[^\w]', '', orig)
                if orig_clean.lower() == clean_word:
                    # Important if it's an acronym (all caps) or proper noun (capitalized, not common word)
                    if (orig_clean.isupper() and len(orig_clean) > 1) or \
                       (orig_clean[0].isupper() and orig_clean.lower() not in stop_words and 
                        orig_clean not in ['Find', 'Get', 'Show', 'Tell', 'Give', 'Provide']):
                        important_terms.append(clean_word)
                    break
            
            # Also collect content terms (not stop words, length > 2)
            if clean_word not in stop_words and len(clean_word) > 2:
                content_terms.append(clean_word)
        
        # Strategy: prioritize important terms, but include relevant content terms
        if important_terms:
            search_terms = important_terms
            # Add a few most relevant content terms that aren't already included
            for term in content_terms:
                if term not in important_terms and len(search_terms) < 4:
                    search_terms.append(term)
        else:
            # If no important terms, use content terms
            search_terms = content_terms[:4]  # Limit to avoid overly complex queries
        
        print(f"[DEBUG] Keyword search - original query: {question}")
        print(f"[DEBUG] Keyword search - important terms: {important_terms}")
        print(f"[DEBUG] Keyword search - search terms: {search_terms}")
        
        # Build expressions - try different strategies
        all_results = []
        
        # NEW: Check if we should search in source field
        # Look for document-related keywords or file extensions
        doc_indicators = ['document', 'file', 'pdf', 'docx', 'doc', 'report', 'presentation']
        search_source = any(indicator in question.lower() for indicator in doc_indicators)
        
        # Also check if any important terms might be document names
        if important_terms and not search_source:
            # If we have proper nouns/acronyms, they might be document names
            search_source = True
        
        # Strategy 1: Search in BOTH content and source fields
        if search_source and important_terms:
            # First try source field for important terms
            for term in important_terms:
                expr = f'source like "%{term.lower()}%"'
                print(f"[DEBUG] Searching in source field: {expr}")
                
                try:
                    results = collection.query(
                        expr=expr,
                        output_fields=["content", "source", "page", "hash", "doc_id"],
                        limit=20
                    )
                    if results:
                        print(f"[DEBUG] Found {len(results)} results in source field for '{term}'")
                        all_results.extend(results)
                except Exception as e:
                    print(f"[DEBUG] Source search error: {str(e)}")
        
        # Strategy 2: Content search with all terms (AND)
        if len(search_terms) <= 3:
            # Use lowercase for case-insensitive search
            conditions = [f'content like "%{word.lower()}%"' for word in search_terms]
            expr = " and ".join(conditions)
            
            print(f"[DEBUG] Keyword search expression (AND): {expr}")
            
            results = collection.query(
                expr=expr,
                output_fields=["content", "source", "page", "hash", "doc_id"],
                limit=20
            )
            
            # Add unique results only
            existing_hashes = {r.get("hash") for r in all_results}
            for r in results:
                if r.get("hash") not in existing_hashes:
                    all_results.append(r)
            
            print(f"[DEBUG] AND search found {len(results)} new results")
        
        # Strategy 3: Any important term (OR) - if AND didn't find enough
        if len(all_results) < 5 and important_terms:
            conditions = [f'content like "%{word}%"' for word in important_terms]
            expr = " or ".join(conditions)
            
            print(f"[DEBUG] Keyword search expression (OR): {expr}")
            
            results = collection.query(
                expr=expr,
                output_fields=["content", "source", "page", "hash", "doc_id"],
                limit=20
            )
            
            # Add unique results
            existing_hashes = {r.get("hash") for r in all_results}
            for r in results:
                if r.get("hash") not in existing_hashes:
                    all_results.append(r)
            
            print(f"[DEBUG] OR search added {len(results)} results")
        
        print(f"[DEBUG] Keyword search total found {len(all_results)} results")
        
        # Convert to document format
        from langchain.schema import Document
        docs = []
        for r in all_results:
            doc = Document(
                page_content=r.get("content", ""),
                metadata={
                    "source": r.get("source", ""),
                    "page": r.get("page", 0),
                    "hash": r.get("hash", ""),
                    "doc_id": r.get("doc_id", "")
                }
            )
            docs.append(doc)
        
        return docs
        
    except Exception as e:
        print(f"[ERROR] Keyword search failed: {str(e)}")
        return []
    finally:
        connections.disconnect(alias="keyword_search")