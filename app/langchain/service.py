import requests
import re
import httpx
import json
from app.core.llm_settings_cache import get_llm_settings
from app.core.embedding_settings_cache import get_embedding_settings
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.core.mcp_tools_cache import get_enabled_mcp_tools
from app.api.v1.endpoints.document import HTTPEndeddingFunction
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus

def build_prompt(prompt: str, thinking: bool = False, is_internal: bool = False) -> str:
    if thinking:
        return (
            "Please show your reasoning step by step before giving the final answer.\n"
            + prompt
        )
    return prompt

def calculate_relevance_score(query: str, context: str) -> float:
    """Calculate relevance score using TF-IDF-like keyword matching for hybrid search"""
    if not context or not query:
        return 0.0
        
    import math
    from collections import Counter
    
    # Convert to lowercase for comparison
    query_lower = query.lower()
    context_lower = context.lower()
    
    # More comprehensive stop words list
    stop_words = {
        'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
        'a', 'an', 'what', 'how', 'when', 'where', 'why', 'like', 'it', 'this', 'that', 'these', 
        'those', 'can', 'could', 'would', 'should', 'may', 'might', 'must', 'shall', 'will',
        'do', 'does', 'did', 'have', 'has', 'had', 'be', 'been', 'being', 'was', 'were',
        'find', 'get', 'show', 'tell', 'about', 'info', 'information', 'news', 'give', 'provide', 
        'need', 'want', 'please', 'help'
    }
    
    # Extract meaningful words
    query_words = [w for w in query_lower.split() if w not in stop_words and len(w) > 2]
    context_words_list = [w for w in context_lower.split() if len(w) > 2]
    
    if not query_words:
        query_words = query_lower.split()
    
    # Calculate term frequency (TF) for context
    context_word_freq = Counter(context_words_list)
    total_words = len(context_words_list)
    
    # Calculate TF-IDF-like score
    tfidf_score = 0.0
    matched_terms = 0
    
    for word in query_words:
        # Direct match
        if word in context_word_freq:
            tf = context_word_freq[word] / total_words if total_words > 0 else 0
            # Simulate IDF with a bonus for rare terms
            idf_bonus = 1.0 if context_word_freq[word] <= 2 else 0.5
            tfidf_score += tf * idf_bonus
            matched_terms += 1
        # Partial match (stemming-like)
        else:
            partial_matches = [w for w in context_word_freq if word in w or w in word]
            if partial_matches:
                best_match = max(partial_matches, key=lambda w: context_word_freq[w])
                tf = context_word_freq[best_match] / total_words if total_words > 0 else 0
                tfidf_score += tf * 0.7  # Lower weight for partial matches
                matched_terms += 0.7
    
    # Normalize by query length
    base_score = matched_terms / len(query_words) if query_words else 0.0
    
    # BM25-like saturation (diminishing returns for high term frequency)
    bm25_score = tfidf_score / (tfidf_score + 0.5)
    
    # Exact phrase match bonus
    phrase_bonus = 0.0
    if len(query_words) > 1 and query_lower in context_lower:
        phrase_bonus = 0.3
    
    # Proximity scoring - reward documents where query terms appear close together
    proximity_bonus = 0.0
    if len(query_words) > 1:
        positions = {}
        for word in query_words:
            positions[word] = [i for i, w in enumerate(context_words_list) if word in w or w in word]
        
        if all(positions.get(w) for w in query_words):
            # Calculate average distance between consecutive query terms
            total_distance = 0
            distance_count = 0
            
            query_word_list = list(query_words)
            for i in range(len(query_word_list) - 1):
                if positions.get(query_word_list[i]) and positions.get(query_word_list[i + 1]):
                    for p1 in positions[query_word_list[i]]:
                        for p2 in positions[query_word_list[i + 1]]:
                            if p2 > p1:  # Ensure order
                                total_distance += (p2 - p1)
                                distance_count += 1
                                break
            
            if distance_count > 0:
                avg_distance = total_distance / distance_count
                # Closer terms get higher bonus
                if avg_distance < 5:
                    proximity_bonus = 0.2
                elif avg_distance < 10:
                    proximity_bonus = 0.1
    
    # Coverage bonus - what percentage of query terms appear in the document
    coverage = matched_terms / len(query_words) if query_words else 0.0
    coverage_bonus = 0.1 if coverage >= 0.8 else 0.0
    
    # Combine all scores
    final_score = (
        base_score * 0.3 +      # Basic term matching
        bm25_score * 0.3 +      # TF-IDF with saturation
        phrase_bonus +          # Exact phrase match
        proximity_bonus +       # Term proximity
        coverage_bonus          # Query coverage
    )
    
    # Ensure score is between 0 and 1
    return min(1.0, max(0.0, final_score))

def classify_query_type(question: str, llm_cfg) -> str:
    """
    Classify the query into one of three categories:
    - RAG: Needs internal company documents/knowledge base
    - TOOLS: Needs function calls (current time, calculations, API calls, etc.)
    - LLM: Can be answered with general knowledge
    """
    print(f"[DEBUG] classify_query_type: question = {question}")
    
    # Get available tools for better classification
    available_tools = get_enabled_mcp_tools()
    tool_descriptions = []
    for tool_name, tool_info in available_tools.items():
        tool_descriptions.append(f"- {tool_name}: {tool_info['description']}")
    
    tools_list = "\n".join(tool_descriptions) if tool_descriptions else "No tools available"
    
    router_prompt = f"""NO_THINK
User question: "{question}"

Available tools:
{tools_list}

IMPORTANT: Only classify as TOOLS if the available tools can actually help answer the question.

Classify this question into exactly ONE category:
- RAG: Question about internal company data, documents, business-specific information, OR questions about specific entities (companies/products/people) that might be covered in internal documents
- TOOLS: Question that can be answered using the available tools listed above
- LLM: Question that can be answered with pure general knowledge without referencing specific entities or time-sensitive information

Classification guidelines:
1. Check if available tools can actually help with the question
2. Look for company-specific indicators (our, internal, company name if it matches your organization)
3. Questions about specific entities (companies, products, people) with topics like progress, development, strategy, performance, or updates should check RAG first for any relevant documents
4. Only use LLM for pure general knowledge questions that don't reference specific entities or time-sensitive information

Examples:
- "What's our Q4 revenue?" → RAG (internal company data)
- "What time is it now?" → TOOLS (get_datetime tool available)
- "Explain machine learning" → LLM (general knowledge)
- "How's Apple's AI progress?" → RAG (check for market research/competitor analysis docs first)
- "How's DBS progress in AI?" → RAG (check for industry reports/analysis first)
- "What is photosynthesis?" → LLM (pure general knowledge)
- "Send email to John" → TOOLS (outlook_send_message available)
- "Find the meeting notes from last week" → RAG (internal documents)

Answer with exactly one word: RAG, TOOLS, or LLM"""

    print(f"[DEBUG] classify_query_type: router_prompt = {router_prompt}")
    prompt = build_prompt(router_prompt, is_internal=True)
    
    llm_api_url = "http://localhost:8000/api/v1/generate_stream"
    payload = {
        "prompt": prompt,
        "temperature": 0.1,  # Lower temperature for more consistent classification
        "top_p": 0.5,
        "max_tokens": 50
    }
    
    text = ""
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", llm_api_url, json=payload) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if line.startswith("data: "):
                    token = line.replace("data: ", "")
                    text += token
    
    print(f"[DEBUG] classify_query_type: LLM raw output = {text}")
    print(f"[DEBUG] classify_query_type: Available tools = {list(available_tools.keys()) if available_tools else 'None'}")
    
    # Extract classification with fallback logic
    text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip().upper()
    
    if "RAG" in text_clean:
        classification = "RAG"
    elif "TOOLS" in text_clean or "TOOL" in text_clean:
        classification = "TOOLS"
    elif "LLM" in text_clean:
        classification = "LLM"
    else:
        print(f"[DEBUG] classify_query_type: No clear classification found, using fallback logic")
        # Fallback logic based on keywords and tool capability
        question_lower = question.lower()
        
        # Check if available tools can actually help with this question
        tool_can_help = False
        if available_tools:
            # Simple heuristics for tool matching
            if any(keyword in question_lower for keyword in ["time", "date", "now"]) and "get_datetime" in available_tools:
                tool_can_help = True
            elif any(keyword in question_lower for keyword in ["email", "send", "message"]) and "outlook_send_message" in available_tools:
                tool_can_help = True
            elif any(keyword in question_lower for keyword in ["jira", "issue", "ticket"]) and any("jira" in tool for tool in available_tools):
                tool_can_help = True
        
        if tool_can_help:
            classification = "TOOLS"
            print(f"[DEBUG] classify_query_type: Fallback classified as TOOLS (tool can help)")
        elif any(keyword in question_lower for keyword in ["our", "company", "internal", "document", "meeting", "policy"]):
            classification = "RAG"
            print(f"[DEBUG] classify_query_type: Fallback classified as RAG (company-specific)")
        elif any(keyword in question_lower for keyword in ["progress", "development", "strategy", "performance", "update", "latest", "recent"]) and \
             any(char.isupper() for char in question if char.isalpha()):  # Check for proper nouns (capitalized words)
            classification = "RAG"
            print(f"[DEBUG] classify_query_type: Fallback classified as RAG (entity-specific with progress/development keywords)")
        else:
            classification = "LLM"
            print(f"[DEBUG] classify_query_type: Fallback classified as LLM (general knowledge)")
    
    print(f"[DEBUG] classify_query_type: Final classification = {classification}")
    return classification

def execute_tools_first(question: str, thinking: bool = False) -> tuple:
    """
    Execute tools first, then generate answer with tool results
    Returns: (tool_results, updated_question_with_context)
    """
    print(f"[DEBUG] execute_tools_first: question = {question}, thinking = {thinking}")
    
    # Get tools context
    mcp_tools_context = get_mcp_tools_context()
    
    # First, ask LLM to identify which tools to use
    tool_selection_prompt = f"""NO_THINK
Question: {question}

{mcp_tools_context}

Based on the question, which tools should be called? Respond with tool calls in this exact format:
<tool>tool_name(parameters)</tool>

If no tools are needed, respond with: NO_TOOLS_NEEDED

Examples:
- For "What time is it?": <tool>get_datetime({{}})</tool>
- For "Weather in Paris": <tool>get_weather({{"location": "Paris"}})</tool>
"""

    llm_cfg = get_llm_settings()
    prompt = build_prompt(tool_selection_prompt, thinking=False)
    
    # Get tool calls from LLM
    llm_api_url = "http://localhost:8000/api/v1/generate_stream"
    payload = {
        "prompt": prompt,
        "temperature": 0.3,
        "top_p": 0.7,
        "max_tokens": 200
    }
    
    tool_selection_text = ""
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", llm_api_url, json=payload) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if line.startswith("data: "):
                    token = line.replace("data: ", "")
                    tool_selection_text += token
    
    print(f"[DEBUG] execute_tools_first: tool_selection_text = {tool_selection_text}")
    
    # Execute identified tools
    tool_results = extract_and_execute_tool_calls(tool_selection_text)
    
    # Build context with tool results
    if tool_results:
        results_context = "\n\nTool Results:\n"
        for result in tool_results:
            if "error" in result:
                results_context += f"- {result['tool']}: Error - {result['error']}\n"
            else:
                results_context += f"- {result['tool']}: {json.dumps(result['result'], indent=2)}\n"
        
        updated_question = f"{question}\n{results_context}\n\nPlease provide a complete answer using the tool results above."
    else:
        updated_question = question
        results_context = ""
    
    return tool_results, updated_question, results_context

def llm_expand_query(question: str, llm_cfg: dict) -> list:
    """Use LLM to generate alternative queries for better retrieval"""
    
    # Extract important terms (proper nouns, acronyms, etc.)
    import re
    words = question.split()
    important_terms = []
    
    for word in words:
        cleaned = re.sub(r'[^\w]', '', word)
        if len(cleaned) > 1:
            # Detect acronyms (all caps) or proper nouns (capitalized)
            if cleaned.isupper() or (cleaned[0].isupper() and cleaned.lower() not in 
                                   ['find', 'get', 'show', 'tell', 'give', 'provide']):
                important_terms.append(cleaned)
    
    # Always include original query
    expanded_queries = [question]
    
    # If we have specific entities, add focused variations
    if important_terms:
        # Add just the important terms as a focused query
        expanded_queries.append(" ".join(important_terms))
        
        # Add combinations with common related terms found in the original query
        content_terms = []
        for word in question.lower().split():
            if word in ['issue', 'issues', 'problem', 'problems', 'outage', 'outages', 
                       'disruption', 'failure', 'down', 'error', 'bug', 'incident']:
                content_terms.append(word)
        
        # Create combinations of important terms with content terms
        for important in important_terms:
            for content in content_terms:
                if len(expanded_queries) < 4:
                    expanded_queries.append(f"{important} {content}")
    
    # If we don't have enough variations, use LLM expansion
    if len(expanded_queries) < 3:
        expand_prompt = f"""Generate 2 alternative ways to ask this question for better document search. 
Keep the same intent but vary the phrasing and keywords.
Original question: {question}

Provide ONLY the 2 alternatives, one per line, no numbering or explanations."""
        
        llm_api_url = "http://localhost:8000/api/v1/generate_stream"
        payload = {
            "prompt": expand_prompt,
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 150
        }
        
        try:
            text = ""
            with httpx.Client(timeout=10) as client:
                with client.stream("POST", llm_api_url, json=payload) as response:
                    for line in response.iter_lines():
                        if not line:
                            continue
                        if isinstance(line, bytes):
                            line = line.decode("utf-8")
                        if line.startswith("data: "):
                            token = line.replace("data: ", "")
                            text += token
            
            # Parse alternatives
            alternatives = [q.strip() for q in text.strip().split('\n') if q.strip() and len(q.strip()) > 5]
            expanded_queries.extend(alternatives[:2])  # Take up to 2 alternatives
            
        except Exception as e:
            print(f"[DEBUG] LLM query expansion failed: {str(e)}")
    
    print(f"[DEBUG] Expanded queries: {expanded_queries}")
    return expanded_queries

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
        
        # Strategy 1: All terms must be present (AND)
        if len(search_terms) <= 3:
            conditions = [f'content like "%{word}%"' for word in search_terms]
            expr = " and ".join(conditions)
            
            print(f"[DEBUG] Keyword search expression (AND): {expr}")
            
            results = collection.query(
                expr=expr,
                output_fields=["content", "source", "page", "hash", "doc_id"],
                limit=20
            )
            all_results.extend(results)
            print(f"[DEBUG] AND search found {len(results)} results")
        
        # Strategy 2: Any important term (OR) - if AND didn't find enough
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

def extract_metadata_hints(question: str) -> dict:
    """Extract potential metadata filters from the query"""
    hints = {}
    
    # Check for date patterns
    import re
    year_pattern = r'\b(20\d{2})\b'
    years = re.findall(year_pattern, question)
    if years:
        hints['years'] = years
    
    # Check for specific document indicators
    if 'pdf' in question.lower() or 'document' in question.lower():
        hints['doc_type'] = 'pdf'
    
    return hints

def suggest_chunking_improvements(query_analysis: dict) -> dict:
    """Suggest chunking improvements based on query patterns"""
    suggestions = {
        'chunk_size': 512,  # Default
        'chunk_overlap': 128,  # Default
        'strategy': 'sentence_based',
        'recommendations': []
    }
    
    if query_analysis['is_short'] or query_analysis['has_acronyms']:
        # Smaller chunks for specific term matching
        suggestions['chunk_size'] = 256
        suggestions['chunk_overlap'] = 64
        suggestions['strategy'] = 'paragraph_based'
        suggestions['recommendations'].append(
            "Consider smaller chunks (256 tokens) for better acronym and specific term matching"
        )
    
    if query_analysis['has_proper_nouns']:
        # Ensure proper context around entities
        suggestions['chunk_overlap'] = 256
        suggestions['recommendations'].append(
            "Increase chunk overlap to preserve entity context across boundaries"
        )
    
    if query_analysis['is_specific']:
        # Precise matching needs smaller chunks
        suggestions['chunk_size'] = 200
        suggestions['strategy'] = 'sentence_based'
        suggestions['recommendations'].append(
            "Use sentence-based chunking for precise term matching"
        )
    
    return suggestions

def analyze_query_type(question: str) -> dict:
    """Analyze query to determine optimal search strategy"""
    analysis = {
        'is_short': len(question.split()) <= 3,
        'has_acronyms': False,
        'has_proper_nouns': False,
        'is_specific': False,
        'keyword_priority': 0.5  # Default 50/50 balance
    }
    
    # Check for acronyms (all caps words)
    words = question.split()
    acronyms = [w for w in words if w.isupper() and len(w) >= 2]
    if acronyms:
        analysis['has_acronyms'] = True
        analysis['keyword_priority'] = 0.7  # Favor keyword search for acronyms
    
    # Check for proper nouns (capitalized words not at start)
    proper_nouns = [w for i, w in enumerate(words) if i > 0 and w[0].isupper()]
    if proper_nouns:
        analysis['has_proper_nouns'] = True
        analysis['keyword_priority'] = max(analysis['keyword_priority'], 0.6)
    
    # Check if query is very specific (contains quotes, specific terms)
    if '"' in question or any(term in question.lower() for term in ['exactly', 'specific', 'precise']):
        analysis['is_specific'] = True
        analysis['keyword_priority'] = 0.8
    
    # Short queries often work better with keyword search
    if analysis['is_short']:
        analysis['keyword_priority'] = max(analysis['keyword_priority'], 0.7)
    
    return analysis

def handle_rag_query(question: str, thinking: bool = False) -> tuple:
    """Handle RAG queries with hybrid search (vector + keyword) - returns context only"""
    print(f"[DEBUG] handle_rag_query: question = {question}, thinking = {thinking}")
    
    embedding_cfg = get_embedding_settings()
    vector_db_cfg = get_vector_db_settings()
    llm_cfg = get_llm_settings()
    
    # Analyze query to determine search strategy
    query_analysis = analyze_query_type(question)
    print(f"[DEBUG] Query analysis: {query_analysis}")
    
    # Get chunking suggestions based on query
    chunking_suggestions = suggest_chunking_improvements(query_analysis)
    if chunking_suggestions['recommendations']:
        print(f"[DEBUG] Chunking recommendations: {', '.join(chunking_suggestions['recommendations'])}")
    
    # Extract metadata hints for filtering
    metadata_hints = extract_metadata_hints(question)
    print(f"[DEBUG] Metadata hints: {metadata_hints}")
    
    # Set up embeddings
    embedding_endpoint = embedding_cfg.get("embedding_endpoint")
    embedding_model = embedding_cfg.get("embedding_model")
    
    print(f"[DEBUG] handle_rag_query: Embedding config - endpoint: {embedding_endpoint}, model: {embedding_model}")
    
    if embedding_endpoint:
        embeddings = HTTPEndeddingFunction(embedding_endpoint)
        print(f"[DEBUG] handle_rag_query: Using HTTP embedding endpoint")
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_cfg["embedding_model"])
        print(f"[DEBUG] handle_rag_query: Using HuggingFace embeddings")
    
    # Connect to vector store
    milvus_cfg = vector_db_cfg["milvus"]
    collection = milvus_cfg.get("MILVUS_DEFAULT_COLLECTION", "default_knowledge")
    uri = milvus_cfg.get("MILVUS_URI")
    token = milvus_cfg.get("MILVUS_TOKEN")
    
    print(f"[DEBUG] handle_rag_query: Connecting to collection '{collection}' at URI '{uri}'")
    
    milvus_store = Milvus(
        embedding_function=embeddings,
        collection_name=collection,
        connection_args={"uri": uri, "token": token},
        text_field="content"
    )
    
    # Retrieve relevant documents - increase k for better coverage
    # IMPORTANT: Milvus with COSINE distance returns lower scores for more similar vectors
    # Score range: 0 (identical) to 2 (opposite)
    SIMILARITY_THRESHOLD = 1.5  # Very inclusive for initial retrieval (let re-ranking filter)
    NUM_DOCS = 50  # Retrieve many candidates for re-ranking to work with
    
    # Use LLM to expand queries for better recall
    queries_to_try = llm_expand_query(question, llm_cfg)
    
    try:
        # HYBRID SEARCH: Run both vector and keyword search in parallel
        all_docs = []
        seen_ids = set()
        
        # 1. Vector search with query expansion
        for query in queries_to_try:
            try:
                docs = milvus_store.similarity_search_with_score(query, k=NUM_DOCS) if hasattr(milvus_store, 'similarity_search_with_score') else [(doc, 0.0) for doc in milvus_store.similarity_search(query, k=NUM_DOCS)]
                
                # Add unique documents
                for doc, score in docs:
                    # Create a unique ID based on content hash
                    doc_id = hash(doc.page_content)
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append((doc, score))
                        
            except Exception as e:
                print(f"[ERROR] handle_rag_query: Failed for query '{query}': {str(e)}")
        
        # 2. ALWAYS perform keyword search in parallel (not just as fallback)
        print(f"[DEBUG] Running keyword search in parallel with vector search")
        keyword_docs = keyword_search_milvus(
            question,
            collection,
            uri=milvus_cfg.get("MILVUS_URI"),
            token=milvus_cfg.get("MILVUS_TOKEN")
        )
        
        print(f"[DEBUG] Vector search found {len(all_docs)} docs, Keyword search found {len(keyword_docs)} docs")
        
        # Add keyword search results with a favorable score (0.8 = good match in cosine distance)
        keyword_boost_count = 0
        for doc in keyword_docs:
            doc_id = hash(doc.page_content)
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append((doc, 0.8))  # Good match score for keyword results
                print(f"[DEBUG] Added keyword-only doc: {doc.page_content[:100]}...")
            else:
                # If document already found by vector search, boost its importance
                for i, (existing_doc, existing_score) in enumerate(all_docs):
                    if hash(existing_doc.page_content) == doc_id:
                        # Improve score if found by both methods
                        old_score = existing_score
                        all_docs[i] = (existing_doc, min(existing_score * 0.7, 0.5))
                        keyword_boost_count += 1
                        print(f"[DEBUG] Boosted doc score from {old_score} to {all_docs[i][1]}")
                        break
        
        print(f"[DEBUG] Keyword search added {len(keyword_docs) - keyword_boost_count} new docs and boosted {keyword_boost_count} existing docs")
        
        if not all_docs:
            print(f"[ERROR] handle_rag_query: No documents retrieved from vector or keyword search")
            
            # Last resort: Try with the most specific terms from the query
            # Extract any capitalized words or unique terms
            fallback_terms = []
            for word in question.split():
                cleaned = re.sub(r'[^\w]', '', word)
                if len(cleaned) > 2 and (cleaned.isupper() or cleaned[0].isupper()):
                    fallback_terms.append(cleaned.lower())
            
            if fallback_terms:
                print(f"[DEBUG] Trying fallback search with terms: {fallback_terms}")
                try:
                    from pymilvus import Collection, connections
                    connections.connect(uri=uri, token=token, alias="fallback_search")
                    collection_obj = Collection(collection, using="fallback_search")
                    collection_obj.load()
                    
                    # Try with the first fallback term
                    expr = f'content like "%{fallback_terms[0]}%"'
                    results = collection_obj.query(
                        expr=expr,
                        output_fields=["content", "source", "page", "hash", "doc_id"],
                        limit=10
                    )
                    
                    if results:
                        print(f"[DEBUG] Fallback search found {len(results)} results")
                        from langchain.schema import Document
                        for r in results:
                            doc = Document(
                                page_content=r.get("content", ""),
                                metadata={
                                    "source": r.get("source", ""),
                                    "page": r.get("page", 0),
                                    "hash": r.get("hash", ""),
                                    "doc_id": r.get("doc_id", "")
                                }
                            )
                            all_docs.append((doc, 1.0))
                    
                    connections.disconnect(alias="fallback_search")
                except Exception as e:
                    print(f"[ERROR] Fallback search failed: {str(e)}")
            
            if not all_docs:
                return "", ""
            
        print(f"[DEBUG] Total unique documents found: {len(all_docs)} (vector + keyword)")
        
        # Sort by score (lower is better for cosine distance)
        docs = sorted(all_docs, key=lambda x: x[1])[:NUM_DOCS]
        
    except Exception as e:
        print(f"[ERROR] handle_rag_query: Failed to search vector store: {str(e)}")
        return "", ""
    
    print(f"[DEBUG] handle_rag_query: Retrieved {len(docs)} documents")
    print(f"[DEBUG] handle_rag_query: Question = '{question}'")
    
    # Debug: Print all retrieved documents with scores
    if docs:
        print(f"[DEBUG] handle_rag_query: First 3 documents:")
        for i, (doc, score) in enumerate(docs[:3]):
            print(f"  Doc {i}: score={score:.4f}, content_preview='{doc.page_content[:100]}...'")
    else:
        print(f"[DEBUG] handle_rag_query: No documents retrieved from vector store!")
        return "", ""
    
    # Fix: For COSINE distance metric, LOWER scores are BETTER matches
    # Filter and rerank documents
    filtered_and_ranked = []
    query_analysis = analyze_query_type(question)  # Get query analysis for scoring
    for doc, score in docs:
        # Convert distance to similarity for easier understanding
        similarity = 1 - (score / 2)  # Convert [0,2] to [1,0]
        print(f"[DEBUG] handle_rag_query: Doc distance={score:.3f}, similarity={similarity:.3f}, content preview = {doc.page_content[:100]}")
        
        if score <= SIMILARITY_THRESHOLD:  # Filter by distance threshold
            # Calculate keyword-based relevance for reranking (hybrid search)
            keyword_relevance = calculate_relevance_score(question, doc.page_content)
            
            # Hybrid search: Balance vector similarity and keyword relevance
            # Use query analysis to determine optimal weighting
            keyword_weight = query_analysis['keyword_priority']
            vector_weight = 1 - keyword_weight
            combined_score = (similarity * vector_weight) + (keyword_relevance * keyword_weight)
            
            filtered_and_ranked.append((doc, score, similarity, keyword_relevance, combined_score))
    
    # Sort by combined score (higher is better)
    filtered_and_ranked.sort(key=lambda x: x[4], reverse=True)
    
    # LLM-based re-ranking for top candidates
    if filtered_and_ranked and llm_cfg:
        # Re-rank top 15 candidates (balance between quality and cost)
        num_to_rerank = min(15, len(filtered_and_ranked))
        print(f"[DEBUG] Starting LLM re-ranking of top {num_to_rerank} documents")
        top_candidates = filtered_and_ranked[:num_to_rerank]
        
        rerank_prompt = f"""Score how relevant each document is to the question on a scale of 0-10.
Question: {question}

Documents to score:
"""
        for i, (doc, _, _, _, _) in enumerate(top_candidates):
            # Take first 300 chars of each doc for context
            content_preview = doc.page_content[:300].replace('\n', ' ')
            rerank_prompt += f"\nDoc {i+1}: {content_preview}...\n"
        
        rerank_prompt += "\nProvide ONLY the scores in format: Doc1:X, Doc2:Y, Doc3:Z, etc. No explanations."
        
        try:
            llm_api_url = "http://localhost:8000/api/v1/generate_stream"
            payload = {
                "prompt": rerank_prompt,
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 100
            }
            
            scores_text = ""
            with httpx.Client(timeout=15) as client:
                with client.stream("POST", llm_api_url, json=payload) as response:
                    for line in response.iter_lines():
                        if not line:
                            continue
                        if isinstance(line, bytes):
                            line = line.decode("utf-8")
                        if line.startswith("data: "):
                            token = line.replace("data: ", "")
                            scores_text += token
            
            # Parse scores
            llm_scores = {}
            for part in scores_text.split(','):
                if ':' in part:
                    doc_part, score_part = part.split(':', 1)
                    doc_num = int(''.join(filter(str.isdigit, doc_part)))
                    try:
                        score = float(''.join(filter(lambda x: x.isdigit() or x == '.', score_part)))
                        llm_scores[doc_num] = score / 10.0  # Normalize to 0-1
                    except:
                        pass
            
            print(f"[DEBUG] LLM scores: {llm_scores}")
            
            # Update scores with LLM re-ranking
            if llm_scores:
                for i, item in enumerate(top_candidates):
                    if (i + 1) in llm_scores:
                        # Combine original score with LLM score (50/50 weight)
                        original_score = item[4]
                        llm_score = llm_scores[i + 1]
                        new_score = (original_score * 0.4) + (llm_score * 0.6)
                        # Update the tuple with new score
                        filtered_and_ranked[i] = (item[0], item[1], item[2], item[3], new_score)
                
                # Re-sort with updated scores
                filtered_and_ranked.sort(key=lambda x: x[4], reverse=True)
                print(f"[DEBUG] Re-ranking complete")
                
        except Exception as e:
            print(f"[DEBUG] LLM re-ranking failed: {str(e)}")
    
    # Sort by final score
    filtered_and_ranked.sort(key=lambda x: x[4], reverse=True)
    
    # Take top documents after reranking, but ensure diversity
    filtered_docs = []
    seen_content_hashes = set()
    
    for item in filtered_and_ranked:
        doc = item[0]
        # Create a simple hash of the first 200 chars to avoid duplicate content
        content_hash = hash(doc.page_content[:200])
        
        if content_hash not in seen_content_hashes:
            seen_content_hashes.add(content_hash)
            filtered_docs.append(doc)
            
        if len(filtered_docs) >= 8:  # Take up to 8 diverse documents
            break
    
    print(f"[DEBUG] handle_rag_query: After filtering and reranking: {len(filtered_docs)} documents")
    print(f"[DEBUG] handle_rag_query: Filtered {len(docs) - len(filtered_and_ranked)} documents by similarity threshold {SIMILARITY_THRESHOLD}")
    
    context = "\n\n".join([doc.page_content for doc in filtered_docs])
    
    # Enhanced relevance detection
    if context.strip():
        print(f"[RAG DEBUG] Query: {question}")
        print(f"[RAG DEBUG] Retrieved {len(filtered_docs)} documents")
        print(f"[RAG DEBUG] Context preview: {context[:200] if context else 'No context'}...")
        
        # Calculate relevance score
        relevance_score = calculate_relevance_score(question, context)
        
        # Get clean keywords for debug (same as used in calculation)
        query_keywords = set(question.lower().split())
        context_words = set(context.lower().split())
        stop_words = {
            'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an',
            'find', 'get', 'show', 'tell', 'about', 'info', 'information', 'news', 'what', 'how', 'when', 'where',
            'why', 'can', 'could', 'would', 'should', 'give', 'provide', 'need', 'want', 'like', 'please', 'help',
            'it', 'this', 'that', 'these', 'those'
        }
        clean_query_keywords = query_keywords - stop_words
        clean_context_words = context_words - stop_words
        
        print(f"[RAG DEBUG] Relevance score: {relevance_score:.2f}")
        print(f"[RAG DEBUG] Query keywords (filtered): {clean_query_keywords}")
        print(f"[RAG DEBUG] Context has {len(clean_context_words)} unique words")
        print(f"[RAG DEBUG] Overlapping words: {clean_query_keywords.intersection(clean_context_words)}")
        
        # If relevance is very low, return no context (will trigger LLM fallback)
        # Dynamic threshold based on query complexity
        min_relevance = 0.15 if len(query_keywords) > 2 else 0.25
        if relevance_score < min_relevance:
            print(f"[RAG DEBUG] Low relevance detected ({relevance_score:.2f} < {min_relevance}), no context returned")
            return "", ""
        
        # Context is relevant, return it for hybrid processing
        print(f"[RAG DEBUG] Good relevance ({relevance_score:.2f}), returning context for hybrid approach")
        return context, ""  # Return empty string for prompt since we handle prompting in main function
    else:
        print(f"[RAG DEBUG] No relevant documents found")
        return "", ""

def rag_answer(question: str, thinking: bool = False, stream: bool = False, conversation_id: str = None, use_langgraph: bool = True):
    """Main function with hybrid RAG+LLM approach prioritizing answer quality"""
    print(f"[DEBUG] rag_answer: incoming question = {question}")
    
    # Use LangGraph implementation if enabled
    if use_langgraph:
        try:
            from app.langchain.langgraph_rag import enhanced_rag_answer
            print(f"[DEBUG] Using LangGraph enhanced RAG implementation")
            return enhanced_rag_answer(
                question=question,
                conversation_id=conversation_id,
                thinking=thinking,
                stream=stream
            )
        except Exception as e:
            print(f"[ERROR] LangGraph implementation failed: {str(e)}")
            print(f"[DEBUG] Falling back to original implementation")
    
    # Original implementation follows...
    # Validate configuration
    llm_cfg = get_llm_settings()
    required_fields = ["model", "thinking_mode", "non_thinking_mode", "max_tokens"]
    missing = [f for f in required_fields if f not in llm_cfg or llm_cfg[f] is None]
    if missing:
        raise RuntimeError(f"Missing required LLM config fields: {', '.join(missing)}")
    
    # STEP 1: Always attempt RAG first to find relevant context
    print(f"[DEBUG] rag_answer: Step 1 - Attempting RAG retrieval")
    rag_context, _ = handle_rag_query(question, thinking)
    print(f"[DEBUG] rag_answer: RAG context length = {len(rag_context) if rag_context else 0}")
    
    # STEP 2: Check if tools can add value
    print(f"[DEBUG] rag_answer: Step 2 - Checking tool applicability")
    query_type = classify_query_type(question, llm_cfg)
    print(f"[DEBUG] rag_answer: query_type = {query_type}")
    
    tool_calls = []
    tool_context = ""
    if query_type == "TOOLS":
        tool_calls, _, tool_context = execute_tools_first(question, thinking)
        if not tool_calls:
            print(f"[DEBUG] rag_answer: No tools actually executed despite TOOLS classification")
    
    # STEP 3: Hybrid approach - combine available sources for optimal answer
    context = ""
    source = ""
    
    if rag_context and tool_calls:
        # BEST CASE: RAG + TOOLS + LLM synthesis
        source = "RAG+TOOLS+LLM"
        context = f"RAG Context:\n{rag_context}\n\nTool Results:\n{tool_context}"
        prompt = f"""You have both internal knowledge base information and tool results to answer this question comprehensively.

Internal Knowledge Base Context:
{rag_context}

Tool Results:
{tool_context}

Question: {question}

Please provide a comprehensive answer that synthesizes information from both the internal knowledge base and the tool results, along with your general knowledge to give the most complete and insightful response.

Answer:"""
        print(f"[DEBUG] rag_answer: Using RAG+TOOLS+LLM hybrid approach")
        
    elif rag_context:
        # IDEAL CASE: RAG + LLM synthesis 
        source = "RAG+LLM"
        context = rag_context
        prompt = f"""You have relevant information from our internal knowledge base. Use this along with your general knowledge to provide a comprehensive, insightful answer.

Internal Knowledge Base Context:
{rag_context}

Question: {question}

Please synthesize the specific information from our knowledge base with broader context and analysis to give the most complete and valuable response.

Answer:"""
        print(f"[DEBUG] rag_answer: Using RAG+LLM hybrid approach")
        
    elif tool_calls:
        # TOOLS + LLM
        source = "TOOLS+LLM"
        context = tool_context
        prompt = f"""Based on the tool results below, provide a comprehensive answer that incorporates this information along with relevant general knowledge.

Tool Results:
{tool_context}

Question: {question}

Answer:"""
        print(f"[DEBUG] rag_answer: Using TOOLS+LLM approach")
        
    elif query_type == "RAG":
        # RAG was attempted but no documents found - still use RAG+LLM to indicate RAG was checked
        source = "RAG+LLM"
        context = ""
        prompt = f"""No specific documents were found in our internal knowledge base for this query, but I'll provide a comprehensive answer based on general knowledge.

Question: {question}

Answer:"""
        print(f"[DEBUG] rag_answer: Using RAG+LLM approach (no documents found)")
        
    else:
        # FALLBACK: Pure LLM
        source = "LLM"
        prompt = f"Question: {question}\n\nAnswer:"
        print(f"[DEBUG] rag_answer: Using pure LLM approach (no relevant context or tools)")
    
    # Apply thinking wrapper if needed
    prompt = build_prompt(prompt, thinking=thinking)
    
    print(f"[DEBUG] rag_answer: final prompt = {prompt[:200]}...")
    print(f"[DEBUG] rag_answer: source = {source}")
    
    # Generate response using internal API
    llm_api_url = "http://localhost:8000/api/v1/generate_stream"
    mode_config = llm_cfg["thinking_mode"] if thinking else llm_cfg["non_thinking_mode"]
    
    payload = {
        "prompt": prompt,
        "temperature": mode_config.get("temperature", 0.7),
        "top_p": mode_config.get("top_p", 1.0),
        "max_tokens": llm_cfg.get("max_tokens", 2048)
    }
    
    if stream:
        def token_stream():
            text = ""
            with httpx.Client(timeout=None) as client:
                with client.stream("POST", llm_api_url, json=payload) as response:
                    for line in response.iter_lines():
                        if not line:
                            continue
                        if isinstance(line, bytes):
                            line = line.decode("utf-8")
                        if line.startswith("data: "):
                            token = line.replace("data: ", "")
                            text += token
                            yield json.dumps({"token": token}) + "\n"
            
            # Extract reasoning and preserve formatting in answer
            reasoning = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
            # Remove thinking tags but preserve all formatting
            answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            
            # Clean up the answer while preserving structure
            if answer:
                # Remove excessive blank lines (more than 2)
                answer = re.sub(r'\n{3,}', '\n\n', answer)
                # Remove leading/trailing whitespace but preserve internal structure
                answer = answer.strip()
                # Ensure proper spacing after periods for readability
                answer = re.sub(r'\.([A-Z])', r'. \1', answer)
                # Fix any missing spaces after colons
                answer = re.sub(r':([A-Z])', r': \1', answer)
            else:
                answer = ""
            
            yield json.dumps({
                "answer": answer,
                "source": source,
                "context": context,
                "reasoning": reasoning[0] if reasoning else None,
                "tool_calls": tool_calls,
                "query_type": source  # Use source as query_type for backward compatibility
            }) + "\n"
        
        return token_stream()
    
    else:
        text = ""
        with httpx.Client(timeout=None) as client:
            with client.stream("POST", llm_api_url, json=payload) as response:
                for line in response.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    if line.startswith("data: "):
                        token = line.replace("data: ", "")
                        text += token
        
        reasoning = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
        answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        
        print(f"[RAG ROUTER] Final answer preview: {answer[:100]}...")
        print(f"[RAG ROUTER] Source used: {source}")
        return {
            "answer": answer,
            "context": context,
            "reasoning": reasoning[0] if reasoning else None,
            "raw": text,
            "route": source,  # Updated to use source
            "source": source,
            "tool_calls": tool_calls,
            "query_type": source  # Use source as query_type for backward compatibility
        }

# Keep existing helper functions
def get_mcp_tools_context():
    """Prepare MCP tools context for LLM prompt"""
    enabled_tools = get_enabled_mcp_tools()
    if not enabled_tools:
        return ""
    
    tools_context = []
    for _, tool_info in enabled_tools.items():
        tool_desc = {
            "name": tool_info["name"],
            "description": tool_info["description"],
            "parameters": tool_info["parameters"]
        }
        tools_context.append(json.dumps(tool_desc, indent=2))
    
    return "\nAvailable tools:\n" + "\n".join(tools_context)

def call_mcp_tool(tool_name, parameters):
    """Call an MCP tool and return the result using info from Redis cache"""
    enabled_tools = get_enabled_mcp_tools()
    if tool_name not in enabled_tools:
        return {"error": f"Tool {tool_name} not found in enabled tools"}
    
    tool_info = enabled_tools[tool_name]
    endpoint = tool_info["endpoint"]
    method = tool_info.get("method", "POST")
    headers = tool_info.get("headers") or {}
    
    # Handle MCP server pattern: use endpoint_prefix if tool-specific endpoint doesn't work
    # Most MCP servers use a generic /invoke endpoint with tool name in payload
    endpoint_prefix = tool_info.get("endpoint_prefix")
    if endpoint_prefix and endpoint.endswith(f"/invoke/{tool_name}"):
        # Use the generic invoke endpoint instead
        endpoint = endpoint_prefix
        print(f"[DEBUG] Using generic MCP endpoint: {endpoint}")
    
    print(f"[DEBUG] Calling MCP tool {tool_name} at {endpoint} with params: {parameters}")
    
    # Add API key authentication if available
    api_key = tool_info.get("api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        print(f"[DEBUG] Added API key authentication for {tool_name}")
    
    # MCP server pattern: always use generic invoke format
    if "/invoke" in endpoint and (endpoint.endswith("/invoke") or endpoint_prefix):
        # Standard MCP format: {"name": "tool_name", "arguments": {parameters}}
        payload = {
            "name": tool_name,
            "arguments": parameters if parameters else {}
        }
        print(f"[DEBUG] Using MCP invoke format")
    else:
        # Fallback to direct parameters for other API patterns
        payload = parameters if parameters else {}
        print(f"[DEBUG] Using direct parameters format")
    
    print(f"[DEBUG] Final payload: {payload}")
    
    try:
        # Set default headers
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        
        if method.upper() == "GET":
            response = requests.get(endpoint, params=payload, headers=headers, timeout=30)
        else:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
        
        response.raise_for_status()
        
        # Handle different response types
        try:
            result = response.json()
            print(f"[DEBUG] MCP tool {tool_name} response: {result}")
            return result
        except json.JSONDecodeError:
            # If response is not JSON, return as text
            return {"result": response.text}
            
    except requests.exceptions.Timeout:
        error_msg = f"Timeout calling tool {tool_name} (30s timeout exceeded)"
        print(f"[ERROR] {error_msg}")
        return {"error": error_msg}
    except requests.exceptions.ConnectionError:
        error_msg = f"Connection error calling tool {tool_name} at {endpoint}. Check if MCP server is running."
        print(f"[ERROR] {error_msg}")
        return {"error": error_msg}
    except requests.exceptions.HTTPError as e:
        try:
            error_detail = e.response.text
        except:
            error_detail = str(e)
        error_msg = f"HTTP {e.response.status_code} error calling tool {tool_name}: {error_detail}"
        print(f"[ERROR] {error_msg}")
        
        # If we get 404 and were using tool-specific endpoint, try generic endpoint
        if e.response.status_code == 404 and endpoint_prefix and not endpoint.endswith("/invoke"):
            print(f"[DEBUG] Retrying with generic endpoint due to 404...")
            return call_mcp_tool_with_generic_endpoint(tool_name, parameters, tool_info)
        
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error calling tool {tool_name}: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return {"error": error_msg}

def call_mcp_tool_with_generic_endpoint(tool_name, parameters, tool_info):
    """Fallback function to retry with generic MCP endpoint"""
    endpoint_prefix = tool_info.get("endpoint_prefix")
    if not endpoint_prefix:
        return {"error": f"No fallback endpoint available for {tool_name}"}
    
    print(f"[DEBUG] Retrying {tool_name} with generic endpoint: {endpoint_prefix}")
    
    headers = tool_info.get("headers") or {}
    api_key = tool_info.get("api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    if "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"
    
    payload = {
        "name": tool_name,
        "arguments": parameters if parameters else {}
    }
    
    try:
        response = requests.post(endpoint_prefix, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        print(f"[DEBUG] Retry successful for {tool_name}: {result}")
        return result
    except Exception as e:
        error_msg = f"Retry failed for {tool_name}: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return {"error": error_msg}

def extract_and_execute_tool_calls(text):
    """Extract tool calls from LLM output and execute them"""
    if "NO_TOOLS_NEEDED" in text:
        return []
        
    tool_calls_pattern = r'<tool>(.*?)\((.*?)\)</tool>'
    tool_calls = re.findall(tool_calls_pattern, text, re.DOTALL)
    
    results = []
    for tool_name, params_str in tool_calls:
        try:
            tool_name = tool_name.strip()
            params_str = params_str.strip()
            
            if params_str == "{}" or params_str == "":
                params = {}
            else:
                try:
                    params = json.loads(params_str)
                except json.JSONDecodeError:
                    print(f"[ERROR] Failed to parse parameters for {tool_name}: {params_str}")
                    params = {}
            
            print(f"[DEBUG] Executing tool call: {tool_name} with params: {params}")
            result = call_mcp_tool(tool_name, params)
            
            # Special handling for date/time responses
            if tool_name == "get_datetime" and "result" in result:
                try:
                    from datetime import datetime
                    iso_date = result["result"]
                    dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
                    formatted_date = dt.strftime("%B %d, %Y at %I:%M %p")
                    result["formatted_date"] = formatted_date
                except Exception as e:
                    print(f"[ERROR] Failed to format date: {str(e)}")
            
            results.append({
                "tool": tool_name,
                "parameters": params,
                "result": result
            })
        except Exception as e:
            print(f"[ERROR] Failed to execute tool call {tool_name}: {str(e)}")
            results.append({
                "tool": tool_name,
                "error": str(e)
            })
    
    return results