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
    """Calculate relevance score between query and context using enhanced keyword matching"""
    if not context or not query:
        return 0.0
        
    # Convert to lowercase for comparison
    query_lower = query.lower()
    context_lower = context.lower()
    
    # Extract keywords
    query_keywords = set(query_lower.split())
    context_words = set(context_lower.split())
    
    # Remove common stop words
    stop_words = {'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'what', 'how', 'when', 'where', 'why', 'like', 'it', 'this', 'that', 'these', 'those'}
    query_keywords = query_keywords - stop_words
    context_words = context_words - stop_words
    
    if not query_keywords:
        return 0.0
    
    # Calculate different relevance metrics
    exact_overlap = len(query_keywords.intersection(context_words))
    base_score = exact_overlap / len(query_keywords)
    
    # Bonus for exact phrase matches
    phrase_bonus = 0.0
    if len(query_keywords) > 1:
        # Check for multi-word phrases
        if query_lower in context_lower:
            phrase_bonus = 0.3
        else:
            # Check for partial phrase matches (2-word combinations)
            query_words_list = [w for w in query_lower.split() if w not in stop_words]
            for i in range(len(query_words_list) - 1):
                bigram = f"{query_words_list[i]} {query_words_list[i+1]}"
                if bigram in context_lower:
                    phrase_bonus = max(phrase_bonus, 0.2)
    
    # Bonus for high keyword density
    density_bonus = 0.0
    if context_words:
        keyword_density = exact_overlap / min(len(context_words), 50)  # Normalize by first 50 words
        density_bonus = min(keyword_density * 0.1, 0.1)  # Max 0.1 bonus
    
    # Final score combining all factors
    relevance_score = min(base_score + phrase_bonus + density_bonus, 1.0)
    
    return relevance_score

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
    mode = llm_cfg["non_thinking_mode"]
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
    print(f"[DEBUG] execute_tools_first: question = {question}")
    
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
    mode = llm_cfg["non_thinking_mode"]
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

def handle_rag_query(question: str, thinking: bool = False) -> tuple:
    """Handle RAG queries with document retrieval and relevance detection - returns context only"""
    print(f"[DEBUG] handle_rag_query: question = {question}")
    
    embedding_cfg = get_embedding_settings()
    vector_db_cfg = get_vector_db_settings()
    
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
    SIMILARITY_THRESHOLD = 1.0  # Increased from 0.3 to be more inclusive (distance threshold)
    NUM_DOCS = 10  # Increased from 8 to retrieve more documents for better context
    
    try:
        docs = milvus_store.similarity_search_with_score(question, k=NUM_DOCS) if hasattr(milvus_store, 'similarity_search_with_score') else [(doc, 0.0) for doc in milvus_store.similarity_search(question, k=NUM_DOCS)]
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
    for doc, score in docs:
        # Convert distance to similarity for easier understanding
        similarity = 1 - (score / 2)  # Convert [0,2] to [1,0]
        print(f"[DEBUG] handle_rag_query: Doc distance={score:.3f}, similarity={similarity:.3f}, content preview = {doc.page_content[:100]}")
        
        if score <= SIMILARITY_THRESHOLD:  # Filter by distance threshold
            # Calculate keyword-based relevance for reranking
            keyword_relevance = calculate_relevance_score(question, doc.page_content)
            
            # Combined score: 70% vector similarity + 30% keyword relevance
            combined_score = (similarity * 0.7) + (keyword_relevance * 0.3)
            
            filtered_and_ranked.append((doc, score, similarity, keyword_relevance, combined_score))
    
    # Sort by combined score (higher is better)
    filtered_and_ranked.sort(key=lambda x: x[4], reverse=True)
    
    # Take top documents after reranking
    filtered_docs = [item[0] for item in filtered_and_ranked[:6]]  # Take top 6 after reranking
    
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
        stop_words = {'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'what', 'how', 'when', 'where', 'why', 'like', 'it', 'this', 'that', 'these', 'those'}
        clean_query_keywords = query_keywords - stop_words
        clean_context_words = context_words - stop_words
        
        print(f"[RAG DEBUG] Relevance score: {relevance_score:.2f}")
        print(f"[RAG DEBUG] Query keywords (filtered): {clean_query_keywords}")
        print(f"[RAG DEBUG] Overlapping words: {clean_query_keywords.intersection(clean_context_words)}")
        
        # If relevance is very low, return no context (will trigger LLM fallback)
        # Lowered threshold to be more inclusive - let LLM decide if context is useful
        if relevance_score < 0.1:  # Less than 10% relevance (was 20%)
            print(f"[RAG DEBUG] Very low relevance detected ({relevance_score:.2f}), no context returned")
            return "", ""
        
        # Context is relevant, return it for hybrid processing
        print(f"[RAG DEBUG] Good relevance ({relevance_score:.2f}), returning context for hybrid approach")
        return context, ""  # Return empty string for prompt since we handle prompting in main function
    else:
        print(f"[RAG DEBUG] No relevant documents found")
        return "", ""

def rag_answer(question: str, thinking: bool = False, stream: bool = False):
    """Main function with hybrid RAG+LLM approach prioritizing answer quality"""
    print(f"[DEBUG] rag_answer: incoming question = {question}")
    
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
    mode = llm_cfg["thinking_mode"] if thinking else llm_cfg["non_thinking_mode"]
    
    payload = {
        "prompt": prompt,
        "temperature": mode.get("temperature", 0.7),
        "top_p": mode.get("top_p", 1.0),
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
    for tool_name, tool_info in enabled_tools.items():
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