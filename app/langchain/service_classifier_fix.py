"""
Fixed classify_query_type function that uses the smart classifier
This can be integrated into service.py to fix the issue
"""

import re
import httpx
from app.langchain.smart_query_classifier import SmartQueryClassifier, integrate_smart_classifier
from app.core.mcp_tools_cache import get_enabled_mcp_tools

def classify_query_type_fixed(question: str, llm_cfg) -> str:
    """
    Enhanced classify_query_type that uses smart classification to avoid
    unnecessary RAG searches for simple queries.
    
    This fixes the issue where deepseek-r1 was reasoning about simple queries
    like "what is today date & time?" and triggering RAG searches.
    """
    print(f"[DEBUG] classify_query_type_fixed: question = {question}")
    
    # First, use the smart classifier for quick pattern-based classification
    smart_result = integrate_smart_classifier(question)
    
    print(f"[DEBUG] Smart classifier result: {smart_result['query_type']} "
          f"(confidence: {smart_result['confidence']:.2%})")
    print(f"[DEBUG] Should use RAG: {smart_result['should_use_rag']} - {smart_result['rag_reasoning']}")
    
    # For high-confidence simple queries, return immediately without LLM reasoning
    if smart_result['confidence'] > 0.8 and not smart_result['should_use_rag']:
        print(f"[DEBUG] High confidence simple query - returning {smart_result['query_type']} without LLM")
        return smart_result['query_type']
    
    # Check for large generation requirements (existing logic)
    from app.langchain.service import detect_large_output_potential
    large_output_analysis = detect_large_output_potential(question)
    from app.core.large_generation_utils import get_config_accessor
    config = get_config_accessor()
    
    if large_output_analysis["likely_large"] and large_output_analysis["estimated_items"] >= config.min_items_for_chunking:
        print(f"[DEBUG] classify_query_type: Detected large generation requirement")
        return "LARGE_GENERATION"
    
    # For uncertain cases or when RAG might be needed, use LLM reasoning
    # but with better context about what NOT to classify as RAG
    if smart_result['should_use_rag'] or smart_result['confidence'] < 0.7:
        print(f"[DEBUG] Using LLM for classification (low confidence or RAG indicated)")
        
        # Get available tools
        available_tools = get_enabled_mcp_tools()
        tool_descriptions = []
        for tool_name, tool_info in available_tools.items():
            tool_descriptions.append(f"- {tool_name}: {tool_info['description']}")
        
        tools_list = "\n".join(tool_descriptions) if tool_descriptions else "No tools available"
        
        # Enhanced prompt that better explains when NOT to use RAG
        router_prompt = f"""NO_THINK
User question: "{question}"

Available tools:
{tools_list}

IMPORTANT CLASSIFICATION RULES:

1. DO NOT classify as RAG for:
   - Simple date/time queries ("what time is it?", "what is today's date?") → TOOLS
   - Basic arithmetic ("what is 2 + 2?") → LLM
   - Greetings ("hello", "how are you?") → LLM
   - General knowledge questions ("what is water?", "capital of France?") → LLM
   - Creative requests ("write a poem", "tell a joke") → LLM
   - Opinion questions ("what do you think about...?") → LLM

2. ONLY classify as RAG when:
   - Question explicitly mentions documents, files, reports, or uploads
   - Question asks about internal company data or policies
   - Question contains "according to", "based on", "from the" + document reference
   - Question asks to search/find/retrieve from specific documents

3. Classify as TOOLS when:
   - Available tools can directly answer the question
   - Question needs current/real-time data (time, weather, web search)
   - Question requires external API calls

4. Classify as LLM for everything else

Classification for this specific question:
- Does it mention any documents/files/uploads? {('YES' if any(word in question.lower() for word in ['document', 'file', 'upload', 'report', 'pdf']) else 'NO')}
- Is it asking for current/real-time info? {('YES' if any(word in question.lower() for word in ['current', 'today', 'now', 'latest', 'real-time']) else 'NO')}
- Is it a simple greeting or basic question? {('YES' if any(pattern in question.lower() for pattern in ['hello', 'hi', 'how are you', 'what is']) else 'NO')}

Answer with exactly one word: RAG, TOOLS, or LLM"""
        
        print(f"[DEBUG] Enhanced router prompt being sent to LLM")
        
        # Call LLM for classification (existing logic)
        from app.langchain.service import build_prompt
        prompt = build_prompt(router_prompt, is_internal=True)
        
        llm_api_url = "http://localhost:8000/api/v1/generate_stream"
        payload = {
            "prompt": prompt,
            "temperature": 0.1,
            "top_p": 0.5,
            "max_tokens": 50
        }
        
        text = ""
        with httpx.Client(timeout=30.0) as client:
            with client.stream("POST", llm_api_url, json=payload) as response:
                for line in response.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    if line.startswith("data: "):
                        token = line.replace("data: ", "")
                        text += token
        
        print(f"[DEBUG] LLM classification output: {text}")
        
        # Extract classification
        text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip().upper()
        
        if "RAG" in text_clean:
            classification = "RAG"
        elif "TOOLS" in text_clean or "TOOL" in text_clean:
            classification = "TOOLS"
        elif "LLM" in text_clean:
            classification = "LLM"
        else:
            # Use smart classifier result as fallback
            classification = smart_result['query_type']
            print(f"[DEBUG] No clear LLM classification, using smart classifier: {classification}")
        
        return classification
    
    # For all other cases, trust the smart classifier
    return smart_result['query_type']