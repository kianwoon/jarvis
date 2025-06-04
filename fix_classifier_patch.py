#!/usr/bin/env python3
"""
Patch to fix the query classifier in service.py
This replaces the classify_query_type function with a smarter version
"""

import os
import shutil

def apply_patch():
    """Apply the classifier fix to service.py"""
    
    service_file = "/Users/kianwoonwong/Downloads/jarvis/app/langchain/service.py"
    
    # First, backup the original file
    backup_file = service_file + ".backup"
    if not os.path.exists(backup_file):
        shutil.copy(service_file, backup_file)
        print(f"Created backup: {backup_file}")
    
    # Read the current file
    with open(service_file, 'r') as f:
        content = f.read()
    
    # Find the classify_query_type function
    import_section = """import re
import time
import json
import httpx
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime

# Add smart classifier import
from app.langchain.smart_query_classifier import SmartQueryClassifier, integrate_smart_classifier
"""
    
    # Replace imports at the top
    if "from app.langchain.smart_query_classifier import" not in content:
        # Find the last import line
        lines = content.split('\n')
        last_import_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                last_import_idx = i
        
        # Insert the new import after the last import
        lines.insert(last_import_idx + 1, "from app.langchain.smart_query_classifier import SmartQueryClassifier, integrate_smart_classifier")
        content = '\n'.join(lines)
        print("Added smart classifier import")
    
    # Now replace the classify_query_type function
    new_function = '''def classify_query_type(question: str, llm_cfg) -> str:
    """
    Enhanced classify_query_type that uses smart classification to avoid
    unnecessary RAG searches for simple queries.
    
    This fixes the issue where deepseek-r1 was reasoning about simple queries
    like "what is today date & time?" and triggering RAG searches.
    """
    print(f"[DEBUG] classify_query_type: question = {question}")
    
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
    large_output_analysis = detect_large_output_potential(question)
    from app.core.large_generation_utils import get_config_accessor
    config = get_config_accessor()
    
    if large_output_analysis["likely_large"] and large_output_analysis["estimated_items"] >= config.min_items_for_chunking:
        print(f"[DEBUG] Detected large generation requirement")
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
        
        tools_list = "\\n".join(tool_descriptions) if tool_descriptions else "No tools available"
        
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

Answer with exactly one word: RAG, TOOLS, or LLM"""
        
        print(f"[DEBUG] Enhanced router prompt being sent to LLM")
        
        # Call LLM for classification (existing logic)
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
    return smart_result['query_type']'''
    
    # Find and replace the function
    import re
    
    # Pattern to find the function definition
    pattern = r'def classify_query_type\(question: str, llm_cfg\) -> str:.*?(?=\ndef|\Z)'
    
    # Replace the function
    content = re.sub(pattern, new_function, content, flags=re.DOTALL)
    
    # Write the updated content
    with open(service_file, 'w') as f:
        f.write(content)
    
    print(f"Successfully patched {service_file}")
    print("\nThe fix does the following:")
    print("1. Uses smart pattern-based classification for simple queries")
    print("2. Avoids LLM reasoning for high-confidence simple queries")
    print("3. Provides better classification rules to the LLM when needed")
    print("4. Specifically handles date/time queries as TOOLS, not RAG")

if __name__ == "__main__":
    apply_patch()