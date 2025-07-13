#!/usr/bin/env python3

import os
import sys
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

from app.langchain.service import unified_llm_synthesis
from app.core.llm_settings_cache import get_llm_settings

def debug_rag_prompt():
    """Debug function to see exactly what prompt is generated for RAG queries"""
    
    # Test the unified_llm_synthesis function with a RAG query
    question = "search internal knowledge base. partnership between beyondsoft and tencent"
    query_type = "RAG"
    rag_context = """Content from partnership documents:
Beyondsoft has a strategic partnership with Tencent that started in 2012. The partnership includes collaboration on Tencent Cloud, TDSQL, and enterprise-level distributed systems."""
    tool_context = ""
    conversation_history = ""
    thinking = False
    rag_sources = []
    
    print("=== DEBUG: Testing unified_llm_synthesis function ===")
    print(f"Question: {question}")
    print(f"Query type: {query_type}")
    print(f"RAG context length: {len(rag_context)}")
    print()
    
    try:
        prompt, source, context = unified_llm_synthesis(
            question=question,
            query_type=query_type,
            rag_context=rag_context,
            tool_context=tool_context,
            conversation_history=conversation_history,
            thinking=thinking,
            rag_sources=rag_sources
        )
        
        print("=== GENERATED PROMPT ===")
        print(prompt)
        print()
        print(f"Source: {source}")
        print(f"Context length: {len(context)}")
        
        # Also check LLM settings
        print("\n=== LLM SETTINGS ===")
        llm_settings = get_llm_settings()
        main_llm = llm_settings.get('main_llm', {})
        print(f"Main LLM system prompt: {main_llm.get('system_prompt', 'NOT FOUND')}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_rag_prompt()