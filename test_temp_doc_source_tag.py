#!/usr/bin/env python3
"""
Test script to verify that temporary document sources are properly tagged
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.langchain.service import unified_llm_synthesis

async def test_temp_doc_source_tagging():
    """Test that temporary documents are properly tagged in source labels"""
    
    print("Testing source tagging for temporary documents...\n")
    
    # Test 1: No RAG context - should show just "LLM"
    print("Test 1: No RAG context")
    prompt1, source1, context1 = unified_llm_synthesis(
        question="What is the weather today?",
        query_type="LLM",
        rag_context="",
        tool_context="",
        conversation_history="",
        thinking=False,
        rag_sources=[]
    )
    print(f"Source: {source1}")
    assert source1 == "LLM", f"Expected 'LLM', got '{source1}'"
    print("✓ Passed\n")
    
    # Test 2: Regular RAG context - should show "RAG+LLM"
    print("Test 2: Regular RAG context")
    prompt2, source2, context2 = unified_llm_synthesis(
        question="What is machine learning?",
        query_type="RAG",
        rag_context="Machine learning is a subset of AI...",
        tool_context="",
        conversation_history="",
        thinking=False,
        rag_sources=[
            {"file": "ml_guide.pdf", "page": 1, "collection": "default_knowledge"}
        ]
    )
    print(f"Source: {source2}")
    assert source2 == "RAG+LLM", f"Expected 'RAG+LLM', got '{source2}'"
    print("✓ Passed\n")
    
    # Test 3: Temporary document context - should show "RAG_TEMP+LLM"
    print("Test 3: Temporary document context")
    prompt3, source3, context3 = unified_llm_synthesis(
        question="What does the uploaded document say?",
        query_type="RAG",
        rag_context="The uploaded document contains...",
        tool_context="",
        conversation_history="",
        thinking=False,
        rag_sources=[
            {"file": "[TEMP] my_document.pdf", "page": 1, "collection": "temp_documents"}
        ]
    )
    print(f"Source: {source3}")
    assert source3 == "RAG_TEMP+LLM", f"Expected 'RAG_TEMP+LLM', got '{source3}'"
    print("✓ Passed\n")
    
    # Test 4: Mixed sources with is_temporary flag - should show "RAG_TEMP+LLM"
    print("Test 4: Mixed sources with is_temporary flag")
    prompt4, source4, context4 = unified_llm_synthesis(
        question="Compare the documents",
        query_type="RAG",
        rag_context="Document comparison...",
        tool_context="",
        conversation_history="",
        thinking=False,
        rag_sources=[
            {"file": "regular.pdf", "page": 1, "collection": "default_knowledge"},
            {"file": "temp.pdf", "page": 1, "collection": "temp_collection", "is_temporary": True}
        ]
    )
    print(f"Source: {source4}")
    assert source4 == "RAG_TEMP+LLM", f"Expected 'RAG_TEMP+LLM', got '{source4}'"
    print("✓ Passed\n")
    
    # Test 5: With tools and temp docs - should show "RAG_TEMP+TOOLS+LLM"
    print("Test 5: With tools and temporary documents")
    prompt5, source5, context5 = unified_llm_synthesis(
        question="Search and analyze the document",
        query_type="TOOLS",
        rag_context="Document content...",
        tool_context="Search results...",
        conversation_history="",
        thinking=False,
        rag_sources=[
            {"file": "uploaded.pdf", "page": 1, "collection": "temp_documents", "is_temporary": True}
        ]
    )
    print(f"Source: {source5}")
    assert source5 == "RAG_TEMP+TOOLS+LLM", f"Expected 'RAG_TEMP+TOOLS+LLM', got '{source5}'"
    print("✓ Passed\n")
    
    print("All tests passed! ✨")
    
    # Test the intelligent_chat endpoint flow
    print("\nTesting intelligent_chat endpoint flow...")
    from app.api.v1.endpoints.intelligent_chat import execute_rag_search
    
    # Mock a RAG search result with temp docs
    rag_result = await execute_rag_search(
        query="test query",
        collections=None,
        conversation_id="test_conv_123",
        include_temp_docs=True
    )
    
    print(f"RAG result includes temp_docs_included flag: {rag_result.get('temp_docs_included', False)}")
    
    print("\nSource tagging is working correctly!")

if __name__ == "__main__":
    asyncio.run(test_temp_doc_source_tagging())