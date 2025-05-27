#!/usr/bin/env python3
"""
Test script for LangGraph-enhanced RAG implementation
"""
import asyncio
import json
import requests
from typing import Optional
import uuid

# API endpoint
API_BASE_URL = "http://localhost:8000/api/v1"

def test_single_query():
    """Test a single query without conversation history"""
    print("\n=== Testing Single Query ===")
    
    query = {
        "question": "What are the benefits of using LangGraph for RAG systems?",
        "thinking": False,
        "use_langgraph": True
    }
    
    response = requests.post(f"{API_BASE_URL}/langchain/rag", json=query, stream=True)
    
    print(f"Query: {query['question']}")
    print("Response:")
    
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            if 'token' in data:
                print(data['token'], end='', flush=True)
            elif 'answer' in data:
                print(f"\n\nMetadata:")
                print(f"- Source: {data.get('source', 'N/A')}")
                print(f"- Conversation ID: {data.get('conversation_id', 'N/A')}")
                if 'metadata' in data:
                    print(f"- Classification Time: {data['metadata'].get('classification_time', 'N/A')}")
                    print(f"- Retrieval Count: {data['metadata'].get('retrieval_count', 'N/A')}")
                    print(f"- Compression Ratio: {data['metadata'].get('compression_ratio', 'N/A')}")

def test_conversation_with_memory():
    """Test conversation with memory persistence"""
    print("\n\n=== Testing Conversation with Memory ===")
    
    # Generate a conversation ID
    conversation_id = str(uuid.uuid4())
    print(f"Conversation ID: {conversation_id}")
    
    # First query
    query1 = {
        "question": "Tell me about machine learning algorithms",
        "thinking": False,
        "conversation_id": conversation_id,
        "use_langgraph": True
    }
    
    print(f"\nQuery 1: {query1['question']}")
    response1 = requests.post(f"{API_BASE_URL}/langchain/rag", json=query1, stream=True)
    
    answer1 = ""
    for line in response1.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            if 'token' in data:
                print(data['token'], end='', flush=True)
            elif 'answer' in data:
                answer1 = data['answer']
    
    # Second query referencing the first
    query2 = {
        "question": "Can you give me a practical example of the first algorithm you mentioned?",
        "thinking": False,
        "conversation_id": conversation_id,
        "use_langgraph": True
    }
    
    print(f"\n\nQuery 2: {query2['question']}")
    response2 = requests.post(f"{API_BASE_URL}/langchain/rag", json=query2, stream=True)
    
    for line in response2.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            if 'token' in data:
                print(data['token'], end='', flush=True)
            elif 'answer' in data:
                print(f"\n\nThis response should reference the previous conversation!")

def test_context_compression():
    """Test context compression with large document retrieval"""
    print("\n\n=== Testing Context Compression ===")
    
    # Query that might retrieve many documents
    query = {
        "question": "Give me a comprehensive overview of all AI technologies, machine learning, deep learning, neural networks, and their applications",
        "thinking": False,
        "use_langgraph": True
    }
    
    print(f"Query: {query['question']}")
    print("(This query is designed to retrieve many documents to test compression)")
    
    response = requests.post(f"{API_BASE_URL}/langchain/rag", json=query, stream=True)
    
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            if 'token' in data:
                print(data['token'], end='', flush=True)
            elif 'answer' in data:
                if 'metadata' in data:
                    print(f"\n\nCompression Statistics:")
                    print(f"- Documents Retrieved: {data['metadata'].get('retrieval_count', 'N/A')}")
                    print(f"- Compression Ratio: {data['metadata'].get('compression_ratio', 'N/A'):.2%}")

def test_hybrid_rag_tools():
    """Test hybrid RAG + Tools execution"""
    print("\n\n=== Testing Hybrid RAG + Tools ===")
    
    query = {
        "question": "What time is it now, and can you tell me about our company's time management policies?",
        "thinking": False,
        "use_langgraph": True
    }
    
    print(f"Query: {query['question']}")
    print("(This query should trigger both tool execution and RAG retrieval)")
    
    response = requests.post(f"{API_BASE_URL}/langchain/rag", json=query, stream=True)
    
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            if 'token' in data:
                print(data['token'], end='', flush=True)
            elif 'answer' in data:
                print(f"\n\nExecution Details:")
                print(f"- Source: {data.get('source', 'N/A')}")
                if 'tool_calls' in data and data['tool_calls']:
                    print(f"- Tools Used: {[t['tool'] for t in data['tool_calls']]}")

def test_fallback_to_original():
    """Test fallback to original implementation"""
    print("\n\n=== Testing Fallback to Original Implementation ===")
    
    query = {
        "question": "What is Python?",
        "thinking": False,
        "use_langgraph": False  # Explicitly disable LangGraph
    }
    
    print(f"Query: {query['question']}")
    print("(Using original implementation)")
    
    response = requests.post(f"{API_BASE_URL}/langchain/rag", json=query, stream=True)
    
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            if 'token' in data:
                print(data['token'], end='', flush=True)
            elif 'answer' in data:
                print(f"\n\nSource: {data.get('source', 'N/A')}")

def main():
    """Run all tests"""
    print("LangGraph RAG Enhancement Test Suite")
    print("====================================")
    
    try:
        # Test 1: Single query
        test_single_query()
        
        # Test 2: Conversation with memory
        test_conversation_with_memory()
        
        # Test 3: Context compression
        test_context_compression()
        
        # Test 4: Hybrid RAG + Tools
        test_hybrid_rag_tools()
        
        # Test 5: Fallback
        test_fallback_to_original()
        
        print("\n\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()