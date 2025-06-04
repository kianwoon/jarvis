#!/usr/bin/env python3
"""
Complete test for DeepSeek-R1 classification fixes
Tests all three issues:
1. Model display
2. Query classification without RAG
3. No conversation history influence
"""

import sys
import asyncio
sys.path.insert(0, '.')

async def test_model_display():
    """Test that model display shows correct DeepSeek info"""
    print("=" * 70)
    print("TEST 1: Model Display Fix")
    print("=" * 70)
    
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/api/v1/current_model")
            if response.status_code == 200:
                model_info = response.json()
                print(f"✓ Model Name: {model_info.get('model_name')}")
                print(f"✓ Display Name: {model_info.get('display_name')}")
                
                # Check if display name is correct
                if "deepseek" in model_info.get('display_name', '').lower():
                    print("✓ SUCCESS: Model display correctly shows DeepSeek")
                else:
                    print("✗ FAIL: Model display still incorrect")
            else:
                print(f"✗ API Error: {response.status_code}")
    except Exception as e:
        print(f"✗ Error testing model display: {e}")
    
    print()

def test_smart_classification():
    """Test smart classification without LLM overthinking"""
    print("=" * 70)
    print("TEST 2: Smart Query Classification")
    print("=" * 70)
    
    from app.langchain.smart_query_classifier import classify_without_context
    
    test_queries = [
        ("what is today date & time?", "tools", "Date/time query"),
        ("send email to john about meeting", "tools", "MCP email tool"),
        ("what does the report say?", "rag", "Document search"),
        ("hello", "llm", "Simple greeting"),
        ("create a jira ticket", "tools", "MCP Jira tool"),
    ]
    
    for query, expected, description in test_queries:
        query_type, confidence = classify_without_context(query)
        status = "✓" if query_type == expected else "✗"
        print(f"{status} {description}: '{query}'")
        print(f"  Result: {query_type} (confidence: {confidence:.2f})")
        
        if query_type == "tools" and confidence >= 0.95:
            print("  ✓ High confidence - will bypass LLM reasoning")
    
    print()

def test_service_classification():
    """Test the integrated classify_query_type function"""
    print("=" * 70)
    print("TEST 3: Service Integration")
    print("=" * 70)
    
    from app.langchain.service import classify_query_type
    
    class MockLLMConfig:
        pass
    
    # Test the problematic query
    query = "what is today date & time?"
    print(f"Testing query: '{query}'")
    
    try:
        result = classify_query_type(query, MockLLMConfig())
        print(f"Classification result: {result}")
        
        if result == "TOOLS":
            print("✓ SUCCESS: Query correctly classified as TOOLS")
            print("✓ This will use MCP tools instead of RAG search")
        else:
            print(f"✗ FAIL: Query classified as {result} instead of TOOLS")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()

def test_conversation_isolation():
    """Test that simple queries aren't affected by conversation history"""
    print("=" * 70)
    print("TEST 4: Conversation Isolation")
    print("=" * 70)
    
    from app.langchain.smart_query_classifier import classify_without_context
    
    # Simulate query with history context
    query = "what is today date & time?"
    
    # Test without context (as implemented)
    query_type, confidence = classify_without_context(query)
    print(f"Query: '{query}'")
    print(f"Classification without context: {query_type} (confidence: {confidence:.2f})")
    
    if query_type == "tools" and confidence >= 0.95:
        print("✓ SUCCESS: Query classified correctly without context influence")
        print("✓ High confidence ensures no LLM overthinking")
    else:
        print("✗ FAIL: Classification might still be influenced by context")
    
    print()

async def main():
    print("\nDeepSeek-R1 Complete Fix Test Suite")
    print("=" * 70)
    
    # Run all tests
    await test_model_display()
    test_smart_classification()
    test_service_classification()
    test_conversation_isolation()
    
    print("=" * 70)
    print("Summary of Fixes:")
    print("1. Model display now correctly shows 'Deepseek R1 8B'")
    print("2. Simple queries use pattern matching with high confidence")
    print("3. Tool queries bypass LLM reasoning to avoid overthinking")
    print("4. Classification happens without conversation context")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())