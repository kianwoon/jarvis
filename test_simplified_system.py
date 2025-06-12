#!/usr/bin/env python3
"""Test the simplified system that bypasses classification"""

import asyncio
import json
import httpx
from datetime import datetime

async def test_simplified_rag():
    """Test the simplified RAG system with bypassed classification"""
    
    # Test queries
    test_queries = [
        {
            "question": "What is the current date and time?",
            "expected": "Should use datetime tool"
        },
        {
            "question": "Tell me about the latest news on AI",
            "expected": "Should use web search tool"
        },
        {
            "question": "What is 2+2?",
            "expected": "Should answer directly without tools"
        },
        {
            "question": "Search for information about Apple's WWDC 2025",
            "expected": "Should use web search for future event"
        }
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for test in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {test['question']}")
            print(f"Expected: {test['expected']}")
            print(f"{'='*60}")
            
            # Make request with skip_classification=True (default)
            response = await client.post(
                "http://localhost:8000/api/v1/langchain/rag",
                json={
                    "question": test['question'],
                    "thinking": False,
                    "skip_classification": True  # Explicitly skip classification
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                # Process streaming response
                full_response = ""
                has_tools = False
                has_classification = False
                
                for line in response.text.strip().split('\n'):
                    if line:
                        try:
                            data = json.loads(line)
                            
                            if "classification" in data:
                                has_classification = True
                                print(f"Classification detected: {data}")
                            
                            if "token" in data:
                                # Accumulate tokens
                                full_response += data["token"]
                                
                            if "answer" in data:
                                # Final answer
                                full_response = data["answer"]
                                source = data.get("source", "unknown")
                                print(f"\nSource: {source}")
                                
                                # Check if tools were used
                                if "TOOL" in source or "tool" in full_response.lower():
                                    has_tools = True
                                    
                        except json.JSONDecodeError:
                            print(f"Failed to parse: {line}")
                
                print(f"\nClassification used: {has_classification}")
                print(f"Tools detected: {has_tools}")
                print(f"\nResponse preview:")
                print(full_response[:500] + "..." if len(full_response) > 500 else full_response)
                
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
    
    print("\n\nTesting build_enhanced_system_prompt function directly...")
    from app.langchain.service import build_enhanced_system_prompt
    
    enhanced_prompt = build_enhanced_system_prompt()
    print("\nEnhanced System Prompt:")
    print("="*60)
    print(enhanced_prompt)
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_simplified_rag())