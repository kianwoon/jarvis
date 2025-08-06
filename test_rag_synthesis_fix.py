#!/usr/bin/env python3
"""Test script to verify RAG synthesis fix - ensures comprehensive responses instead of tool syntax"""

import asyncio
import httpx
import json
import sys
import time

# Test configuration
API_URL = "http://localhost:8000/api/v1/langchain/rag"
TEST_QUERIES = [
    "search internal knowledge base, partnership between beyondsoft and alibaba",
    "tell me about the beyondsoft alibaba partnership from the knowledge base",
    "what information do we have about beyondsoft and alibaba working together"
]

async def test_rag_query(query: str, session_id: str = None):
    """Test a single RAG query"""
    print(f"\n{'='*80}")
    print(f"Testing query: {query}")
    print(f"{'='*80}")
    
    if not session_id:
        session_id = f"test_session_{int(time.time())}"
    
    request_data = {
        "question": query,
        "thinking": False,
        "stream": True,
        "session_id": session_id,
        "use_langgraph": False,
        "collections": ["default_knowledge"],
        "collection_strategy": "auto"
    }
    
    print(f"Request data: {json.dumps(request_data, indent=2)}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            complete_response = ""
            token_count = 0
            has_tool_syntax = False
            has_documents = False
            source_type = None
            
            async with client.stream("POST", API_URL, json=request_data) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            
                            # Check for different response types
                            if "type" in data and data["type"] == "status":
                                print(f"[STATUS] {data.get('message', '')}")
                            
                            elif "token" in data:
                                token = data["token"]
                                complete_response += token
                                token_count += 1
                                
                                # Check for tool syntax in tokens
                                if "<tool>" in token or "</tool>" in token:
                                    has_tool_syntax = True
                                    print(f"[WARNING] Tool syntax detected in token: {token}")
                                
                                # Print progress every 20 tokens
                                if token_count % 20 == 0:
                                    print(f"[PROGRESS] Received {token_count} tokens...")
                            
                            elif "answer" in data:
                                # Final answer received
                                final_answer = data["answer"]
                                if final_answer and not complete_response:
                                    complete_response = final_answer
                                
                                source_type = data.get("source", "unknown")
                                
                                # Check for tool syntax in final answer
                                if "<tool>" in final_answer or "</tool>" in final_answer:
                                    has_tool_syntax = True
                                    print(f"[WARNING] Tool syntax in final answer!")
                                
                                # Check for documents
                                if "documents" in data:
                                    has_documents = True
                                    doc_count = len(data["documents"])
                                    print(f"[INFO] Received {doc_count} documents")
                            
                            elif "error" in data:
                                print(f"[ERROR] {data['error']}")
                            
                        except json.JSONDecodeError:
                            # Not JSON, might be raw text
                            complete_response += line
            
            # Analyze results
            print(f"\n{'='*80}")
            print("ANALYSIS RESULTS:")
            print(f"{'='*80}")
            
            print(f"✓ Response length: {len(complete_response)} characters")
            print(f"✓ Token count: {token_count}")
            print(f"✓ Source type: {source_type}")
            print(f"✓ Has documents: {has_documents}")
            
            # Check for issues
            issues = []
            if has_tool_syntax:
                issues.append("❌ FAILURE: Response contains tool syntax (<tool>...</tool>)")
            
            if len(complete_response) < 100:
                issues.append(f"❌ FAILURE: Response too short ({len(complete_response)} chars)")
            
            if "rag_knowledge_search" in complete_response.lower():
                issues.append("❌ FAILURE: Response contains raw tool name")
            
            if not complete_response or complete_response.strip() == "":
                issues.append("❌ FAILURE: Empty response received")
            
            # Check for synthesis indicators
            synthesis_indicators = [
                "partnership" in complete_response.lower(),
                "beyondsoft" in complete_response.lower() or "beyond soft" in complete_response.lower(),
                "alibaba" in complete_response.lower(),
                len(complete_response) > 500  # Should be comprehensive
            ]
            
            if not any(synthesis_indicators):
                issues.append("⚠️ WARNING: Response may not be properly synthesized")
            
            if issues:
                print("\n❌ ISSUES FOUND:")
                for issue in issues:
                    print(f"  {issue}")
                print("\n🔍 Response preview (first 500 chars):")
                print(complete_response[:500])
                return False
            else:
                print("\n✅ SUCCESS: Response is properly synthesized!")
                print("\n📝 Response preview (first 500 chars):")
                print(complete_response[:500])
                return True
                
        except httpx.HTTPStatusError as e:
            print(f"[ERROR] HTTP error: {e.response.status_code} - {e.response.text}")
            return False
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False

async def main():
    """Run all tests"""
    print("="*80)
    print("RAG SYNTHESIS FIX VERIFICATION TEST")
    print("="*80)
    print("\nThis test verifies that RAG queries return synthesized responses")
    print("instead of raw tool syntax like <tool>rag_knowledge_search(...)</tool>\n")
    
    # Use same session for all tests to check conversation continuity
    session_id = f"test_session_{int(time.time())}"
    
    results = []
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(TEST_QUERIES)}")
        success = await test_rag_query(query, session_id)
        results.append((query, success))
        
        # Small delay between tests
        if i < len(TEST_QUERIES):
            await asyncio.sleep(2)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    success_count = sum(1 for _, success in results if success)
    
    for query, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {query[:60]}...")
    
    print(f"\nOverall: {success_count}/{len(results)} tests passed")
    
    if success_count == len(results):
        print("\n🎉 ALL TESTS PASSED! RAG synthesis is working correctly.")
        return 0
    else:
        print(f"\n❌ {len(results) - success_count} tests failed. RAG synthesis needs more work.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)