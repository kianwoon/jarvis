#!/usr/bin/env python3
"""
Test the complete flow of temporary document source tagging
by calling the actual API endpoint
"""
import asyncio
import aiohttp
import json

async def test_temp_doc_flow():
    """Test that temporary documents show correct source tags"""
    
    print("Testing temporary document source tagging through API...\n")
    
    # Test URL - adjust if running on different port
    base_url = "http://localhost:8000"
    
    # Test with a question that would use temp docs
    test_payload = {
        "question": "What does my uploaded document say about AI?",
        "thinking": False,
        "conversation_id": "test_temp_doc_123",
        "use_hybrid_rag": True,
        "hybrid_strategy": "temp_priority",
        "include_temp_docs": True
    }
    
    print("Sending request with hybrid RAG enabled...")
    print(f"Payload: {json.dumps(test_payload, indent=2)}\n")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/api/v1/langchain/rag",
                json=test_payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    # Read streaming response
                    chunks = []
                    async for line in response.content:
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                chunks.append(chunk)
                                if 'source' in chunk:
                                    print(f"Source tag received: {chunk['source']}")
                            except json.JSONDecodeError:
                                pass
                    
                    print("\nChecking results...")
                    # Look for source tags in the response
                    sources_found = [c.get('source') for c in chunks if 'source' in c]
                    if sources_found:
                        final_source = sources_found[-1]  # Get the last source tag
                        print(f"Final source tag: {final_source}")
                        
                        if "RAG_TEMP" in final_source:
                            print("✅ Success! Temporary document source is properly tagged.")
                        else:
                            print("❌ Source tag doesn't include RAG_TEMP")
                    else:
                        print("❌ No source tags found in response")
                else:
                    print(f"❌ Request failed with status: {response.status}")
                    text = await response.text()
                    print(f"Error: {text}")
    except aiohttp.ClientConnectorError:
        print("❌ Could not connect to server. Make sure the Jarvis server is running on port 8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Note: This test requires the Jarvis server to be running\n")
    asyncio.run(test_temp_doc_flow())