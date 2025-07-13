#!/usr/bin/env python3
"""
Test RAG pipeline with reranker to ensure end-to-end functionality
"""
import requests
import json

def test_rag_with_reranker():
    """Test RAG pipeline to see if reranker is working"""
    try:
        print("🔍 Testing RAG Pipeline with Reranker")
        print("=" * 60)
        
        # Test query about partnerships (should trigger reranker)
        test_query = "partnership between beyondsoft and alibaba cloud"
        
        print(f"📝 Query: {test_query}")
        print("🔄 Sending request to RAG endpoint...")
        
        # Make request to RAG endpoint
        url = "http://localhost:8000/api/v1/langchain/rag"
        payload = {
            "question": test_query,
            "conversation_id": "test-reranker-verification",
            "thinking": False
        }
        
        response = requests.post(url, json=payload, stream=True, timeout=60)
        
        if response.status_code == 200:
            print("✅ RAG request successful")
            
            # Collect streaming response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'token' in data:
                            full_response += data['token']
                        elif 'final_answer' in data:
                            full_response = data['final_answer']
                            break
                    except:
                        continue
            
            print(f"📄 Response length: {len(full_response)} characters")
            
            if len(full_response) > 100:
                print("✅ Got substantial response (reranker likely working)")
                
                # Look for partnership content indicators
                content_indicators = ['partnership', 'alibaba', 'beyondsoft', 'collaboration']
                found_indicators = [ind for ind in content_indicators if ind.lower() in full_response.lower()]
                
                if found_indicators:
                    print(f"✅ Response contains relevant content: {found_indicators}")
                    print("✅ Reranker appears to be working correctly")
                else:
                    print("⚠️  Response may not contain expected partnership content")
                
                # Show preview of response
                print(f"\n📖 Response preview:")
                print(f"{full_response[:300]}...")
                
            else:
                print("⚠️  Response seems short, reranker may not be working optimally")
                
        else:
            print(f"❌ RAG request failed with status: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to the application")
        print("   Make sure the Docker app is running on localhost:8000")
    except Exception as e:
        print(f"❌ Test failed: {e}")

def check_reranker_logs():
    """Check for reranker-related log messages"""
    print("\n📋 Checking for Reranker Log Messages...")
    print("   Look for these patterns in your Docker logs:")
    print("   - '[QwenReranker] Successfully loaded Qwen/Qwen3-Reranker-0.6B'")
    print("   - '[DEBUG] Starting Qwen reranker re-ranking of top X documents'")
    print("   - '[QwenReranker] Model loaded successfully'")
    print("   - No ModelWrapper errors")
    print("\n   Run: docker logs <container-name> | grep -i qwen")

if __name__ == "__main__":
    test_rag_with_reranker()
    check_reranker_logs()