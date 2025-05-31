#!/usr/bin/env python3
"""
Test script to verify how Jarvis handles conversation history
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000/api/v1"

def test_conversation_memory():
    """Test if Jarvis remembers conversation context"""
    
    print("Testing Jarvis Conversation Memory\n")
    print("=" * 50)
    
    # Start a conversation
    conversation_id = "test-conv-123"
    
    # First message: Define an acronym
    print("\n1. First message - defining BYS:")
    first_request = {
        "question": "BYS is refer to Beyondsoft",
        "conversation_id": conversation_id,
        "use_langgraph": False  # Disabled due to Redis issues
    }
    
    response1 = requests.post(f"{BASE_URL}/langchain/rag", json=first_request)
    if response1.status_code == 200:
        # Parse streaming response
        for line in response1.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "answer" in data:
                        print(f"Response: {data['answer'][:200]}...")
                except:
                    pass
    else:
        print(f"Error: {response1.status_code}")
    
    time.sleep(2)
    
    # Second message: Ask about BYS
    print("\n2. Second message - asking about BYS:")
    second_request = {
        "question": "What is BYS's main business?",
        "conversation_id": conversation_id,
        "use_langgraph": False
    }
    
    response2 = requests.post(f"{BASE_URL}/langchain/rag", json=second_request)
    if response2.status_code == 200:
        full_response = ""
        for line in response2.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "token" in data:
                        full_response += data["token"]
                    elif "answer" in data:
                        full_response = data["answer"]
                except:
                    pass
        
        print(f"Full Response: {full_response}")
        
        # Check if the response mentions Beyondsoft
        if "beyondsoft" in full_response.lower():
            print("\n✅ SUCCESS: Jarvis remembered that BYS refers to Beyondsoft!")
        else:
            print("\n❌ FAILURE: Jarvis did not remember that BYS refers to Beyondsoft")
            print("\nThis indicates the conversation history is not being passed to the LLM properly.")
    else:
        print(f"Error: {response2.status_code}")
    
    # Test with multi-agent system
    print("\n" + "=" * 50)
    print("\n3. Testing with Multi-Agent System:")
    
    # Pass conversation history explicitly
    multi_agent_request = {
        "question": "What services does BYS provide?",
        "conversation_id": conversation_id,
        "conversation_history": [
            {"role": "user", "content": "BYS is refer to Beyondsoft"},
            {"role": "assistant", "content": "I understand that BYS refers to Beyondsoft."}
        ]
    }
    
    response3 = requests.post(f"{BASE_URL}/langchain/multi-agent", json=multi_agent_request)
    if response3.status_code == 200:
        for line in response3.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if data.get("type") == "final_response":
                        print(f"Multi-Agent Response: {data['response'][:500]}...")
                        if "beyondsoft" in data['response'].lower():
                            print("\n✅ Multi-agent system used the conversation history!")
                        else:
                            print("\n❌ Multi-agent system did not use the conversation history")
                except:
                    pass
    else:
        print(f"Error: {response3.status_code}")

if __name__ == "__main__":
    test_conversation_memory()