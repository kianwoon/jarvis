#!/usr/bin/env python3
"""
Test script to verify the conversation memory fix
"""
import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000/api/v1"

def test_conversation_memory():
    """Test if Jarvis now remembers conversation context"""
    
    print("Testing Jarvis Conversation Memory Fix")
    print("=" * 50)
    
    # Generate a unique conversation ID
    conversation_id = f"test-memory-{int(time.time())}"
    
    print(f"\nUsing conversation ID: {conversation_id}")
    
    # Test 1: Define an acronym
    print("\n1. First message - defining BYS:")
    print("   User: 'BYS is refer to Beyondsoft'")
    
    first_request = {
        "question": "BYS is refer to Beyondsoft",
        "session_id": conversation_id,  # Using session_id as frontend does
        "use_langgraph": False
    }
    
    try:
        response1 = requests.post(f"{BASE_URL}/langchain/rag", json=first_request, timeout=30)
        if response1.status_code == 200:
            full_response = ""
            for line in response1.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "token" in data:
                            full_response += data["token"]
                        elif "answer" in data:
                            full_response = data["answer"]
                    except:
                        pass
            print(f"   Assistant: {full_response[:150]}...")
        else:
            print(f"   Error: {response1.status_code}")
            return
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    time.sleep(2)
    
    # Test 2: Ask about BYS
    print("\n2. Second message - asking about BYS:")
    print("   User: 'What is BYS's main business?'")
    
    second_request = {
        "question": "What is BYS's main business?",
        "session_id": conversation_id,
        "use_langgraph": False
    }
    
    try:
        response2 = requests.post(f"{BASE_URL}/langchain/rag", json=second_request, timeout=30)
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
            
            print(f"   Assistant: {full_response}")
            
            # Check if the response mentions Beyondsoft
            if "beyondsoft" in full_response.lower():
                print("\n✅ SUCCESS: Jarvis remembered that BYS refers to Beyondsoft!")
                print("   The conversation history is working correctly.")
            else:
                print("\n❌ FAILURE: Jarvis did not remember that BYS refers to Beyondsoft")
                print("   The conversation history might not be working properly.")
        else:
            print(f"   Error: {response2.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Continue the conversation
    print("\n3. Third message - follow-up question:")
    print("   User: 'How many employees does BYS have?'")
    
    third_request = {
        "question": "How many employees does BYS have?",
        "session_id": conversation_id,
        "use_langgraph": False
    }
    
    try:
        response3 = requests.post(f"{BASE_URL}/langchain/rag", json=third_request, timeout=30)
        if response3.status_code == 200:
            full_response = ""
            for line in response3.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "token" in data:
                            full_response += data["token"]
                        elif "answer" in data:
                            full_response = data["answer"]
                    except:
                        pass
            
            print(f"   Assistant: {full_response[:200]}...")
            
            if "beyondsoft" in full_response.lower() or "bys" in full_response.lower():
                print("\n✅ Context maintained: Still remembering BYS = Beyondsoft")
            else:
                print("\n⚠️  Context may be lost")
        else:
            print(f"   Error: {response3.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_conversation_memory()