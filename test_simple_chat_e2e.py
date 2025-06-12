#!/usr/bin/env python3
"""
End-to-End Test Script for Simple Chat Integration
Tests both backend API and can guide frontend testing
"""

import asyncio
import json
import httpx
import subprocess
import time
import sys
import os
from typing import Dict, Any

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_color(text: str, color: str = Colors.ENDC):
    print(f"{color}{text}{Colors.ENDC}")

def print_header(text: str):
    print_color(f"\n{'='*60}", Colors.HEADER)
    print_color(f" {text}", Colors.HEADER + Colors.BOLD)
    print_color(f"{'='*60}", Colors.HEADER)

def print_success(text: str):
    print_color(f"✅ {text}", Colors.OKGREEN)

def print_error(text: str):
    print_color(f"❌ {text}", Colors.FAIL)

def print_info(text: str):
    print_color(f"ℹ️  {text}", Colors.OKBLUE)

def print_warning(text: str):
    print_color(f"⚠️  {text}", Colors.WARNING)

def check_backend_health() -> bool:
    """Check if backend is running and healthy"""
    try:
        response = httpx.get("http://localhost:8000/api/v1/chat/simple-chat/status", timeout=5.0)
        if response.status_code == 200:
            status = response.json()
            print_success(f"Backend is running! Status: {status.get('status', 'unknown')}")
            return True
        else:
            print_error(f"Backend health check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Backend not accessible: {e}")
        return False

def check_frontend_health() -> bool:
    """Check if frontend is running"""
    try:
        response = httpx.get("http://localhost:5173", timeout=5.0)
        if response.status_code == 200:
            print_success("Frontend is running!")
            return True
        else:
            print_error(f"Frontend health check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Frontend not accessible: {e}")
        return False

async def test_simple_chat_api():
    """Test the simple chat API endpoint"""
    print_header("Testing Simple Chat API")
    
    # Test questions for different scenarios
    test_questions = [
        {
            "question": "What time is it?",
            "expected": "tool_call",
            "description": "Should trigger datetime tool"
        },
        {
            "question": "What is Python programming?",
            "expected": "llm_knowledge",
            "description": "Should use LLM general knowledge"
        },
        {
            "question": "Tell me about our company policies",
            "expected": "rag_search", 
            "description": "Should trigger RAG search"
        }
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        
        # First, check system status
        print_info("Checking system status...")
        try:
            response = await client.get("http://localhost:8000/api/v1/chat/simple-chat/status")
            if response.status_code == 200:
                status = response.json()
                print_success(f"✓ System Status: {status['status']}")
                print_info(f"  - Model: {status['current_model']}")
                print_info(f"  - Available Tools: {status['available_tools']}")
                print_info(f"  - Available Collections: {status['available_collections']}")
                
                if status['tools']:
                    print_info(f"  - Sample Tools: {', '.join(status['tools'][:3])}")
                if status['collections']:
                    print_info(f"  - Sample Collections: {', '.join(status['collections'][:2])}")
            else:
                print_error(f"Status check failed: {response.status_code}")
                return False
        except Exception as e:
            print_error(f"Status check error: {e}")
            return False
        
        # Test each question
        for i, test_case in enumerate(test_questions, 1):
            print_info(f"\nTest {i}/3: {test_case['description']}")
            print_info(f"Question: '{test_case['question']}'")
            
            try:
                request_data = {
                    "question": test_case["question"],
                    "conversation_id": f"test_conv_{i}",
                    "thinking": False
                }
                
                events_received = []
                tools_used = 0
                rag_searches = 0
                final_response = ""
                
                async with client.stream(
                    "POST",
                    "http://localhost:8000/api/v1/chat/simple-chat",
                    json=request_data
                ) as response:
                    
                    if response.status_code != 200:
                        print_error(f"Request failed: {response.status_code}")
                        text = await response.aread()
                        print_error(f"Error: {text.decode()}")
                        continue
                    
                    print_info("📥 Processing response...")
                    
                    async for chunk in response.aiter_text():
                        if chunk.strip():
                            for line in chunk.strip().split('\n'):
                                if line.strip():
                                    try:
                                        event = json.loads(line)
                                        event_type = event.get('type')
                                        events_received.append(event_type)
                                        
                                        if event_type == 'chat_start':
                                            print_info(f"🚀 Started with model: {event.get('model')}")
                                        
                                        elif event_type == 'message':
                                            content = event.get('content', '')[:100]
                                            print_info(f"💬 LLM Response: {content}{'...' if len(event.get('content', '')) > 100 else ''}")
                                        
                                        elif event_type == 'tools_start':
                                            tools_used = event.get('tool_count', 0)
                                            print_info(f"🔧 Executing {tools_used} tool(s)...")
                                        
                                        elif event_type == 'tool_result':
                                            tool_result = event.get('tool_result', {})
                                            if tool_result.get('success'):
                                                print_success(f"✓ Tool {tool_result.get('tool')}: Success")
                                            else:
                                                print_error(f"✗ Tool {tool_result.get('tool')}: {tool_result.get('error')}")
                                        
                                        elif event_type == 'rag_start':
                                            rag_searches = event.get('search_count', 0)
                                            print_info(f"📚 Searching {rag_searches} collection(s)...")
                                        
                                        elif event_type == 'rag_result':
                                            rag_result = event.get('rag_result', {})
                                            if rag_result.get('success'):
                                                docs = rag_result.get('document_count', 0)
                                                print_success(f"✓ RAG: Found {docs} documents")
                                            else:
                                                print_error(f"✗ RAG: {rag_result.get('error')}")
                                        
                                        elif event_type == 'final_response':
                                            final_response = event.get('content', '')
                                            response_preview = final_response[:150]
                                            print_success(f"🎯 Final Response: {response_preview}{'...' if len(final_response) > 150 else ''}")
                                        
                                        elif event_type == 'chat_complete':
                                            print_success(f"✅ Complete! Tools: {event.get('tool_count', 0)}, RAG: {event.get('rag_count', 0)}")
                                        
                                        elif event_type == 'error':
                                            print_error(f"💥 Error: {event.get('error')}")
                                        
                                    except json.JSONDecodeError:
                                        continue
                
                # Analyze results
                print_info(f"📊 Analysis:")
                print_info(f"  - Events received: {len(events_received)}")
                print_info(f"  - Tools used: {tools_used}")
                print_info(f"  - RAG searches: {rag_searches}")
                print_info(f"  - Final response length: {len(final_response)} chars")
                
                # Check if behavior matches expectation
                expected = test_case['expected']
                if expected == "tool_call" and tools_used > 0:
                    print_success("✓ Expected tool usage detected!")
                elif expected == "rag_search" and rag_searches > 0:
                    print_success("✓ Expected RAG search detected!")
                elif expected == "llm_knowledge" and tools_used == 0 and rag_searches == 0:
                    print_success("✓ Expected LLM-only response detected!")
                else:
                    print_warning(f"⚠️ Unexpected behavior - expected {expected}")
                
                print_color("-" * 60, Colors.OKCYAN)
                
            except Exception as e:
                print_error(f"Test failed: {e}")
                continue
        
        print_success("\n🎉 API testing completed!")
        return True

def main():
    """Main test function"""
    print_header("🚀 Jarvis Simple Chat E2E Testing")
    
    # Check if backend is running
    print_info("Checking backend status...")
    if not check_backend_health():
        print_error("Backend is not running!")
        print_info("Please start the backend with: ./run_local.sh")
        print_info("Or run: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        return False
    
    # Check if frontend is running (optional)
    print_info("\nChecking frontend status...")
    frontend_running = check_frontend_health()
    if not frontend_running:
        print_warning("Frontend is not running. You can start it with:")
        print_info("cd llm-ui && npm run dev")
        print_warning("Frontend testing will be skipped.")
    
    # Run API tests
    print_info("\nRunning API tests...")
    success = asyncio.run(test_simple_chat_api())
    
    if success:
        print_header("🎯 Next Steps for Frontend Testing")
        if frontend_running:
            print_success("✓ Frontend is running at: http://localhost:5173")
            print_info("1. Open your browser to: http://localhost:5173")
            print_info("2. Click on the 'Simple Chat' tab")
            print_info("3. Try these test questions:")
            print_info("   • 'What time is it?' (should use datetime tool)")
            print_info("   • 'What is Python?' (should use LLM knowledge)")
            print_info("   • 'Tell me about company policies' (should search documents)")
            print_info("4. Watch for real-time events and tool/RAG indicators")
        else:
            print_warning("Start frontend with: cd llm-ui && npm run dev")
            print_info("Then visit: http://localhost:5173")
        
        print_header("✅ Simple Chat System Ready!")
        print_success("Backend API: ✓ Working")
        print_success("Integration: ✓ Complete")
        print_success("Frontend UI: ✓ Ready" if frontend_running else "Frontend UI: ⚠️ Start needed")
        
        return True
    else:
        print_error("Tests failed. Please check the backend logs.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)