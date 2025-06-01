#!/usr/bin/env python3
"""
Test script for the Context-Limit-Transcending System
Demonstrates generating 100+ items that would normally exceed context limits
"""

import asyncio
import json
import requests
import time
from typing import List, Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
LARGE_GENERATION_ENDPOINT = f"{API_BASE_URL}/large-generation"
PROGRESS_ENDPOINT = f"{API_BASE_URL}/large-generation/progress"
CLEANUP_ENDPOINT = f"{API_BASE_URL}/large-generation/cleanup"

def test_large_generation_api():
    """Test the large generation API with different scenarios"""
    
    test_cases = [
        {
            "name": "100 Marketing Questions",
            "task_description": "Generate 100 comprehensive marketing questionnaire questions for a new product launch, covering market research, customer demographics, competitive analysis, pricing strategy, and promotional tactics",
            "target_count": 100,
            "chunk_size": 15
        },
        {
            "name": "50 Technical Interview Questions", 
            "task_description": "Generate 50 detailed technical interview questions for a senior software engineer position, covering algorithms, system design, databases, and coding best practices",
            "target_count": 50,
            "chunk_size": 12
        },
        {
            "name": "200 Product Ideas",
            "task_description": "Generate 200 innovative product ideas for a tech startup, including SaaS tools, mobile apps, IoT devices, and AI-powered solutions",
            "target_count": 200,
            "chunk_size": 20
        }
    ]
    
    print("🚀 Testing Context-Limit-Transcending System")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        print(f"Target: {test_case['target_count']} items")
        print(f"Chunk size: {test_case['chunk_size']}")
        print("-" * 40)
        
        # Run the test
        session_id = run_large_generation_test(test_case)
        
        if session_id:
            print(f"✅ Test completed successfully!")
            print(f"Session ID: {session_id}")
            
            # Optional: Clean up the session
            cleanup_response = requests.delete(f"{CLEANUP_ENDPOINT}/{session_id}")
            if cleanup_response.status_code == 200:
                print(f"🧹 Session cleaned up")
        else:
            print(f"❌ Test failed")
        
        print("\n" + "=" * 60)
        
        # Wait between tests
        if i < len(test_cases):
            print("⏳ Waiting 5 seconds before next test...")
            time.sleep(5)

def run_large_generation_test(test_case: Dict[str, Any]) -> str:
    """Run a single large generation test and return session_id if successful"""
    
    payload = {
        "task_description": test_case["task_description"],
        "target_count": test_case["target_count"],
        "chunk_size": test_case["chunk_size"],
        "use_redis": True,
        "conversation_id": f"test_{int(time.time())}"
    }
    
    try:
        print(f"🔄 Starting generation...")
        
        # Make streaming request
        response = requests.post(
            LARGE_GENERATION_ENDPOINT,
            json=payload,
            stream=True,
            timeout=600  # 10 minute timeout
        )
        
        if response.status_code != 200:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        # Process streaming events
        session_id = None
        total_chunks = 0
        completed_chunks = 0
        total_items = 0
        
        for line in response.iter_lines():
            if line:
                try:
                    line_text = line.decode('utf-8')
                    if line_text.strip():
                        event = json.loads(line_text)
                        event_type = event.get("type", "unknown")
                        
                        if event_type == "task_decomposed":
                            total_chunks = event.get("total_chunks", 0)
                            estimated_duration = event.get("estimated_duration", 0)
                            print(f"📊 Task decomposed into {total_chunks} chunks")
                            print(f"⏱️  Estimated duration: {estimated_duration} seconds")
                            
                        elif event_type == "execution_started":
                            session_id = event.get("session_id")
                            print(f"🎯 Execution started (Session: {session_id[:8]}...)")
                            
                        elif event_type == "chunk_started":
                            chunk_num = event.get("chunk_number", 0)
                            items_range = event.get("items_range", "")
                            print(f"  🔨 Chunk {chunk_num}/{total_chunks} started (items {items_range})")
                            
                        elif event_type == "chunk_completed":
                            completed_chunks += 1
                            chunk_items = event.get("items_generated", 0)
                            total_items = event.get("cumulative_items", 0)
                            progress = event.get("progress_percentage", 0)
                            exec_time = event.get("execution_time", 0)
                            
                            print(f"  ✅ Chunk {completed_chunks}/{total_chunks} completed")
                            print(f"     Items: {chunk_items} (Total: {total_items}) | Progress: {progress:.1f}% | Time: {exec_time:.1f}s")
                            
                            # Show partial results
                            partial_results = event.get("partial_results", [])
                            if partial_results:
                                print(f"     Last items: {', '.join(partial_results[:2])}...")
                            
                        elif event_type == "chunk_failed":
                            chunk_num = event.get("chunk_number", 0)
                            error = event.get("error", "Unknown error")
                            print(f"  ❌ Chunk {chunk_num} failed: {error}")
                            
                        elif event_type == "quality_check_started":
                            items_count = event.get("items_to_validate", 0)
                            print(f"🔍 Quality check started for {items_count} items")
                            
                        elif event_type == "quality_check_completed":
                            quality_score = event.get("quality_score", 0)
                            recommendations = event.get("recommendations", [])
                            print(f"📈 Quality score: {quality_score:.2f}")
                            if recommendations:
                                print(f"💡 Recommendations: {len(recommendations)} items")
                            
                        elif event_type == "large_generation_completed":
                            final_count = event.get("actual_count", 0)
                            target_count = event.get("target_count", 0)
                            completion_rate = event.get("completion_rate", 0)
                            total_time = event.get("metadata", {}).get("total_execution_time", 0)
                            
                            print(f"🎉 Generation completed!")
                            print(f"   Generated: {final_count}/{target_count} items ({completion_rate:.1%})")
                            print(f"   Total time: {total_time:.1f} seconds")
                            print(f"   Session: {session_id}")
                            
                            return session_id
                            
                        elif event_type == "large_generation_error":
                            error = event.get("error", "Unknown error")
                            print(f"❌ Generation error: {error}")
                            return None
                            
                except json.JSONDecodeError:
                    continue  # Skip malformed JSON lines
                except Exception as e:
                    print(f"⚠️  Error processing event: {e}")
                    continue
        
        return session_id
        
    except requests.exceptions.Timeout:
        print(f"⏰ Request timed out after 10 minutes")
        return None
    except requests.exceptions.RequestException as e:
        print(f"🌐 Request error: {e}")
        return None
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return None

def test_progress_api():
    """Test the progress tracking API"""
    print("\n🔍 Testing Progress Tracking API")
    print("-" * 40)
    
    # This would typically use a real session_id from an ongoing generation
    test_session_id = "test_session_123"
    
    try:
        response = requests.get(f"{PROGRESS_ENDPOINT}/{test_session_id}")
        
        if response.status_code == 200:
            progress_data = response.json()
            print(f"✅ Progress API working")
            print(f"Response: {json.dumps(progress_data, indent=2)}")
        else:
            print(f"⚠️  Progress API returned {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Progress API error: {e}")

def main():
    """Main test function"""
    print("🧪 Context-Limit-Transcending System Test Suite")
    print("=" * 60)
    print("This will test the ability to generate 100+ items that would")
    print("normally exceed LLM context limits by using intelligent chunking.")
    print()
    
    # Check if the API is accessible
    try:
        health_response = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ API server is accessible")
        else:
            print(f"⚠️  API server returned {health_response.status_code}")
    except:
        print("❌ API server is not accessible. Please start the server first.")
        print("   Run: python main.py or your FastAPI startup command")
        return
    
    print()
    
    # Run tests
    test_large_generation_api()
    test_progress_api()
    
    print("\n🏁 Test suite completed!")
    print("\nNext steps:")
    print("1. Check Redis for any remaining session data")
    print("2. Monitor system performance during large generations")
    print("3. Experiment with different chunk sizes for optimization")

if __name__ == "__main__":
    main()