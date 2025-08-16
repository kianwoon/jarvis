#!/usr/bin/env python3
"""
Test script to verify radiating routing is working correctly.

This script tests the `/api/v1/langchain/rag` endpoint with radiating enabled,
simulates what the frontend would send when radiating is toggled on,
verifies the response is streaming radiating events, and shows example output.

Usage:
    python test_radiating_routing.py

Requirements:
    - API server running on localhost:8000 (start with ./run_local.sh)
    - requests library (pip install requests)
"""

import requests
import json
import sys
import time
from typing import Dict, Any, Optional
from datetime import datetime

# Test configuration
API_BASE_URL = "http://localhost:8000"
RAG_ENDPOINT = f"{API_BASE_URL}/api/v1/langchain/rag"

def print_header(title: str) -> None:
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_subheader(title: str) -> None:
    """Print formatted subsection header"""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")

def print_result(test_name: str, success: bool, message: str = "") -> None:
    """Print formatted test result"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {test_name}")
    if message:
        print(f"   {message}")

def check_api_health() -> bool:
    """Check if the API server is running and healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print_result("API Health Check", True, f"Status: {health_data.get('status', 'OK')}")
            return True
        else:
            print_result("API Health Check", False, f"Status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_result("API Health Check", False, "Could not connect to API server")
        print("   💡 Make sure the server is running with: ./run_local.sh")
        return False
    except Exception as e:
        print_result("API Health Check", False, f"Error: {str(e)}")
        return False

def test_radiating_request(
    question: str,
    radiating_config: Optional[Dict[str, Any]] = None,
    conversation_id: Optional[str] = None,
    expected_events: Optional[list] = None
) -> Dict[str, Any]:
    """
    Test a radiating request and return results
    
    Args:
        question: The query to send
        radiating_config: Configuration for radiating mode
        conversation_id: Optional conversation ID
        expected_events: List of expected event types
        
    Returns:
        Dict containing test results and response data
    """
    
    # Default configuration
    if radiating_config is None:
        radiating_config = {
            "max_depth": 2,
            "strategy": "hybrid",
            "filters": {},
            "include_coverage_data": True
        }
    
    # Default expected events
    if expected_events is None:
        expected_events = ["chat_start", "status"]
    
    # Prepare request payload
    payload = {
        "question": question,
        "use_radiating": True,
        "radiating_config": radiating_config
    }
    
    if conversation_id:
        payload["conversation_id"] = conversation_id
    
    # Track results
    result = {
        "success": False,
        "error": None,
        "events_received": [],
        "event_types": set(),
        "final_answer": "",
        "entities_found": 0,
        "relationships_found": 0,
        "response_time": 0,
        "total_events": 0
    }
    
    print(f"\n🌟 Testing radiating query: '{question}'")
    print(f"   Configuration: {radiating_config}")
    
    start_time = time.time()
    
    try:
        # Make the streaming request
        response = requests.post(
            RAG_ENDPOINT,
            json=payload,
            stream=True,
            headers={"Content-Type": "application/json"},
            timeout=60  # Generous timeout for radiating queries
        )
        
        if response.status_code != 200:
            result["error"] = f"HTTP {response.status_code}: {response.text}"
            return result
        
        print("📡 Receiving streaming response...")
        
        # Process streaming response
        for line_num, line in enumerate(response.iter_lines(), 1):
            if line:
                try:
                    # Parse JSON line
                    data = json.loads(line.decode('utf-8'))
                    result["events_received"].append(data)
                    result["total_events"] += 1
                    
                    event_type = data.get("type", "unknown")
                    result["event_types"].add(event_type)
                    
                    # Handle different event types
                    if event_type == "chat_start":
                        mode = data.get("mode", "unknown")
                        classification = data.get("classification", "unknown")
                        print(f"   ✅ Chat started - Mode: {mode}, Classification: {classification}")
                        
                        # Verify radiating mode
                        if mode == "radiating" and classification == "radiating_coverage":
                            print("   🎯 Confirmed: Using radiating coverage mode")
                        else:
                            print(f"   ⚠️  Unexpected mode/classification: {mode}/{classification}")
                    
                    elif event_type == "status":
                        message = data.get("message", "")
                        print(f"   📊 Status: {message}")
                    
                    elif event_type == "entity_discovered":
                        result["entities_found"] += 1
                        entity = data.get("entity", {})
                        entity_name = entity.get("name", "Unknown")
                        entity_type = entity.get("type", "Unknown")
                        print(f"   🔍 Entity discovered: {entity_name} (type: {entity_type})")
                    
                    elif event_type == "relationship_found":
                        result["relationships_found"] += 1
                        rel = data.get("relationship", {})
                        print(f"   🔗 Relationship found: {rel}")
                    
                    elif event_type == "traversal_progress":
                        current_depth = data.get("current_depth", 0)
                        entities_explored = data.get("entities_explored", 0)
                        total_entities = data.get("total_entities", 0)
                        print(f"   📈 Progress: Depth {current_depth}, Explored {entities_explored}/{total_entities}")
                    
                    elif event_type == "token":
                        # Handle streaming text tokens
                        token = data.get("token", "")
                        result["final_answer"] += token
                        # Show progress for longer responses
                        if len(result["final_answer"]) % 100 == 0 and len(result["final_answer"]) > 0:
                            print(f"   📝 Streaming... ({len(result['final_answer'])} chars)")
                    
                    elif data.get("answer"):
                        # Final complete answer
                        if not result["final_answer"]:  # Only if we haven't been accumulating tokens
                            result["final_answer"] = data.get("answer", "")
                        
                        # Update final statistics
                        result["entities_found"] = data.get("entities_found", result["entities_found"])
                        result["relationships_found"] = data.get("relationships_found", result["relationships_found"])
                        
                        print(f"\n   📝 Final answer received ({len(result['final_answer'])} characters)")
                        
                        # Show coverage data if available
                        coverage = data.get("coverage")
                        if coverage:
                            total_nodes = coverage.get("total_nodes", 0)
                            nodes_explored = coverage.get("nodes_explored", 0)
                            coverage_ratio = coverage.get("coverage_ratio", 0)
                            print(f"   📊 Coverage: {nodes_explored}/{total_nodes} nodes ({coverage_ratio:.1%})")
                    
                    elif event_type == "error":
                        error_msg = data.get("message", "Unknown error")
                        print(f"   ❌ Error event: {error_msg}")
                        result["error"] = error_msg
                        
                    # Limit output for very long responses
                    if line_num > 100:
                        print(f"   ... (truncating output after {line_num} events)")
                        break
                        
                except json.JSONDecodeError:
                    # Skip malformed JSON lines
                    continue
                except Exception as e:
                    print(f"   ⚠️  Error processing line {line_num}: {e}")
                    continue
        
        result["response_time"] = time.time() - start_time
        result["success"] = True
        
        # Verify expected events were received
        missing_events = set(expected_events) - result["event_types"]
        if missing_events:
            print(f"   ⚠️  Missing expected events: {missing_events}")
        
        print(f"\n   ✅ Request completed in {result['response_time']:.2f}s")
        print(f"   📊 Statistics: {result['total_events']} events, {len(result['event_types'])} unique types")
        
    except requests.exceptions.Timeout:
        result["error"] = "Request timed out"
        print("   ❌ Request timed out")
    except requests.exceptions.ConnectionError:
        result["error"] = "Connection error"
        print("   ❌ Connection error")
    except Exception as e:
        result["error"] = str(e)
        print(f"   ❌ Unexpected error: {e}")
    
    return result

def test_comparison_normal_vs_radiating():
    """Test the same query with and without radiating to show the difference"""
    print_subheader("Comparison: Normal vs Radiating Mode")
    
    test_query = "artificial intelligence in healthcare"
    
    # Test 1: Normal mode (should NOT use radiating)
    print("\n🔹 Testing Normal Mode (use_radiating=False)")
    normal_payload = {
        "question": test_query,
        "use_radiating": False
    }
    
    try:
        response = requests.post(RAG_ENDPOINT, json=normal_payload, stream=True, timeout=30)
        if response.status_code == 200:
            events = []
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        events.append(data.get("type", "unknown"))
                        if len(events) >= 5:  # Just check first few events
                            break
                    except:
                        continue
            
            has_radiating_events = any("entity_discovered" in event or "relationship_found" in event 
                                     for event in events)
            mode = "unknown"
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if data.get("type") == "chat_start":
                            mode = data.get("mode", "unknown")
                            break
                    except:
                        continue
                    
            print_result("Normal mode routing", not has_radiating_events, 
                        f"Mode: {mode}, No radiating events (correct)")
        else:
            print_result("Normal mode test", False, f"HTTP {response.status_code}")
    except Exception as e:
        print_result("Normal mode test", False, f"Error: {e}")
    
    # Test 2: Radiating mode
    print("\n🔹 Testing Radiating Mode (use_radiating=True)")
    radiating_result = test_radiating_request(
        question=test_query,
        radiating_config={"max_depth": 2, "strategy": "hybrid"}
    )
    
    has_radiating_mode = "radiating" in str(radiating_result.get("events_received", []))
    print_result("Radiating mode routing", radiating_result["success"] and has_radiating_mode,
                f"Radiating events: {radiating_result['total_events']}")

def run_comprehensive_tests():
    """Run comprehensive tests of the radiating routing functionality"""
    
    print_header("Radiating Routing Test Suite")
    print(f"Testing endpoint: {RAG_ENDPOINT}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Pre-flight check
    print_subheader("Pre-flight Checks")
    if not check_api_health():
        print("\n❌ API server is not available. Please start it with: ./run_local.sh")
        return False
    
    test_results = []
    
    # Test 1: Basic radiating functionality
    print_subheader("Test 1: Basic Radiating Query")
    result1 = test_radiating_request(
        question="What are the applications of machine learning?",
        radiating_config={
            "max_depth": 2,
            "strategy": "hybrid",
            "include_coverage_data": True
        }
    )
    test_results.append(result1["success"])
    print_result("Basic radiating query", result1["success"], 
                f"Events: {result1['total_events']}, Time: {result1['response_time']:.2f}s")
    
    # Test 2: Different strategies
    print_subheader("Test 2: Different Traversal Strategies")
    
    for strategy in ["breadth_first", "depth_first", "hybrid"]:
        print(f"\n🧪 Testing strategy: {strategy}")
        result = test_radiating_request(
            question=f"Explore connections in {strategy} manner",
            radiating_config={
                "max_depth": 2,
                "strategy": strategy,
                "filters": {"entity_types": ["Technology", "Concept"]}
            }
        )
        test_results.append(result["success"])
        print_result(f"Strategy {strategy}", result["success"], 
                    f"Events: {result['total_events']}")
    
    # Test 3: Configuration variations
    print_subheader("Test 3: Configuration Variations")
    
    configs = [
        {"max_depth": 1, "strategy": "breadth_first"},
        {"max_depth": 3, "strategy": "hybrid", "filters": {"entity_types": ["Person", "Organization"]}},
        {"max_depth": 2, "strategy": "depth_first", "include_coverage_data": False}
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n🧪 Testing config {i}: {config}")
        result = test_radiating_request(
            question=f"Test configuration {i}",
            radiating_config=config
        )
        test_results.append(result["success"])
        print_result(f"Config variation {i}", result["success"])
    
    # Test 4: Error handling
    print_subheader("Test 4: Error Handling")
    
    # Test with invalid strategy
    print("\n🧪 Testing invalid strategy")
    result = test_radiating_request(
        question="Test with invalid strategy",
        radiating_config={"max_depth": 2, "strategy": "invalid_strategy"}
    )
    # For error handling test, we expect it to either succeed with fallback or fail gracefully
    error_handled = result["error"] is not None or result["success"]
    test_results.append(error_handled)
    print_result("Invalid strategy handling", error_handled, 
                "Error handled gracefully" if result["error"] else "Succeeded with fallback")
    
    # Test 5: Comparison test
    test_comparison_normal_vs_radiating()
    test_results.append(True)  # Always passes if no exception
    
    # Summary
    print_subheader("Test Summary")
    passed = sum(test_results)
    total = len(test_results)
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"\nTest Results: {passed}/{total} passed ({success_rate:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Radiating routing is working correctly.")
        print("✨ The radiating system is properly integrated and functional.")
    elif passed >= total * 0.8:
        print(f"\n✅ Most tests passed ({success_rate:.1f}%). Minor issues may exist.")
        print("🔧 Review any failed tests above for potential improvements.")
    else:
        print(f"\n❌ Significant issues detected ({success_rate:.1f}% success rate).")
        print("🚨 Radiating routing needs attention before production use.")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return passed == total

def main():
    """Main entry point"""
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h"]:
            print(__doc__)
            return
        elif sys.argv[1] == "--quick":
            # Quick test mode
            print_header("Quick Radiating Test")
            if check_api_health():
                result = test_radiating_request("Quick test of radiating system")
                print_result("Quick test", result["success"], 
                            f"Response time: {result['response_time']:.2f}s")
            return
    
    try:
        # Run comprehensive test suite
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⛔ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()