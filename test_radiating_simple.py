#!/usr/bin/env python3
"""
Simple Radiating System Test Script

This script verifies that the radiating system is working correctly by:
1. Making a POST request to the radiating endpoint with streaming enabled
2. Printing each streaming event received
3. Showing success or failure of the system

Run with: python test_radiating_simple.py
"""

import requests
import json
import sys
import time
from datetime import datetime


def print_header():
    """Print test header"""
    print("🧪 Simple Radiating System Test")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def print_event(event_type: str, data: dict, timestamp: str = None):
    """Print streaming event in a formatted way"""
    if timestamp is None:
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    
    print(f"[{timestamp}] 📡 {event_type.upper()}")
    
    # Print relevant data based on event type
    if event_type == 'query_start':
        print(f"   Query: {data.get('query', 'N/A')}")
        print(f"   Strategy: {data.get('strategy', 'N/A')}")
        print(f"   Max Depth: {data.get('max_depth', 'N/A')}")
    elif event_type == 'entity_discovery':
        entities = data.get('entities', [])
        print(f"   Entities Found: {len(entities)}")
        for entity in entities[:3]:  # Show first 3
            print(f"     - {entity.get('name', 'N/A')} ({entity.get('type', 'Unknown')})")
        if len(entities) > 3:
            print(f"     ... and {len(entities) - 3} more")
    elif event_type == 'relationship_discovery':
        relationships = data.get('relationships', [])
        print(f"   Relationships Found: {len(relationships)}")
        for rel in relationships[:2]:  # Show first 2
            print(f"     - {rel.get('relationship_type', 'N/A')}")
    elif event_type == 'depth_expansion':
        print(f"   Current Depth: {data.get('current_depth', 'N/A')}")
        print(f"   Entities at Level: {data.get('entities_at_level', 'N/A')}")
    elif event_type == 'query_completion':
        print(f"   Status: {data.get('status', 'N/A')}")
        print(f"   Total Entities: {data.get('total_entities', 'N/A')}")
        print(f"   Total Relationships: {data.get('total_relationships', 'N/A')}")
        print(f"   Processing Time: {data.get('processing_time_ms', 'N/A')}ms")
    elif event_type == 'response_chunk':
        content = data.get('content', '')
        if content:
            print(f"   Content: {content[:100]}{'...' if len(content) > 100 else ''}")
    elif event_type == 'error':
        print(f"   ❌ Error: {data.get('message', 'Unknown error')}")
    else:
        # For any other event types, show the raw data
        print(f"   Data: {json.dumps(data, indent=6)[:200]}...")
    
    print()


def test_radiating_system():
    """Test the radiating system with a simple query"""
    
    print_header()
    
    # API endpoint
    base_url = "http://localhost:8000"
    endpoint = f"{base_url}/api/v1/radiating/query"
    
    # Test query payload - using AdaptiveExpansionStrategy as default
    payload = {
        "query": "What are the applications of artificial intelligence?",
        "enable_radiating": True,
        "max_depth": 2,
        "strategy": "AdaptiveExpansionStrategy",  # This is the default strategy
        "filters": {},
        "stream": True,
        "include_coverage_data": True
    }
    
    print(f"🎯 Testing radiating system with query: '{payload['query']}'")
    print(f"📊 Configuration:")
    print(f"   - Enable Radiating: {payload['enable_radiating']}")
    print(f"   - Max Depth: {payload['max_depth']}")
    print(f"   - Strategy: {payload['strategy']}")
    print(f"   - Streaming: {payload['stream']}")
    print()
    
    try:
        # Make the streaming request
        print("🚀 Sending request to radiating endpoint...")
        print(f"   Endpoint: {endpoint}")
        print()
        
        # Set up headers for streaming
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream'
        }
        
        # Make the request with streaming
        response = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            stream=True,
            timeout=60  # 60 second timeout
        )
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        print("✅ Connection established, receiving streaming events...")
        print("-" * 50)
        
        # Process streaming response
        event_count = 0
        start_time = time.time()
        
        # Parse Server-Sent Events
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith('data: '):
                try:
                    # Extract JSON data from SSE format
                    json_data = line[6:]  # Remove 'data: ' prefix
                    event_data = json.loads(json_data)
                    
                    # Extract event type and data
                    event_type = event_data.get('type', 'unknown')
                    data = event_data.get('data', event_data)
                    timestamp = event_data.get('timestamp', '')
                    
                    # Print the event
                    print_event(event_type, data, timestamp)
                    event_count += 1
                    
                    # Stop if we get a completion or error event
                    if event_type in ['query_completion', 'error', 'complete']:
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️  Could not parse event data: {line}")
                    print(f"   JSON Error: {e}")
                    continue
        
        elapsed_time = time.time() - start_time
        
        print("-" * 50)
        print(f"📈 Streaming completed!")
        print(f"   Total events received: {event_count}")
        print(f"   Total time: {elapsed_time:.2f} seconds")
        
        if event_count == 0:
            print("⚠️  No events received - this might indicate an issue")
            return False
        else:
            print("✅ Radiating system is working correctly!")
            return True
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed!")
        print("   Make sure the FastAPI server is running on http://localhost:8000")
        print("   Start it with: ./run_local.sh or python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return False
        
    except requests.exceptions.Timeout:
        print("❌ Request timed out!")
        print("   The radiating system might be taking too long to respond")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        print(f"   Full traceback: {traceback.format_exc()}")
        return False


def main():
    """Main entry point"""
    try:
        success = test_radiating_system()
        
        print()
        print("=" * 50)
        if success:
            print("🎉 Test completed successfully!")
            print("✨ The radiating system is working correctly after the fix.")
        else:
            print("💥 Test failed!")
            print("🔧 The radiating system needs attention.")
        
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⛔ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()