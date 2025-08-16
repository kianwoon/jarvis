#!/usr/bin/env python3
"""
Simple Radiating System Verification Script

This script verifies that the radiating system is working correctly after recent fixes.
It performs a minimal test with streaming enabled to check if:
1. The streaming endpoint responds correctly
2. Events are being received in proper format
3. Entities are being discovered
4. No critical errors occur

Usage:
    python verify_radiating_fix.py

Expected output:
    - Clear pass/fail result
    - Event types received during streaming
    - Entity discovery status
    - Any errors encountered
"""

import asyncio
import aiohttp
import json
import time
import sys
from datetime import datetime
from typing import Dict, List, Any


class RadiatingVerifier:
    """Simple verifier for the radiating system fixes"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.events_received = []
        self.entities_discovered = []
        self.errors = []
        
    async def test_health_check(self) -> bool:
        """Test if the radiating health endpoint is responsive"""
        print("🏥 Testing radiating health endpoint...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/v1/radiating/health", timeout=5) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        is_healthy = health_data.get('healthy', False)
                        
                        if is_healthy:
                            print("✅ Health check: PASSED")
                            return True
                        else:
                            print(f"❌ Health check: FAILED - System not healthy: {health_data}")
                            return False
                    else:
                        print(f"❌ Health check: FAILED - HTTP {response.status}")
                        return False
                        
        except Exception as e:
            print(f"❌ Health check: FAILED - {str(e)}")
            return False
    
    async def test_streaming_query(self) -> bool:
        """Test the main streaming radiating query functionality"""
        print("\n🚀 Testing streaming radiating query...")
        
        # Test configuration - minimal settings
        query_data = {
            "query": "What is artificial intelligence?",
            "enable_radiating": True,
            "max_depth": 2,
            "strategy": "hybrid",
            "stream": True,
            "include_coverage_data": False
        }
        
        print(f"📝 Query: '{query_data['query']}'")
        print(f"⚙️ Config: depth={query_data['max_depth']}, strategy={query_data['strategy']}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/v1/radiating/query",
                    json=query_data,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"❌ Query failed with HTTP {response.status}: {error_text}")
                        return False
                    
                    # Check if it's actually streaming
                    content_type = response.headers.get('content-type', '')
                    if 'text/event-stream' not in content_type:
                        print(f"❌ Expected streaming response, got: {content_type}")
                        return False
                    
                    print("📡 Receiving streaming events...")
                    
                    event_count = 0
                    start_time = time.time()
                    
                    # Read streaming response
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        if line.startswith('data: '):
                            event_data = line[6:]  # Remove 'data: ' prefix
                            
                            try:
                                event = json.loads(event_data)
                                event_count += 1
                                self.events_received.append(event)
                                
                                event_type = event.get('type', 'unknown')
                                print(f"  📦 Event {event_count}: {event_type}")
                                
                                # Check for entities
                                if 'entities' in event and event['entities']:
                                    entities = event['entities']
                                    self.entities_discovered.extend(entities)
                                    print(f"    🔍 Found {len(entities)} entities")
                                
                                # Check for errors
                                if event_type == 'error':
                                    error_msg = event.get('message', 'Unknown error')
                                    self.errors.append(error_msg)
                                    print(f"    ❌ Error: {error_msg}")
                                
                                # Stop after reasonable number of events or completion
                                if event_type in ['completed', 'final'] or event_count >= 20:
                                    break
                                    
                            except json.JSONDecodeError as e:
                                print(f"    ⚠️ Failed to parse event: {event_data[:100]}...")
                                self.errors.append(f"JSON decode error: {str(e)}")
                    
                    elapsed = time.time() - start_time
                    print(f"⏱️ Streaming completed in {elapsed:.2f} seconds")
                    
                    return event_count > 0
                    
        except asyncio.TimeoutError:
            print("❌ Query timed out after 30 seconds")
            return False
        except Exception as e:
            print(f"❌ Query failed: {str(e)}")
            self.errors.append(str(e))
            return False
    
    def analyze_results(self) -> bool:
        """Analyze the results and determine if the system is working"""
        print("\n📊 RESULTS ANALYSIS")
        print("=" * 50)
        
        # Basic metrics
        total_events = len(self.events_received)
        total_entities = len(self.entities_discovered)
        total_errors = len(self.errors)
        
        print(f"📦 Events received: {total_events}")
        print(f"🔍 Entities discovered: {total_entities}")
        print(f"❌ Errors encountered: {total_errors}")
        
        # Event type analysis
        if self.events_received:
            event_types = {}
            for event in self.events_received:
                event_type = event.get('type', 'unknown')
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            print(f"\n📋 Event types received:")
            for event_type, count in event_types.items():
                print(f"  - {event_type}: {count}")
        
        # Entity analysis
        if self.entities_discovered:
            print(f"\n🎯 Sample entities discovered:")
            for i, entity in enumerate(self.entities_discovered[:5]):
                name = entity.get('name', 'Unknown')
                relevance = entity.get('relevance_score', 0)
                print(f"  {i+1}. {name} (relevance: {relevance:.3f})")
        
        # Error analysis
        if self.errors:
            print(f"\n⚠️ Errors encountered:")
            for i, error in enumerate(self.errors[:3], 1):
                print(f"  {i}. {error}")
        
        # Determine success criteria
        success_criteria = {
            "events_received": total_events > 0,
            "no_critical_errors": total_errors == 0 or all('timeout' not in str(e).lower() for e in self.errors),
            "entities_discovered": total_entities > 0,
            "proper_streaming": any(event.get('type') in ['progress', 'entity', 'relationship', 'completed'] 
                                  for event in self.events_received)
        }
        
        print(f"\n✅ SUCCESS CRITERIA:")
        for criterion, passed in success_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  - {criterion.replace('_', ' ').title()}: {status}")
        
        overall_success = all(success_criteria.values())
        return overall_success
    
    def print_verdict(self, health_ok: bool, streaming_ok: bool, analysis_ok: bool):
        """Print the final verdict"""
        print("\n" + "=" * 60)
        print("🎯 FINAL VERDICT")
        print("=" * 60)
        
        if health_ok and streaming_ok and analysis_ok:
            print("🎉 RADIATING SYSTEM: WORKING CORRECTLY")
            print("✅ All tests passed successfully!")
            print("✅ The fixes appear to be working as expected.")
            print("\nThe radiating system is ready for use with:")
            print("  - Streaming functionality working")
            print("  - Entity discovery operational")
            print("  - Health endpoints responsive")
            return True
        else:
            print("❌ RADIATING SYSTEM: ISSUES DETECTED")
            print("\nFailure summary:")
            if not health_ok:
                print("  ❌ Health check failed")
            if not streaming_ok:
                print("  ❌ Streaming query failed")
            if not analysis_ok:
                print("  ❌ Results analysis failed")
            
            print("\n🔧 Recommended actions:")
            print("  1. Check service logs for detailed error information")
            print("  2. Verify all required services are running (Neo4j, Redis)")
            print("  3. Ensure database schema is up to date")
            print("  4. Review recent code changes for potential issues")
            return False


async def main():
    """Main verification function"""
    print("🔍 RADIATING SYSTEM VERIFICATION")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Purpose: Verify radiating system fixes are working")
    
    verifier = RadiatingVerifier()
    
    try:
        # Run tests
        health_ok = await verifier.test_health_check()
        streaming_ok = await verifier.test_streaming_query() if health_ok else False
        analysis_ok = verifier.analyze_results() if streaming_ok else False
        
        # Print final verdict
        success = verifier.print_verdict(health_ok, streaming_ok, analysis_ok)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())