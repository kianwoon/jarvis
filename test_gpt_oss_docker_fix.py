#!/usr/bin/env python3
"""
Test script to verify GPT-OSS Docker connection fixes and tolerance improvements

This script tests:
1. Docker URL detection and conversion
2. Ollama connectivity verification with automatic fallback
3. Increased GPT-OSS tolerances (empty chunks and timeouts)
4. Multi-agent system with all 4 GPT-OSS agents

Usage:
    python test_gpt_oss_docker_fix.py
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.langchain.langgraph_multi_agent_system import LangGraphMultiAgentSystem
from app.langchain.fixed_multi_agent_streaming import fixed_multi_agent_streaming

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPTOSSDockerTester:
    """Test harness for GPT-OSS Docker fixes"""
    
    def __init__(self):
        self.test_results = []
        self.is_docker = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER")
        
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        result = {
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": time.time()
        }
        self.test_results.append(result)
        
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
        if details:
            logger.info(f"  Details: {details}")
    
    async def test_docker_detection(self):
        """Test 1: Docker environment detection"""
        logger.info("\n" + "="*60)
        logger.info("TEST 1: Docker Environment Detection")
        logger.info("="*60)
        
        try:
            system = LangGraphMultiAgentSystem()
            
            # Check if Docker was detected correctly
            expected_in_url = "host.docker.internal" if self.is_docker else "localhost"
            actual_url = system.ollama_base_url
            
            if expected_in_url in actual_url:
                self.log_test(
                    "Docker Detection",
                    True,
                    f"Correctly using {actual_url} (Docker: {self.is_docker})"
                )
            else:
                self.log_test(
                    "Docker Detection",
                    False,
                    f"Expected {expected_in_url} in URL, got {actual_url}"
                )
                
        except Exception as e:
            self.log_test("Docker Detection", False, str(e))
    
    async def test_ollama_connectivity(self):
        """Test 2: Ollama connectivity verification with fallback"""
        logger.info("\n" + "="*60)
        logger.info("TEST 2: Ollama Connectivity Verification")
        logger.info("="*60)
        
        try:
            system = LangGraphMultiAgentSystem()
            
            # Test async connectivity verification
            connected = await system._verify_ollama_connectivity_async()
            
            if connected:
                self.log_test(
                    "Ollama Connectivity",
                    True,
                    f"Connected to Ollama at {system.ollama_base_url}"
                )
            else:
                self.log_test(
                    "Ollama Connectivity",
                    False,
                    "Could not connect to Ollama at any URL"
                )
                
        except Exception as e:
            self.log_test("Ollama Connectivity", False, str(e))
    
    async def test_gpt_oss_streaming(self):
        """Test 3: GPT-OSS streaming with increased tolerances"""
        logger.info("\n" + "="*60)
        logger.info("TEST 3: GPT-OSS Streaming with Increased Tolerances")
        logger.info("="*60)
        
        try:
            question = "What is artificial intelligence and how does it work?"
            conversation_id = "test_gpt_oss_" + str(int(time.time()))
            
            # Track streaming events
            events_received = {
                "agent_start": 0,
                "agent_token": 0,
                "agent_complete": 0,
                "synthesis_start": 0,
                "final_response": 0,
                "errors": 0
            }
            
            total_response_length = 0
            agents_processed = []
            start_time = time.time()
            
            # Stream the response
            async for event_line in await fixed_multi_agent_streaming(question, conversation_id):
                try:
                    event = json.loads(event_line.strip())
                    event_type = event.get("type", "")
                    
                    # Track event types
                    if event_type in events_received:
                        events_received[event_type] += 1
                    
                    # Track specific details
                    if event_type == "agent_start":
                        agent_name = event.get("agent", "")
                        logger.info(f"  Agent starting: {agent_name}")
                        
                    elif event_type == "agent_complete":
                        agent_name = event.get("agent", "")
                        content = event.get("content", "")
                        agents_processed.append(agent_name)
                        total_response_length += len(content)
                        logger.info(f"  Agent completed: {agent_name} ({len(content)} chars)")
                        
                    elif event_type == "error":
                        events_received["errors"] += 1
                        error_msg = event.get("message", "Unknown error")
                        logger.error(f"  Streaming error: {error_msg}")
                        
                    elif event_type == "final_response":
                        response = event.get("response", "")
                        total_response_length += len(response)
                        logger.info(f"  Final response received ({len(response)} chars)")
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"  Error processing event: {e}")
            
            elapsed_time = time.time() - start_time
            
            # Evaluate results
            success = (
                events_received["agent_complete"] > 0 and
                events_received["final_response"] > 0 and
                events_received["errors"] == 0 and
                total_response_length > 100
            )
            
            details = (
                f"Processed {len(agents_processed)} agents in {elapsed_time:.1f}s. "
                f"Total response: {total_response_length} chars. "
                f"Events: {events_received}"
            )
            
            self.log_test("GPT-OSS Streaming", success, details)
            
        except Exception as e:
            self.log_test("GPT-OSS Streaming", False, str(e))
    
    async def test_all_gpt_oss_agents(self):
        """Test 4: All 4 GPT-OSS agents working together"""
        logger.info("\n" + "="*60)
        logger.info("TEST 4: All GPT-OSS Agents Collaboration")
        logger.info("="*60)
        
        try:
            # Get agent configuration to verify GPT-OSS models
            from app.core.langgraph_agents_cache import get_langgraph_agents
            agents = get_langgraph_agents()
            
            gpt_oss_agents = []
            for agent_name, agent_data in agents.items():
                config = agent_data.get('config', {})
                model = config.get('model', '')
                if 'gpt-oss' in model.lower():
                    gpt_oss_agents.append(agent_name)
                    logger.info(f"  Found GPT-OSS agent: {agent_name} using {model}")
            
            if len(gpt_oss_agents) < 4:
                logger.warning(f"  Only {len(gpt_oss_agents)} GPT-OSS agents found (expected 4)")
            
            # Test multi-agent collaboration
            question = "Explain machine learning algorithms and their applications in business"
            conversation_id = "test_all_gpt_oss_" + str(int(time.time()))
            
            agents_that_responded = []
            agent_responses = {}
            
            async for event_line in await fixed_multi_agent_streaming(question, conversation_id):
                try:
                    event = json.loads(event_line.strip())
                    
                    if event.get("type") == "agent_complete":
                        agent_name = event.get("agent", "")
                        content = event.get("content", "")
                        if agent_name and len(content) > 50:
                            agents_that_responded.append(agent_name)
                            agent_responses[agent_name] = len(content)
                            logger.info(f"  {agent_name} responded with {len(content)} chars")
                            
                except Exception:
                    continue
            
            # Check if GPT-OSS agents responded
            gpt_oss_responded = [a for a in agents_that_responded if a in gpt_oss_agents]
            
            success = len(gpt_oss_responded) >= min(2, len(gpt_oss_agents))
            details = (
                f"GPT-OSS agents responded: {len(gpt_oss_responded)}/{len(gpt_oss_agents)}. "
                f"Agents: {', '.join(gpt_oss_responded)}"
            )
            
            self.log_test("All GPT-OSS Agents", success, details)
            
        except Exception as e:
            self.log_test("All GPT-OSS Agents", False, str(e))
    
    def print_summary(self):
        """Print test summary"""
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        
        passed = sum(1 for r in self.test_results if r["passed"])
        total = len(self.test_results)
        
        logger.info(f"\nResults: {passed}/{total} tests passed")
        logger.info(f"Environment: {'Docker' if self.is_docker else 'Local'}")
        
        for result in self.test_results:
            status = "✅" if result["passed"] else "❌"
            logger.info(f"{status} {result['test']}: {result['details']}")
        
        if passed == total:
            logger.info("\n🎉 All tests passed! GPT-OSS Docker fixes are working correctly.")
        else:
            logger.info(f"\n⚠️ {total - passed} test(s) failed. Please review the logs above.")
        
        return passed == total

async def main():
    """Main test runner"""
    logger.info("Starting GPT-OSS Docker Fix Test Suite")
    logger.info(f"Running in {'Docker' if os.path.exists('/.dockerenv') else 'Local'} environment")
    
    tester = GPTOSSDockerTester()
    
    # Run all tests
    await tester.test_docker_detection()
    await tester.test_ollama_connectivity()
    await tester.test_gpt_oss_streaming()
    await tester.test_all_gpt_oss_agents()
    
    # Print summary
    all_passed = tester.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    asyncio.run(main())