#!/usr/bin/env python3
"""
Simple Langfuse Integration Test

Tests the basic Langfuse integration functionality.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/Users/kianwoonwong/Downloads/jarvis')

from app.core.langfuse_integration import tracer
from app.core.langfuse_settings_cache import get_langfuse_settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_basic_langfuse():
    """Test basic Langfuse functionality"""
    
    print("🔍 Testing Basic Langfuse Integration")
    print("=" * 40)
    
    # Check Langfuse configuration
    config = get_langfuse_settings()
    print(f"📋 Langfuse config: enabled={config.get('enabled')}, host={config.get('host')}")
    
    if not config.get('enabled'):
        print("❌ Langfuse is disabled in configuration")
        return False
    
    # Test tracer initialization
    if not tracer.is_enabled():
        print("⚠️ Tracer not enabled, attempting to initialize...")
        success = tracer.initialize()
        if not success:
            print("❌ Failed to initialize tracer")
            return False
    
    print("✅ Langfuse tracer is enabled")
    
    # Test trace creation
    print("\n🔄 Testing trace creation...")
    try:
        test_trace = tracer.create_trace(
            name="test-pipeline-integration",
            input="Test input for pipeline integration",
            metadata={
                "test_type": "integration_test",
                "component": "pipeline_langfuse",
                "version": "1.0"
            }
        )
        
        if test_trace:
            print("✅ Successfully created test trace")
            
            # Test agent span creation
            print("🔄 Testing agent span creation...")
            agent_span = tracer.create_agent_span(
                test_trace,
                "test_agent",
                "Test query for agent",
                metadata={
                    "pipeline_id": "test_pipeline",
                    "agent_index": 0,
                    "total_agents": 2
                }
            )
            
            if agent_span:
                print("✅ Successfully created agent span")
                
                # Test tool span creation
                print("🔄 Testing tool span creation...")
                tool_span = tracer.create_tool_span(
                    test_trace,
                    "test_tool",
                    {
                        "param1": "value1",
                        "param2": "value2",
                        "agent": "test_agent"
                    }
                )
                
                if tool_span:
                    print("✅ Successfully created tool span")
                    
                    # Test span ending
                    print("🔄 Testing span completion...")
                    
                    # End tool span
                    tracer.end_span_with_result(
                        tool_span,
                        {"result": "Tool executed successfully", "status": "success"},
                        success=True
                    )
                    print("✅ Successfully ended tool span")
                    
                    # End agent span  
                    tracer.end_span_with_result(
                        agent_span,
                        {"response": "Agent completed successfully", "tokens": 150},
                        success=True
                    )
                    print("✅ Successfully ended agent span")
                    
                else:
                    print("❌ Failed to create tool span")
                    return False
            else:
                print("❌ Failed to create agent span")
                return False
        else:
            print("❌ Failed to create test trace")
            return False
            
        # Test pipeline workflow span
        print("🔄 Testing pipeline workflow span...")
        workflow_span = tracer.create_multi_agent_workflow_span(
            test_trace,
            "sequential",
            ["agent1", "agent2"]
        )
        
        if workflow_span:
            print("✅ Successfully created workflow span")
            
            # End workflow span
            tracer.end_span_with_result(
                workflow_span,
                {
                    "pipeline_id": "test_pipeline",
                    "total_agents": 2,
                    "collaboration_mode": "sequential",
                    "status": "completed"
                },
                success=True
            )
            print("✅ Successfully ended workflow span")
        else:
            print("❌ Failed to create workflow span")
            return False
            
        # Flush traces
        print("🔄 Flushing traces...")
        tracer.flush()
        print("✅ Successfully flushed traces")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logger.error(f"Test error: {e}", exc_info=True)
        return False

async def main():
    """Main test function"""
    print("🧪 Simple Langfuse Integration Test")
    print("===================================")
    
    success = await test_basic_langfuse()
    
    print(f"\n🏆 Test Result: {'PASSED' if success else 'FAILED'}")
    
    if success:
        print("\n🎉 Langfuse integration is working correctly!")
        print("✅ Traces can be created")
        print("✅ Agent spans can be created")
        print("✅ Tool spans can be created")
        print("✅ Workflow spans can be created")
        print("✅ Spans can be ended properly")
        print("✅ Traces can be flushed")
        print("\n💡 The pipeline Langfuse integration should now work correctly.")
    else:
        print("\n❌ Langfuse integration has issues that need to be resolved.")
        
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)