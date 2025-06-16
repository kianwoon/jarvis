#!/usr/bin/env python3
"""
Test Pipeline Langfuse Integration

This script tests the complete Langfuse integration for agentic pipelines,
ensuring that pipeline traces, agent spans, and tool spans are properly created.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/Users/kianwoonwong/Downloads/jarvis')

from app.core.pipeline_executor import PipelineExecutor
from app.core.langfuse_integration import tracer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_pipeline_langfuse_integration():
    """Test complete pipeline Langfuse integration"""
    
    print("🔍 Testing Pipeline Langfuse Integration")
    print("=" * 50)
    
    # Check if Langfuse is enabled
    if not tracer.is_enabled():
        print("⚠️ Langfuse tracing is not enabled - checking configuration...")
        from app.core.langfuse_settings_cache import get_langfuse_settings
        config = get_langfuse_settings()
        print(f"Langfuse config: enabled={config.get('enabled')}, host={config.get('host')}")
        if not config.get('enabled'):
            print("❌ Langfuse is disabled in configuration")
            return False
    else:
        print("✅ Langfuse tracing is enabled")
    
    # Initialize pipeline executor
    executor = PipelineExecutor()
    
    # Test with a simple 2-agent pipeline
    print("\n📋 Testing with simple 2-agent pipeline...")
    
    try:
        # Get available pipelines
        pipelines = await executor.pipeline_manager.get_all_pipelines()
        if not pipelines:
            print("❌ No pipelines found in database")
            return False
            
        print(f"📊 Found {len(pipelines)} pipelines:")
        for pipeline in pipelines:
            print(f"  - Pipeline {pipeline['id']}: {pipeline['name']} ({pipeline['collaboration_mode']})")
        
        # Use the first active pipeline
        test_pipeline = None
        for pipeline in pipelines:
            if pipeline['is_active']:
                test_pipeline = pipeline
                break
                
        if not test_pipeline:
            print("❌ No active pipelines found")
            return False
            
        print(f"\n🚀 Testing pipeline: {test_pipeline['name']} (ID: {test_pipeline['id']})")
        
        # Test input
        test_input = {
            "query": "Test pipeline Langfuse integration - analyze the current status of our system and provide recommendations",
            "context": {
                "test_run": True,
                "integration_test": "langfuse_pipeline"
            }
        }
        
        print(f"📝 Test query: {test_input['query'][:100]}...")
        
        # Execute pipeline and track events
        print("\n🔄 Executing pipeline...")
        
        execution_events = []
        langfuse_operations = []
        
        # Track Langfuse operations
        original_create_trace = tracer.create_trace
        original_create_agent_span = tracer.create_agent_span
        original_create_tool_span = tracer.create_tool_span
        original_end_span = tracer.end_span_with_result
        
        def track_create_trace(*args, **kwargs):
            result = original_create_trace(*args, **kwargs)
            langfuse_operations.append(f"✅ Created main trace: {kwargs.get('name', 'unnamed')}")
            return result
            
        def track_create_agent_span(*args, **kwargs):
            result = original_create_agent_span(*args, **kwargs)
            langfuse_operations.append(f"✅ Created agent span: {kwargs.get('agent_name', args[1] if len(args) > 1 else 'unnamed')}")
            return result
            
        def track_create_tool_span(*args, **kwargs):
            result = original_create_tool_span(*args, **kwargs)
            tool_name = args[1] if len(args) > 1 else kwargs.get('tool_name', 'unnamed')
            langfuse_operations.append(f"✅ Created tool span: {tool_name}")
            return result
            
        def track_end_span(*args, **kwargs):
            result = original_end_span(*args, **kwargs)
            success = kwargs.get('success', True)
            status = "success" if success else "error"
            langfuse_operations.append(f"✅ Ended span with {status}")
            return result
        
        # Patch tracer methods
        tracer.create_trace = track_create_trace
        tracer.create_agent_span = track_create_agent_span
        tracer.create_tool_span = track_create_tool_span
        tracer.end_span_with_result = track_end_span
        
        try:
            # Execute pipeline
            result = await executor.execute_pipeline(
                pipeline_id=test_pipeline['id'],
                input_data=test_input,
                trigger_type="integration_test"
            )
            
            execution_events.append("✅ Pipeline execution completed")
            
        except Exception as e:
            execution_events.append(f"❌ Pipeline execution failed: {e}")
            logger.error(f"Pipeline execution error: {e}")
            
        finally:
            # Restore original methods
            tracer.create_trace = original_create_trace
            tracer.create_agent_span = original_create_agent_span
            tracer.create_tool_span = original_create_tool_span
            tracer.end_span_with_result = original_end_span
        
        # Display results
        print("\n📊 Execution Events:")
        for event in execution_events:
            print(f"  {event}")
            
        print("\n🔗 Langfuse Operations:")
        for operation in langfuse_operations:
            print(f"  {operation}")
            
        # Analysis
        print("\n📈 Integration Analysis:")
        
        # Check for required Langfuse operations
        has_main_trace = any("main trace" in op for op in langfuse_operations)
        has_agent_spans = any("agent span" in op for op in langfuse_operations)
        has_tool_spans = any("tool span" in op for op in langfuse_operations)
        has_ended_spans = any("Ended span" in op for op in langfuse_operations)
        
        print(f"  📋 Main trace created: {'✅' if has_main_trace else '❌'}")
        print(f"  👥 Agent spans created: {'✅' if has_agent_spans else '❌'}")
        print(f"  🔧 Tool spans created: {'✅' if has_tool_spans else '❌'}")
        print(f"  🏁 Spans properly ended: {'✅' if has_ended_spans else '❌'}")
        
        # Overall assessment
        integration_score = sum([has_main_trace, has_agent_spans, has_tool_spans, has_ended_spans])
        
        print(f"\n🎯 Integration Score: {integration_score}/4")
        
        if integration_score == 4:
            print("🎉 EXCELLENT: Complete Langfuse integration working!")
        elif integration_score >= 3:
            print("✅ GOOD: Most Langfuse features working")
        elif integration_score >= 2:
            print("⚠️ PARTIAL: Some Langfuse features working")
        else:
            print("❌ POOR: Langfuse integration needs work")
            
        return integration_score >= 3
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logger.error(f"Test error: {e}", exc_info=True)
        return False

async def main():
    """Main test function"""
    print("🧪 Pipeline Langfuse Integration Test")
    print("====================================")
    
    success = await test_pipeline_langfuse_integration()
    
    print(f"\n🏆 Test Result: {'PASSED' if success else 'FAILED'}")
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)