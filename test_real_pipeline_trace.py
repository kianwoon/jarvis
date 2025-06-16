#!/usr/bin/env python3
"""
Test Real Pipeline Trace

Tests creating an actual pipeline trace that should appear in Langfuse.
"""

import asyncio
import sys
import json

# Add the project root to Python path
sys.path.insert(0, '/app')

from app.core.langfuse_integration import tracer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_real_pipeline_trace():
    """Test creating a real pipeline trace"""
    
    print("🔍 Testing Real Pipeline Trace Creation")
    print("=" * 45)
    
    try:
        # Create a main pipeline trace
        print("🔄 Creating main pipeline trace...")
        main_trace = tracer.create_trace(
            name="agentic-pipeline-sequential",
            input="Test query: Analyze customer inquiries and respond appropriately",
            metadata={
                "pipeline_id": "5",
                "execution_id": "test_175",
                "collaboration_mode": "sequential", 
                "trigger_type": "manual_test",
                "total_agents": 2
            }
        )
        
        if not main_trace:
            print("❌ Failed to create main trace")
            return False
            
        print("✅ Created main pipeline trace")
        
        # Create pipeline workflow span
        print("🔄 Creating pipeline workflow span...")
        workflow_span = tracer.create_multi_agent_workflow_span(
            main_trace,
            "sequential",
            ["customer service 1", "customer service 2"]
        )
        
        if not workflow_span:
            print("❌ Failed to create workflow span")
            return False
            
        print("✅ Created pipeline workflow span")
        
        # Simulate agent 1 execution
        print("🔄 Simulating Agent 1 execution...")
        agent1_span = tracer.create_agent_span(
            main_trace,
            "customer service 1", 
            "Test query: Analyze customer inquiries and respond appropriately",
            metadata={
                "pipeline_id": "5",
                "execution_id": "test_175",
                "agent_index": 0,
                "total_agents": 2,
                "tools_available": ["get_latest_emails", "find_email", "read_email"]
            }
        )
        
        if not agent1_span:
            print("❌ Failed to create agent 1 span")
            return False
            
        print("✅ Created agent 1 span")
        
        # Simulate tool execution within agent 1
        print("🔄 Simulating tool execution for Agent 1...")
        tool_span = tracer.create_tool_span(
            main_trace,
            "get_latest_emails",
            {
                "limit": 10,
                "folder": "INBOX",
                "agent": "customer service 1"
            }
        )
        
        if not tool_span:
            print("❌ Failed to create tool span")
            return False
            
        print("✅ Created tool span")
        
        # End tool span with result
        print("🔄 Ending tool span with result...")
        tracer.end_span_with_result(
            tool_span,
            {
                "emails_found": 5,
                "subjects": ["Customer Inquiry #123", "Follow-up needed", "Service Request"],
                "status": "success"
            },
            success=True
        )
        print("✅ Ended tool span")
        
        # End agent 1 span
        print("🔄 Ending agent 1 span...")
        tracer.end_span_with_result(
            agent1_span,
            {
                "response": "Found 5 recent customer emails. The inquiries include support requests and follow-ups.",
                "tools_used": ["get_latest_emails"],
                "tokens_used": 125,
                "duration": 2.3
            },
            success=True
        )
        print("✅ Ended agent 1 span")
        
        # Simulate agent 2 execution  
        print("🔄 Simulating Agent 2 execution...")
        agent2_span = tracer.create_agent_span(
            main_trace,
            "customer service 2",
            "Based on previous analysis: Found 5 recent customer emails. Please follow up appropriately.",
            metadata={
                "pipeline_id": "5", 
                "execution_id": "test_175",
                "agent_index": 1,
                "total_agents": 2,
                "tools_available": ["gmail_send", "draft_email"]
            }
        )
        
        if not agent2_span:
            print("❌ Failed to create agent 2 span")
            return False
            
        print("✅ Created agent 2 span")
        
        # Simulate tool execution for agent 2
        print("🔄 Simulating tool execution for Agent 2...")
        tool2_span = tracer.create_tool_span(
            main_trace,
            "gmail_send", 
            {
                "to": "support@example.com",
                "subject": "Follow-Up on Customer Inquiries", 
                "body": "Dear Team, Following up on recent customer inquiries...",
                "agent": "customer service 2"
            }
        )
        
        if not tool2_span:
            print("❌ Failed to create tool2 span")
            return False
            
        print("✅ Created tool2 span")
        
        # End tool2 span
        print("🔄 Ending tool2 span...")
        tracer.end_span_with_result(
            tool2_span,
            {
                "message_id": "msg_12345",
                "status": "sent",
                "recipient": "support@example.com"
            },
            success=True
        )
        print("✅ Ended tool2 span")
        
        # End agent 2 span
        print("🔄 Ending agent 2 span...")
        tracer.end_span_with_result(
            agent2_span,
            {
                "response": "Successfully sent follow-up email to support team regarding customer inquiries.",
                "tools_used": ["gmail_send"],
                "tokens_used": 89,
                "duration": 1.8
            },
            success=True
        )
        print("✅ Ended agent 2 span")
        
        # End workflow span
        print("🔄 Ending workflow span...")
        tracer.end_span_with_result(
            workflow_span,
            {
                "pipeline_id": "5",
                "execution_id": "test_175", 
                "total_agents": 2,
                "completed_agents": 2,
                "collaboration_mode": "sequential",
                "final_output": "Pipeline completed successfully. Analyzed customer emails and sent follow-up."
            },
            success=True
        )
        print("✅ Ended workflow span")
        
        # Flush traces to ensure they're sent
        print("🔄 Flushing traces to Langfuse...")
        tracer.flush()
        print("✅ Flushed traces")
        
        print("\n🎉 Successfully created complete pipeline trace!")
        print("📊 Trace includes:")
        print("  ✅ Main pipeline trace")
        print("  ✅ Pipeline workflow span")
        print("  ✅ 2 Agent spans")
        print("  ✅ 2 Tool spans")
        print("  ✅ All spans properly ended")
        
        print(f"\n🔗 Check Langfuse UI at: http://localhost:3000")
        print(f"📋 Look for trace: 'agentic-pipeline-sequential'")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logger.error(f"Test error: {e}", exc_info=True)
        return False

async def main():
    """Main test function"""
    print("🧪 Real Pipeline Trace Test")
    print("==========================")
    
    success = await test_real_pipeline_trace()
    
    print(f"\n🏆 Test Result: {'PASSED' if success else 'FAILED'}")
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)