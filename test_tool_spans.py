#!/usr/bin/env python3
"""
Test Tool Spans in Langfuse

Creates a simple test to verify tool spans appear correctly in Langfuse.
"""

import asyncio
import sys
sys.path.insert(0, '/app')

async def test_tool_spans():
    """Test creating multiple tool spans"""
    
    print("🔧 Testing Tool Spans Creation")
    print("=" * 35)
    
    try:
        from app.core.langfuse_integration import get_tracer
        tracer = get_tracer()
        
        if not tracer.is_enabled():
            print("❌ Langfuse not enabled")
            return False
            
        # Create main trace
        trace = tracer.create_trace(
            name="test-multiple-tools",
            input="Testing multiple tool execution",
            metadata={"test": "tool_spans"}
        )
        
        if not trace:
            print("❌ Failed to create trace")
            return False
        print("✅ Created main trace")
        
        # Create agent span
        agent_span = tracer.create_agent_span(
            trace,
            "test_agent",
            "Test multiple tool usage",
            metadata={"test": "tool_spans"}
        )
        
        if not agent_span:
            print("❌ Failed to create agent span")
            return False
        print("✅ Created agent span")
        
        # Create multiple tool spans
        tools_to_test = [
            ("find_email", {"subject": "test email"}),
            ("read_email", {"email_id": "12345"}),
            ("gmail_send", {"to": "test@example.com", "subject": "test"})
        ]
        
        tool_spans = []
        for tool_name, params in tools_to_test:
            tool_span = tracer.create_tool_span(trace, tool_name, params)
            if tool_span:
                print(f"✅ Created tool span: {tool_name}")
                # End the span immediately with test result
                tracer.end_span_with_result(
                    tool_span,
                    {"status": "success", "tool": tool_name, "test": True},
                    success=True
                )
                print(f"✅ Ended tool span: {tool_name}")
                tool_spans.append(tool_span)
            else:
                print(f"❌ Failed to create tool span: {tool_name}")
        
        # End agent span
        tracer.end_span_with_result(
            agent_span,
            {"tools_used": [tool[0] for tool in tools_to_test], "status": "completed"},
            success=True
        )
        print("✅ Ended agent span")
        
        # Update trace
        trace.update(
            output=f"Test completed with {len(tool_spans)} tool spans",
            metadata={"tool_count": len(tool_spans)}
        )
        print("✅ Updated trace")
        
        # Flush
        tracer.flush()
        print("✅ Flushed traces")
        
        print(f"\n🎯 Results:")
        print(f"  📊 Main trace: Created")
        print(f"  👤 Agent span: Created")
        print(f"  🔧 Tool spans: {len(tool_spans)}/{len(tools_to_test)}")
        print(f"  💾 Flushed: Yes")
        
        if len(tool_spans) == len(tools_to_test):
            print(f"\n✅ All {len(tool_spans)} tool spans created successfully!")
            print("🔗 Check Langfuse UI for: 'test-multiple-tools' trace")
            return True
        else:
            print(f"\n⚠️ Only {len(tool_spans)}/{len(tools_to_test)} tool spans created")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def main():
    success = await test_tool_spans()
    print(f"\n🏆 Test Result: {'PASSED' if success else 'FAILED'}")
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)