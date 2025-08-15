#!/usr/bin/env python3
"""
Test script to verify that agents no longer output their system prompts/instructions
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.db import get_db_session, LangGraphAgent as LanggraphAgent

def check_linkedin_writer_prompt():
    """Check the LinkedIn writer agent's system prompt"""
    with get_db_session() as db:
        agent = db.query(LanggraphAgent).filter(LanggraphAgent.name == "Linkedin writer").first()
        if agent:
            print(f"✅ Found LinkedIn writer agent")
            print(f"📊 System prompt length: {len(agent.system_prompt)} characters")
            print(f"🔍 First 200 chars of system prompt:")
            print("-" * 50)
            print(agent.system_prompt[:200])
            print("-" * 50)
            
            # Check for problematic patterns
            problematic_patterns = [
                "## Your role",
                "## Content Style",
                "## Content Creation Process",
                "### Step 1:",
                "### Step 2:",
                "**Hook (First 2 lines)**",
                "**Reality Check Section**"
            ]
            
            found_patterns = []
            for pattern in problematic_patterns:
                if pattern in agent.system_prompt:
                    found_patterns.append(pattern)
            
            if found_patterns:
                print(f"⚠️  System prompt contains instruction patterns that might be exposed:")
                for pattern in found_patterns:
                    print(f"   - {pattern}")
                print("\n✅ But with the fix, these should now be hidden from output")
            else:
                print("✅ No problematic patterns found")
        else:
            print("❌ LinkedIn writer agent not found in database")

async def test_agent_response():
    """Test that the agent doesn't output its instructions"""
    import httpx
    
    # Test endpoint with LinkedIn writer
    url = "http://localhost:8000/api/v1/langchain/agent-chat"
    
    test_request = {
        "question": "Write a short LinkedIn post about the importance of AI ethics",
        "selected_agent": "Linkedin writer",
        "conversation_id": "test-" + str(os.getpid())
    }
    
    print("\n🧪 Testing agent response...")
    print(f"📝 Request: {test_request['question']}")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, json=test_request)
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "")
                
                print("\n📤 Response received:")
                print("-" * 50)
                print(answer[:500] if len(answer) > 500 else answer)
                print("-" * 50)
                
                # Check if response contains instruction patterns
                instruction_patterns = [
                    "## Your role",
                    "## Content Style",
                    "## Content Creation Process",
                    "### Step 1",
                    "### Step 2",
                    "<system_instructions>",
                    "internal guidance",
                    "Please respond directly to the user"
                ]
                
                leaked_patterns = []
                for pattern in instruction_patterns:
                    if pattern.lower() in answer.lower():
                        leaked_patterns.append(pattern)
                
                if leaked_patterns:
                    print("\n❌ WARNING: Response contains instruction patterns:")
                    for pattern in leaked_patterns:
                        print(f"   - {pattern}")
                    print("\n⚠️  The fix may not be working correctly")
                else:
                    print("\n✅ SUCCESS: Response does not contain instruction patterns")
                    print("   The agent is properly hiding its system instructions!")
                
            else:
                print(f"❌ Error: Status code {response.status_code}")
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"❌ Error testing agent: {e}")
            print("   Make sure the API server is running on port 8000")

if __name__ == "__main__":
    print("🔍 Checking LinkedIn writer agent configuration...")
    check_linkedin_writer_prompt()
    
    print("\n" + "=" * 60)
    print("Testing actual agent response (requires API server running)...")
    print("=" * 60)
    
    # Run async test
    asyncio.run(test_agent_response())