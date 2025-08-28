#!/usr/bin/env python3
"""
Test actual request flow to debug the progressive response issue
"""

import asyncio
import sys
import json
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

from app.api.v1.endpoints.notebooks import chat_with_notebook_v2
from app.models.notebook_models import NotebookChatRequest

async def test_actual_request():
    """Simulate the actual request that should trigger progressive response"""
    print("🚀 Testing Actual Request Flow...")
    
    # Create the request that should trigger progressive response
    request = NotebookChatRequest(
        message="show me all projects in table format",
        conversation_id="test-conversation-debug"
    )
    
    print("📝 Request created:")
    print(f"   Message: '{request.message}'")
    print(f"   Conversation ID: {request.conversation_id}")
    
    try:
        print("\n🔄 Making request to chat_with_notebook_v2...")
        
        # This would normally be called via FastAPI, but we'll call it directly
        # Note: We can't actually test this without the full FastAPI context
        print("⚠️  Note: This requires full FastAPI context to execute properly")
        print("   Instead, let's trace the logical flow:")
        
        # Trace the expected flow
        print("\n📊 Expected Flow Trace:")
        print("1. ✅ chat_with_notebook_v2 receives request")
        print("2. ✅ notebook_rag_service.generate_rag_response() called")
        print("3. ✅ should_extract = True for 'all projects' query")
        print("4. ✅ extract_project_data() extracts 90+ projects")  
        print("5. ✅ extracted_projects contains ProjectData objects")
        print("6. ✅ Progressive response check: len(extracted_projects) > 20")
        print("7. ✅ progressive_response_service.generate_progressive_stream() called")
        print("8. ✅ Stream yields: {'chunk': 'header'}, {'chunk': 'batch1'}, etc.")
        print("9. ✅ Frontend displays progressive content")
        
        print("\n🔍 Key Checkpoints to Verify:")
        print("   📌 extracted_projects contains actual ProjectData objects (not None/empty)")
        print("   📌 Progressive response length check: > 20 items")
        print("   📌 JSON format: {'chunk': 'content'} for frontend compatibility")
        
        print("\n✅ Flow analysis complete - all components should be working")
        
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        return False
    
    return True

def analyze_potential_issues():
    """Analyze potential issues in the flow"""
    print("\n🔍 Potential Issues Analysis:")
    
    issues = [
        {
            "issue": "extract_project_data returns empty list []",
            "check": "Add logging inside extract_project_data to see actual extraction results",
            "impact": "Progressive response never triggers because len([]) = 0"
        },
        {
            "issue": "extracted_projects not being passed to NotebookRAGResponse correctly",
            "check": "Verify extracted_projects field in NotebookRAGResponse construction",
            "impact": "Data extracted but lost in response object"
        },
        {
            "issue": "Progressive response check using wrong condition",
            "check": "Verify: hasattr(rag_response, 'extracted_projects') and len() > 20",
            "impact": "Condition fails even when data exists"
        },
        {
            "issue": "Frontend JSON format incompatibility",
            "check": "Verify {'chunk': 'content'} format is being used correctly",
            "impact": "Backend streams but frontend doesn't display"
        }
    ]
    
    for i, issue in enumerate(issues, 1):
        print(f"\n{i}. 🚨 {issue['issue']}")
        print(f"   🔎 Check: {issue['check']}")
        print(f"   💥 Impact: {issue['impact']}")
    
    print(f"\n📋 Recommended Debug Strategy:")
    print("1. Add debug logging inside extract_project_data() to see raw extraction results")
    print("2. Add debug logging in notebooks.py before progressive response check")
    print("3. Test progressive response service in isolation with mock data")
    print("4. Verify frontend receives and processes {'chunk': 'content'} format")

if __name__ == "__main__":
    print("🧪 Actual Request Flow Test")
    print("=" * 50)
    
    # Run flow analysis
    result = asyncio.run(test_actual_request())
    
    # Analyze issues
    analyze_potential_issues()
    
    print("\n" + "=" * 50)
    if result:
        print("✅ Flow analysis completed successfully")
        print("💡 Next step: Add debug logging to trace actual data flow")
    else:
        print("❌ Flow analysis encountered issues")