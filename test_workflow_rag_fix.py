#!/usr/bin/env python3
"""
Test Workflow RAG Response Fix
==============================

Comprehensive test to verify that the workflow agent correctly interprets
RAG responses after our fixes.
"""

import sys
import os
import json

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from app.automation.integrations.mcp_bridge import MCPToolsBridge
from app.automation.core.workflow_prompt_generator import WorkflowPromptGenerator

def test_mcp_bridge_response():
    """Test that MCP bridge returns enhanced response format"""
    print("=" * 60)
    print("1. Testing MCP Bridge Enhanced Response Format")
    print("=" * 60)
    
    bridge = MCPToolsBridge()
    
    # Test RAG tool execution
    parameters = {
        "query": "beyondsoft alibaba partnership",
        "max_documents": 5,
        "include_content": True
    }
    
    result = bridge.execute_tool_sync("rag_knowledge_search", parameters)
    
    print(f"\n✅ Tool executed: {result.get('success')}")
    
    if result.get("success"):
        tool_result = result.get("result", {})
        
        # Check for enhanced fields
        print(f"\n📊 Enhanced Response Fields:")
        print(f"   - has_results: {tool_result.get('has_results')}")
        print(f"   - documents_found: {tool_result.get('documents_found')}")
        print(f"   - text_summary present: {'text_summary' in tool_result}")
        
        if 'text_summary' in tool_result:
            print(f"\n📝 Text Summary Preview:")
            print(tool_result['text_summary'][:500])
        
        return tool_result
    else:
        print(f"❌ Tool execution failed: {result.get('error')}")
        return None

def test_workflow_prompt():
    """Test that workflow prompt includes RAG interpretation instructions"""
    print("\n" + "=" * 60)
    print("2. Testing Workflow Prompt RAG Instructions")
    print("=" * 60)
    
    generator = WorkflowPromptGenerator()
    
    # Generate tools section with RAG tool
    tools = ["rag_knowledge_search", "get_datetime"]
    tools_section = generator.generate_tools_section(tools)
    
    print("\n📝 Generated Tools Section:")
    print("-" * 40)
    
    # Check for RAG interpretation instructions
    if "RAG Knowledge Search Tool Response Interpretation" in tools_section:
        print("✅ RAG interpretation instructions FOUND in prompt")
        
        # Extract and display the RAG instructions
        lines = tools_section.split('\n')
        in_rag_section = False
        rag_instructions = []
        
        for line in lines:
            if "RAG Knowledge Search Tool Response Interpretation" in line:
                in_rag_section = True
            elif in_rag_section:
                if line.startswith("###") and "RAG" not in line:
                    break
                rag_instructions.append(line)
        
        print("\n📋 RAG Instructions:")
        for line in rag_instructions[:10]:  # Show first 10 lines
            print(line)
    else:
        print("❌ RAG interpretation instructions NOT FOUND in prompt")
    
    return tools_section

def simulate_agent_interpretation(rag_response):
    """Simulate how an agent would interpret the enhanced response"""
    print("\n" + "=" * 60)
    print("3. Simulating Agent Interpretation")
    print("=" * 60)
    
    if not rag_response:
        print("❌ No response to interpret")
        return False
    
    print("\n🤖 Agent Interpretation Logic:")
    
    # Check 1: has_results flag
    has_results = rag_response.get('has_results', False)
    print(f"   1. Check has_results flag: {has_results}")
    
    # Check 2: documents_found > 0
    docs_found = rag_response.get('documents_found', 0)
    print(f"   2. Check documents_found > 0: {docs_found} > 0 = {docs_found > 0}")
    
    # Check 3: text_summary contains SUCCESS
    text_summary = rag_response.get('text_summary', '')
    has_success = '✅ SUCCESS' in text_summary
    print(f"   3. Check text_summary contains SUCCESS: {has_success}")
    
    # Check 4: documents array not empty
    documents = rag_response.get('documents', [])
    has_docs = len(documents) > 0
    print(f"   4. Check documents array not empty: {has_docs}")
    
    # Overall interpretation
    correct_interpretation = has_results and docs_found > 0 and has_docs
    
    print(f"\n📊 Final Interpretation:")
    if correct_interpretation:
        print(f"   ✅ CORRECT: Agent should report {docs_found} documents found")
        print(f"   Agent should summarize the partnership information")
    else:
        print(f"   ❌ INCORRECT: Agent might still claim 'zero results'")
    
    return correct_interpretation

def main():
    """Run all tests"""
    print("🔧 WORKFLOW RAG RESPONSE FIX VERIFICATION")
    print("=" * 60)
    print("Testing the complete fix for workflow RAG response interpretation\n")
    
    # Test 1: MCP Bridge Response
    rag_response = test_mcp_bridge_response()
    
    # Test 2: Workflow Prompt
    prompt = test_workflow_prompt()
    
    # Test 3: Simulate Agent Interpretation
    if rag_response:
        interpretation_correct = simulate_agent_interpretation(rag_response)
    else:
        interpretation_correct = False
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 3
    
    if rag_response and rag_response.get('has_results'):
        print("✅ Test 1: MCP Bridge returns enhanced format - PASSED")
        tests_passed += 1
    else:
        print("❌ Test 1: MCP Bridge returns enhanced format - FAILED")
    
    if prompt and "RAG Knowledge Search Tool Response Interpretation" in prompt:
        print("✅ Test 2: Workflow prompt includes RAG instructions - PASSED")
        tests_passed += 1
    else:
        print("❌ Test 2: Workflow prompt includes RAG instructions - FAILED")
    
    if interpretation_correct:
        print("✅ Test 3: Agent interpretation logic correct - PASSED")
        tests_passed += 1
    else:
        print("❌ Test 3: Agent interpretation logic correct - FAILED")
    
    print(f"\n📊 Overall: {tests_passed}/{tests_total} tests passed")
    
    if tests_passed == tests_total:
        print("\n🎉 SUCCESS: All fixes verified! The workflow should now correctly")
        print("   interpret RAG responses and return partnership information.")
    else:
        print("\n⚠️  WARNING: Some tests failed. The workflow may still have issues.")
    
    # Save test results
    test_results = {
        "mcp_response": rag_response,
        "has_rag_instructions": "RAG Knowledge Search Tool Response Interpretation" in prompt if prompt else False,
        "interpretation_correct": interpretation_correct,
        "tests_passed": tests_passed,
        "tests_total": tests_total
    }
    
    with open("workflow_rag_fix_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    print("\n💾 Test results saved to workflow_rag_fix_results.json")

if __name__ == "__main__":
    main()