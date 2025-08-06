#!/usr/bin/env python3
"""
Test script to verify workflow agent makes only 1 LLM call and provides detailed responses
Compares workflow mode vs standard chat mode responses
"""

import asyncio
import json
import time
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_workflow_single_call():
    """Test that workflow makes only 1 LLM call and provides detailed responses"""
    
    print("\n" + "="*80)
    print("TESTING WORKFLOW SINGLE CALL FIX")
    print("="*80)
    
    # Test query that requires RAG search
    test_query = "What is the partnership between DBS and AWS about?"
    
    print(f"\n📝 Test Query: {test_query}")
    print("-"*80)
    
    # Initialize workflow executor
    from app.automation.core.agent_workflow_executor import AgentWorkflowExecutor
    executor = AgentWorkflowExecutor()
    
    # Create a simple workflow configuration with one agent (using correct structure)
    workflow_config = {
        "workflow_id": 14,
        "execution_pattern": "sequential",
        "nodes": [
            {
                "id": "agent_1",
                "type": "agentnode",  # Correct type for agent nodes
                "data": {
                    "type": "AgentNode",  # Node data type
                    "node": {
                        "agent_name": "Research Analyst",  # This is the key field
                        "name": "Research Analyst",
                        "role": "Senior Research Analyst",
                        "description": "Expert at analyzing documents and extracting insights",
                        "system_prompt": "You are a senior research analyst. Provide detailed, comprehensive analysis with specific metrics and data points.",
                        "tools": ["rag_knowledge_search"],
                        "config": {
                            "temperature": 0.7,
                            "max_tokens": 4000,
                            "timeout": 60,
                            "model": "qwen3:30b-a3b"
                        }
                    }
                }
            }
        ],
        "edges": [],
        "cache_nodes": [],
        "transform_nodes": [],
        "parallel_nodes": [],
        "condition_nodes": [],
        "state_nodes": [],
        "api_nodes": []
    }
    
    print("\n🚀 Executing workflow with single agent...")
    print("-"*40)
    
    # Track LLM calls
    llm_call_count = 0
    token_counts = []
    response_lengths = []
    start_time = time.time()
    
    # Execute workflow
    workflow_response = ""
    tool_results = []
    
    async for update in executor.execute_agent_workflow(
        workflow_id=14,
        execution_id=f"test_{int(time.time())}",
        workflow_config=workflow_config,
        message=test_query
    ):
        update_type = update.get("type", "")
        
        if update_type == "agent_token":
            # Count tokens being streamed
            pass
        elif update_type == "agent_complete":
            llm_call_count += 1
            content = update.get("content", "")
            workflow_response = content
            
            # Extract metrics from logs
            print(f"\n✅ Agent completed LLM call #{llm_call_count}")
            print(f"   Response length: {len(content)} chars")
            
            # Check for tool usage
            if "tools_used" in update:
                tool_results = update["tools_used"]
                print(f"   Tools used: {len(tool_results)}")
        elif update_type == "workflow_complete":
            print(f"\n✅ Workflow completed!")
    
    execution_time = time.time() - start_time
    
    # Analyze the response
    print("\n" + "="*80)
    print("📊 ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\n1️⃣ LLM Call Count: {llm_call_count}")
    if llm_call_count == 1:
        print("   ✅ SUCCESS: Only 1 LLM call made (matching standard chat)")
    else:
        print(f"   ❌ FAILED: Made {llm_call_count} LLM calls (should be 1)")
    
    print(f"\n2️⃣ Response Length: {len(workflow_response)} chars")
    if len(workflow_response) > 1500:
        print("   ✅ Comprehensive response generated")
    else:
        print("   ⚠️ Response might be too brief")
    
    print(f"\n3️⃣ Execution Time: {execution_time:.2f} seconds")
    
    # Check for specific content quality indicators
    print("\n4️⃣ Response Quality Check:")
    
    quality_checks = {
        "Has specific years/dates": any(year in workflow_response for year in ["2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]),
        "Has specific numbers": any(char.isdigit() for char in workflow_response),
        "Has company names": "AWS" in workflow_response or "Amazon" in workflow_response or "DBS" in workflow_response,
        "Has document references": "document" in workflow_response.lower() or "found" in workflow_response.lower(),
        "Has detailed analysis": len(workflow_response) > 2000,
        "Has structured sections": "**" in workflow_response or "##" in workflow_response or "•" in workflow_response
    }
    
    for check, passed in quality_checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")
    
    # Show response preview
    print("\n5️⃣ Response Preview (first 500 chars):")
    print("-"*40)
    print(workflow_response[:500])
    if len(workflow_response) > 500:
        print("...")
    
    # Check for specific metrics that should be in the response
    print("\n6️⃣ Specific Metrics Check:")
    expected_metrics = [
        "16", "years", "partnership",
        "1,800", "engineers", 
        "4,600", "staff",
        "2,800", "engineers",
        "Alibaba", "Taobao", "Tmall", "Ant"
    ]
    
    found_metrics = []
    missing_metrics = []
    
    for metric in expected_metrics:
        if metric.lower() in workflow_response.lower():
            found_metrics.append(metric)
        else:
            missing_metrics.append(metric)
    
    if found_metrics:
        print(f"   ✅ Found metrics: {', '.join(found_metrics)}")
    if missing_metrics:
        print(f"   ⚠️ Missing metrics: {', '.join(missing_metrics)}")
    
    # Overall assessment
    print("\n" + "="*80)
    print("🎯 OVERALL ASSESSMENT")
    print("="*80)
    
    success_criteria = [
        llm_call_count == 1,
        len(workflow_response) > 1500,
        quality_checks["Has specific numbers"],
        quality_checks["Has company names"],
        len(found_metrics) > len(missing_metrics)
    ]
    
    success_rate = sum(success_criteria) / len(success_criteria) * 100
    
    if success_rate >= 80:
        print(f"✅ SUCCESS: Workflow fix is working! ({success_rate:.0f}% criteria met)")
        print("   - Eliminated redundant LLM call")
        print("   - Generating comprehensive responses")
        print("   - Including specific metrics and details")
    elif success_rate >= 60:
        print(f"⚠️ PARTIAL SUCCESS: Some improvements needed ({success_rate:.0f}% criteria met)")
    else:
        print(f"❌ NEEDS WORK: Fix not fully effective ({success_rate:.0f}% criteria met)")
    
    return {
        "llm_calls": llm_call_count,
        "response_length": len(workflow_response),
        "execution_time": execution_time,
        "quality_score": success_rate
    }

if __name__ == "__main__":
    print("🔧 Starting Workflow Single Call Fix Test...")
    print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        result = asyncio.run(test_workflow_single_call())
        print("\n✅ Test completed successfully!")
        
        # Exit with appropriate code
        if result["llm_calls"] == 1 and result["quality_score"] >= 80:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Needs improvement
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)