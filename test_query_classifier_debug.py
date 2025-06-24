#!/usr/bin/env python3
"""
Test script to debug the EnhancedQueryClassifier
"""
import asyncio
import sys
import os
import json
import logging

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_query_classifier():
    """Test the EnhancedQueryClassifier with the specific query"""
    
    print("=" * 80)
    print("TESTING ENHANCED QUERY CLASSIFIER")
    print("=" * 80)
    
    # Test query
    test_query = "what is today date & time ?"
    print(f"Test query: '{test_query}'")
    print()
    
    try:
        # First, let's check what MCP tools are available
        print("1. Checking MCP tools cache...")
        try:
            from app.core.mcp_tools_cache import get_enabled_mcp_tools
            mcp_tools = get_enabled_mcp_tools()
            print(f"   Available MCP tools: {len(mcp_tools)} tools")
            if mcp_tools:
                for tool_name, tool_info in mcp_tools.items():
                    print(f"   - {tool_name}: {tool_info.get('description', 'No description')}")
            else:
                print("   NO MCP TOOLS FOUND!")
            print()
        except Exception as e:
            print(f"   ERROR loading MCP tools: {e}")
            print()
        
        # Check query classifier settings
        print("2. Checking query classifier settings...")
        threshold = 0.6  # Default threshold
        try:
            from app.core.query_classifier_settings_cache import get_query_classifier_settings
            settings = get_query_classifier_settings()
            print(f"   Query classifier settings: {json.dumps(settings, indent=2)}")
            threshold = settings.get("direct_execution_threshold", 0.6)
            print(f"   Direct execution threshold: {threshold}")
            print()
        except Exception as e:
            print(f"   ERROR loading query classifier settings: {e}")
            print(f"   Using default threshold: {threshold}")
            print()
        
        # Initialize the classifier
        print("3. Initializing EnhancedQueryClassifier...")
        from app.langchain.enhanced_query_classifier import EnhancedQueryClassifier
        classifier = EnhancedQueryClassifier()
        print(f"   Classifier initialized successfully")
        print(f"   Available MCP tools in classifier: {len(classifier.mcp_tool_names)} tools")
        print(f"   MCP tool names: {list(classifier.mcp_tool_names)}")
        print(f"   Available RAG collections: {len(classifier.rag_collections)} collections")
        print(f"   RAG collection names: {list(classifier.rag_collections.keys())}")
        print()
        
        # Test classification
        print("4. Running classification...")
        results = await classifier.classify(test_query)
        print(f"   Classification results: {len(results)} results")
        print()
        
        # Analyze results in detail
        print("5. Detailed results analysis:")
        for i, result in enumerate(results):
            print(f"   Result {i+1}:")
            print(f"     - Query type: {result.query_type.value}")
            print(f"     - Confidence: {result.confidence:.3f}")
            print(f"     - Suggested tools: {result.suggested_tools}")
            print(f"     - Suggested agents: {result.suggested_agents}")
            print(f"     - Matched patterns: {result.matched_patterns}")
            print(f"     - Metadata: {json.dumps(result.metadata, indent=6)}")
            print()
        
        # Check the specific condition from service.py
        print("6. Checking direct bypass condition:")
        print(f"   Condition: results and results[0].confidence >= threshold and results[0].suggested_tools")
        
        if results:
            first_result = results[0]
            confidence_check = first_result.confidence >= threshold
            tools_check = bool(first_result.suggested_tools)
            
            print(f"   - results exists: True")
            print(f"   - results[0].confidence: {first_result.confidence:.3f}")
            print(f"   - threshold: {threshold}")
            print(f"   - confidence >= threshold: {confidence_check}")
            print(f"   - results[0].suggested_tools: {first_result.suggested_tools}")
            print(f"   - suggested_tools exists: {tools_check}")
            print(f"   - FINAL CONDITION: {confidence_check and tools_check}")
            
            if confidence_check and tools_check:
                print(f"   ✅ DIRECT BYPASS SHOULD TRIGGER!")
                print(f"   Tool to execute: {first_result.suggested_tools[0]}")
            else:
                print(f"   ❌ DIRECT BYPASS WILL NOT TRIGGER")
                if not confidence_check:
                    print(f"      - Confidence too low: {first_result.confidence:.3f} < {threshold}")
                if not tools_check:
                    print(f"      - No suggested tools")
        else:
            print(f"   - results exists: False")
            print(f"   ❌ DIRECT BYPASS WILL NOT TRIGGER - no results")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_query_classifier())