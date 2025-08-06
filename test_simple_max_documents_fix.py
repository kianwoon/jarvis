#!/usr/bin/env python3
"""
Simple test to verify the RAG tool configuration was updated correctly.
"""

import json
import logging
from app.core.mcp_tools_cache import get_enabled_mcp_tools
from app.core.rag_tool_config import get_rag_tool_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_tool_config():
    """Test that the RAG tool configuration was updated"""
    
    print("🧪 Testing RAG tool configuration update...")
    print("=" * 60)
    
    # Check the RAG tool config directly
    rag_config = get_rag_tool_config()
    
    print(f"✅ RAG tool name: {rag_config.get_tool_name()}")
    print(f"✅ RAG tool description: {rag_config.get_description()}")
    
    # Get the tool definition
    tool_def = rag_config.get_tool_definition("Test collections description")
    
    max_docs_param = tool_def.get("parameters", {}).get("properties", {}).get("max_documents", {})
    max_docs_desc = max_docs_param.get("description", "Not found")
    
    print(f"📋 max_documents parameter description: {max_docs_desc}")
    
    # Check if the description was updated
    if "defaults to 8" in max_docs_desc and "Only specify if you need a different value" in max_docs_desc:
        print("✅ RAG tool configuration updated correctly")
        print("🎯 The tool now specifies default value of 8 and discourages unnecessary usage")
        return True
    else:
        print("❌ RAG tool configuration was not updated correctly")
        return False

def test_mcp_tools_cache():
    """Test that the MCP tools cache has the updated tool"""
    
    print("\n🧪 Testing MCP tools cache update...")
    print("=" * 60)
    
    # Get tools from cache
    tools = get_enabled_mcp_tools()
    
    if 'rag_knowledge_search' not in tools:
        print("❌ rag_knowledge_search tool not found in cache")
        return False
    
    rag_tool = tools['rag_knowledge_search']
    rag_params = rag_tool.get('parameters', {})
    max_docs_param = rag_params.get('properties', {}).get('max_documents', {})
    max_docs_desc = max_docs_param.get('description', 'Not found')
    
    print(f"📋 Cached tool max_documents description: {max_docs_desc}")
    
    # Check if the cached version has the update
    if "defaults to 8" in max_docs_desc and "Only specify if you need a different value" in max_docs_desc:
        print("✅ MCP tools cache updated correctly")
        print("🎯 Agents will now see the corrected tool description")
        return True
    else:
        print("❌ MCP tools cache was not updated correctly")
        print("🔧 You may need to restart the application or clear the cache again")
        return False

def main():
    """Main test function"""
    print("🚀 Starting RAG tool configuration verification")
    print("=" * 80)
    
    config_ok = test_rag_tool_config()
    cache_ok = test_mcp_tools_cache()
    
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY:")
    print(f"   RAG Tool Config: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"   MCP Tools Cache: {'✅ PASS' if cache_ok else '❌ FAIL'}")
    
    if config_ok and cache_ok:
        print("\n🎉 SUCCESS: All configurations updated correctly!")
        print("📝 The agent should no longer hardcode max_documents: 5")
        print("📈 It will either omit the parameter (using default 8) or specify a different value if needed")
    else:
        print("\n❌ FAILURE: Some configurations need additional fixes")
    
    return config_ok and cache_ok

if __name__ == "__main__":
    result = main()
    exit(0 if result else 1)