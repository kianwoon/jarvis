#!/usr/bin/env python3
"""
Test script to verify the service layer search optimization fix
"""
import sys
import os

# Add the app directory to path
sys.path.insert(0, '/Users/kianwoonwong/Downloads/jarvis')

def test_service_optimization():
    """Test the search optimization in service layer"""
    
    print("Testing Service Layer Search Optimization...")
    
    try:
        from app.langchain.service import _map_tool_parameters_service
        
        # Test cases with search tools
        test_cases = [
            ("google_search", {"query": "what are the best AI models for coding?"}),
            ("web_search", {"query": "latest Python frameworks 2025"}),
            ("tavily_search", {"query": "I want to know about machine learning"}),
            ("regular_tool", {"param": "value"})  # Non-search tool
        ]
        
        for tool_name, params in test_cases:
            print(f"\n--- Testing {tool_name} ---")
            print(f"Original params: {params}")
            
            try:
                # This should now work without event loop errors
                result_tool, result_params = _map_tool_parameters_service(tool_name, params)
                print(f"Result tool: {result_tool}")
                print(f"Result params: {result_params}")
                
                if 'query' in params and 'query' in result_params:
                    if params['query'] != result_params['query']:
                        print("✅ Query was optimized!")
                    else:
                        print("⚠️  Query unchanged (optimization disabled or failed)")
                else:
                    print("ℹ️  Non-search tool processed normally")
                    
            except Exception as e:
                print(f"❌ Error processing {tool_name}: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_service_optimization()