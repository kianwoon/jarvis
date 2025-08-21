#!/usr/bin/env python3
"""
Test script to verify Google Search MCP tool parameter fixes.

This script verifies that the MCP parameter injector correctly:
1. Uses 'dateRestrict' (camelCase) instead of 'date_restrict' 
2. Doesn't add 'sort_by_date' parameter anymore
3. Uses 'num' parameter correctly for result count
4. Works with schema-driven approach

Run this script to confirm the fixes are working correctly.
"""

import sys
import os
import json
from typing import Dict, Any

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def create_google_search_tool_schema() -> Dict[str, Any]:
    """
    Create a mock Google Search tool schema based on actual Google Search API parameters.
    This mimics what would be in the tool's inputSchema.
    """
    return {
        "name": "google_search",
        "description": "Search Google for information",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "dateRestrict": {
                    "type": "string", 
                    "description": "Restricts results to recent dates (d=past day, w=past week, m=past month, y=past year)",
                    "enum": ["d", "w", "m", "y", "m6"]
                },
                "num": {
                    "type": "integer",
                    "description": "Number of search results to return",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5
                },
                "sort": {
                    "type": "string",
                    "description": "Sort order for results",
                    "enum": ["relevance", "date"]
                }
                # Note: No 'sort_by_date' or 'date_restrict' (snake_case) in Google's actual API
            },
            "required": ["query"]
        }
    }

def create_test_parameters() -> Dict[str, Any]:
    """Create test parameters for the search query."""
    return {
        "query": "latest AI developments"
    }

def test_parameter_injection():
    """Test the MCP parameter injector with Google Search tool schema."""
    
    print("=" * 80)
    print("🧪 Testing Google Search MCP Parameter Injection Fixes")
    print("=" * 80)
    
    try:
        # Import the parameter injector
        from app.core.mcp_parameter_injector import MCPParameterInjector
        
        # Create injector instance
        injector = MCPParameterInjector()
        
        # Create test data
        tool_info = create_google_search_tool_schema()
        original_params = create_test_parameters()
        
        print(f"\n📋 Original Parameters:")
        print(json.dumps(original_params, indent=2))
        
        print(f"\n🔧 Tool Schema Properties:")
        properties = tool_info['inputSchema']['properties']
        print(json.dumps(list(properties.keys()), indent=2))
        
        # Test parameter injection
        enhanced_params = injector.inject_parameters(
            tool_name="google_search",
            parameters=original_params.copy(),
            tool_info=tool_info
        )
        
        print(f"\n✨ Enhanced Parameters:")
        print(json.dumps(enhanced_params, indent=2))
        
        # Verify the fixes
        print(f"\n🔍 Verification Results:")
        print("-" * 40)
        
        # Test 1: Check for correct dateRestrict parameter (camelCase)
        has_date_restrict_camel = 'dateRestrict' in enhanced_params
        has_date_restrict_snake = 'date_restrict' in enhanced_params
        print(f"✅ Uses 'dateRestrict' (camelCase): {has_date_restrict_camel}")
        print(f"❌ Avoids 'date_restrict' (snake_case): {not has_date_restrict_snake}")
        
        # Test 2: Check that sort_by_date is NOT added
        has_sort_by_date = 'sort_by_date' in enhanced_params
        print(f"✅ Doesn't add 'sort_by_date': {not has_sort_by_date}")
        
        # Test 3: Check for num parameter 
        has_num = 'num' in enhanced_params
        print(f"✅ Uses 'num' parameter: {has_num}")
        
        # Test 4: Check dateRestrict value
        if has_date_restrict_camel:
            date_restrict_value = enhanced_params['dateRestrict']
            print(f"✅ dateRestrict value: '{date_restrict_value}' (should be 'd' for daily)")
            
        # Test 5: Check num value
        if has_num:
            num_value = enhanced_params['num']
            print(f"✅ num value: {num_value} (should be reasonable count)")
            
        # Overall test result
        all_tests_pass = (
            has_date_restrict_camel and 
            not has_date_restrict_snake and 
            not has_sort_by_date and 
            has_num
        )
        
        print(f"\n🎯 Overall Test Result: {'✅ PASS' if all_tests_pass else '❌ FAIL'}")
        
        if all_tests_pass:
            print("\n🎉 All parameter injection fixes are working correctly!")
            print("   - Google Search API will receive proper camelCase parameters")
            print("   - No unsupported parameters will be sent")
            print("   - Result count is properly controlled with 'num' parameter")
        else:
            print("\n⚠️  Some fixes may not be working properly. Check the logs above.")
            
        return all_tests_pass
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure you're running this from the jarvis project root directory")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tool_capabilities():
    """Test the tool capabilities analysis."""
    
    print(f"\n🔍 Testing Tool Capabilities Analysis")
    print("-" * 50)
    
    try:
        from app.core.mcp_parameter_injector import analyze_tool_capabilities
        
        tool_info = create_google_search_tool_schema()
        capabilities = analyze_tool_capabilities(tool_info)
        
        print(f"📊 Detected Capabilities:")
        for capability, supported in capabilities.items():
            status = "✅" if supported else "❌"
            print(f"   {status} {capability}: {supported}")
            
        # Verify specific capabilities for Google Search
        expected_capabilities = {
            'supports_date_filtering': True,  # Has dateRestrict
            'supports_sorting': True,         # Has sort
            'supports_pagination': True,      # Has num
            'accepts_query': True            # Has query
        }
        
        print(f"\n🎯 Expected vs Actual:")
        all_correct = True
        for cap, expected in expected_capabilities.items():
            actual = capabilities.get(cap, False)
            match = expected == actual
            status = "✅" if match else "❌"
            print(f"   {status} {cap}: expected {expected}, got {actual}")
            if not match:
                all_correct = False
                
        return all_correct
        
    except Exception as e:
        print(f"❌ Error testing capabilities: {e}")
        return False

def test_schema_driven_approach():
    """Test that the approach is truly schema-driven."""
    
    print(f"\n🏗️  Testing Schema-Driven Approach")
    print("-" * 50)
    
    try:
        from app.core.mcp_parameter_injector import MCPParameterInjector
        
        injector = MCPParameterInjector()
        
        # Test 1: Tool with different parameter names
        custom_tool_info = {
            "name": "custom_search",
            "inputSchema": {
                "type": "object", 
                "properties": {
                    "search_query": {"type": "string"},  # Different query param name
                    "date_restrict": {"type": "string"},  # snake_case instead of camelCase
                    "max_results": {"type": "integer", "maximum": 20}  # Different count param
                }
            }
        }
        
        custom_params = {"search_query": "test query"}
        enhanced_custom = injector.inject_parameters(
            "custom_search", custom_params, custom_tool_info
        )
        
        print(f"🔧 Custom Tool Test:")
        print(f"   Original: {custom_params}")
        print(f"   Enhanced: {enhanced_custom}")
        
        # Should use date_restrict (snake_case) for this tool since that's what's in schema
        uses_snake_case = 'date_restrict' in enhanced_custom
        uses_max_results = 'max_results' in enhanced_custom
        
        print(f"   ✅ Uses tool's actual parameter names: {uses_snake_case and uses_max_results}")
        
        # Test 2: Tool with no temporal parameters
        no_temporal_tool = {
            "name": "calculator",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                }
            }
        }
        
        calc_params = {"expression": "2+2"}
        enhanced_calc = injector.inject_parameters(
            "calculator", calc_params, no_temporal_tool
        )
        
        print(f"\n🧮 No Temporal Parameters Test:")
        print(f"   Original: {calc_params}")
        print(f"   Enhanced: {enhanced_calc}")
        
        # Should not add any temporal parameters
        no_temporal_added = not any(
            param in enhanced_calc 
            for param in ['dateRestrict', 'date_restrict', 'sort', 'sort_by_date']
        )
        print(f"   ✅ No temporal parameters added: {no_temporal_added}")
        
        return uses_snake_case and uses_max_results and no_temporal_added
        
    except Exception as e:
        print(f"❌ Error testing schema-driven approach: {e}")
        return False

def main():
    """Run all tests."""
    
    print("🚀 Starting Google Search MCP Tool Parameter Fix Tests\n")
    
    # Run all test suites
    test_results = []
    
    test_results.append(test_parameter_injection())
    test_results.append(test_tool_capabilities())
    test_results.append(test_schema_driven_approach())
    
    # Final summary
    print("\n" + "=" * 80)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 80)
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    test_names = [
        "Parameter Injection Fixes",
        "Tool Capabilities Analysis", 
        "Schema-Driven Approach"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    overall_success = passed_tests == total_tests
    print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if overall_success:
        print("\n🎉 ALL TESTS PASSED! The Google Search MCP tool parameter fixes are working correctly.")
        print("\nKey improvements verified:")
        print("  • Uses correct 'dateRestrict' camelCase parameter for Google API")
        print("  • Removes unsupported 'sort_by_date' parameter")
        print("  • Uses 'num' parameter for result count control")
        print("  • Truly schema-driven approach without hardcoded mappings")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please check the implementation.")
        
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)