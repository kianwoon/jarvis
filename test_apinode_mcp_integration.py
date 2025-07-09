#!/usr/bin/env python3
"""
Test script for APINode MCP integration
Tests the complete flow of APINode to MCP tool registration and execution
"""
import asyncio
import json
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.automation.integrations.apinode_mcp_bridge import apinode_mcp_bridge

def create_sample_workflow():
    """Create a sample workflow with APINode connected to AgentNode"""
    return {
        "nodes": [
            {
                "id": "input-1",
                "type": "InputNode",
                "data": {
                    "type": "InputNode",
                    "node": {
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"}
                            }
                        }
                    }
                }
            },
            {
                "id": "api-1",
                "type": "APINode",
                "data": {
                    "type": "APINode",
                    "node": {
                        "label": "Weather API",
                        "base_url": "https://api.weather.com",
                        "endpoint_path": "/v1/current",
                        "http_method": "GET",
                        "authentication_type": "api_key",
                        "auth_header_name": "X-API-Key",
                        "auth_token": "test_key_123",
                        "request_schema": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "Location to get weather for"
                                }
                            },
                            "required": ["location"]
                        },
                        "response_schema": {
                            "type": "object",
                            "properties": {
                                "temperature": {"type": "number"},
                                "condition": {"type": "string"}
                            }
                        },
                        "enable_mcp_tool": True,
                        "tool_description": "Get current weather information for a location"
                    }
                }
            },
            {
                "id": "agent-1",
                "type": "AgentNode",
                "data": {
                    "type": "AgentNode",
                    "node": {
                        "agent_name": "weather_assistant",
                        "label": "Weather Assistant",
                        "custom_prompt": "You are a weather assistant that can help users get weather information."
                    }
                }
            },
            {
                "id": "output-1",
                "type": "OutputNode",
                "data": {
                    "type": "OutputNode",
                    "node": {
                        "output_format": "text"
                    }
                }
            }
        ],
        "edges": [
            {
                "id": "edge-1",
                "source": "input-1",
                "target": "agent-1",
                "sourceHandle": "data",
                "targetHandle": "input"
            },
            {
                "id": "edge-2",
                "source": "api-1",
                "target": "agent-1",
                "sourceHandle": "response",
                "targetHandle": "context"
            },
            {
                "id": "edge-3",
                "source": "agent-1",
                "target": "output-1",
                "sourceHandle": "output",
                "targetHandle": "result"
            }
        ]
    }

async def test_apinode_discovery():
    """Test APINode discovery and tool registration"""
    print("=== Testing APINode Discovery ===")
    
    # Create sample workflow
    workflow_config = create_sample_workflow()
    workflow_id = "test_workflow_123"
    
    # Test tool discovery
    tools = apinode_mcp_bridge.discover_apinode_tools(workflow_config, workflow_id)
    
    print(f"Discovered {len(tools)} tools:")
    for tool_name, tool_info in tools.items():
        print(f"  - {tool_name}: {tool_info.get('description', 'No description')}")
        print(f"    Parameters: {list(tool_info.get('parameters', {}).get('properties', {}).keys())}")
        print(f"    Required: {tool_info.get('parameters', {}).get('required', [])}")
    
    return tools

async def test_tool_registration():
    """Test tool registration and unregistration"""
    print("\n=== Testing Tool Registration ===")
    
    # Create sample workflow
    workflow_config = create_sample_workflow()
    workflow_id = "test_workflow_456"
    
    # Register tools
    registered_count = apinode_mcp_bridge.register_workflow_tools(workflow_id, workflow_config)
    print(f"Registered {registered_count} tools")
    
    # Check registered tools
    workflow_tools = apinode_mcp_bridge.get_workflow_tools(workflow_id)
    print(f"Workflow tools: {list(workflow_tools.keys())}")
    
    all_tools = apinode_mcp_bridge.get_all_registered_tools()
    print(f"All registered tools: {list(all_tools)}")
    
    # Unregister tools
    unregistered_count = apinode_mcp_bridge.unregister_workflow_tools(workflow_id)
    print(f"Unregistered {unregistered_count} tools")
    
    # Verify cleanup
    workflow_tools_after = apinode_mcp_bridge.get_workflow_tools(workflow_id)
    print(f"Workflow tools after cleanup: {list(workflow_tools_after.keys())}")

async def test_tool_execution():
    """Test APINode tool execution"""
    print("\n=== Testing Tool Execution ===")
    
    # Create sample workflow
    workflow_config = create_sample_workflow()
    workflow_id = "test_workflow_789"
    
    # Register tools
    registered_count = apinode_mcp_bridge.register_workflow_tools(workflow_id, workflow_config)
    print(f"Registered {registered_count} tools for execution test")
    
    # Get the tool name
    workflow_tools = apinode_mcp_bridge.get_workflow_tools(workflow_id)
    if not workflow_tools:
        print("No tools found for execution test")
        return
    
    tool_name = list(workflow_tools.keys())[0]
    print(f"Testing execution of tool: {tool_name}")
    
    # Test parameters
    parameters = {
        "location": "New York"
    }
    
    try:
        # This will fail because it's trying to make a real API call
        # but we can test the validation and setup
        result = await apinode_mcp_bridge.execute_apinode_tool(
            tool_name=tool_name,
            parameters=parameters,
            workflow_id=workflow_id,
            execution_id="test_execution_123"
        )
        print(f"Execution result: {result}")
    except Exception as e:
        print(f"Expected error (API call would fail): {str(e)}")
    
    # Clean up
    apinode_mcp_bridge.unregister_workflow_tools(workflow_id)

async def test_validation():
    """Test parameter validation"""
    print("\n=== Testing Parameter Validation ===")
    
    # Create workflow with more complex validation
    workflow_config = create_sample_workflow()
    workflow_id = "test_workflow_validation"
    
    # Modify the API node to have more validation
    api_node = workflow_config["nodes"][1]  # The API node
    api_node["data"]["node"]["request_schema"] = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Location to get weather for"
            },
            "units": {
                "type": "string",
                "description": "Temperature units (celsius/fahrenheit)"
            },
            "include_forecast": {
                "type": "boolean",
                "description": "Include forecast data"
            }
        },
        "required": ["location"]
    }
    
    # Register tools
    registered_count = apinode_mcp_bridge.register_workflow_tools(workflow_id, workflow_config)
    print(f"Registered {registered_count} tools for validation test")
    
    # Get the tool name
    workflow_tools = apinode_mcp_bridge.get_workflow_tools(workflow_id)
    tool_name = list(workflow_tools.keys())[0]
    
    # Test valid parameters
    valid_params = {
        "location": "San Francisco",
        "units": "celsius",
        "include_forecast": True
    }
    
    try:
        result = await apinode_mcp_bridge.execute_apinode_tool(
            tool_name=tool_name,
            parameters=valid_params,
            workflow_id=workflow_id,
            execution_id="test_validation_1"
        )
        print("Valid parameters: Test would execute (API call expected to fail)")
    except Exception as e:
        print(f"Valid parameters error: {str(e)}")
    
    # Test missing required parameter
    invalid_params = {
        "units": "fahrenheit"
    }
    
    try:
        result = await apinode_mcp_bridge.execute_apinode_tool(
            tool_name=tool_name,
            parameters=invalid_params,
            workflow_id=workflow_id,
            execution_id="test_validation_2"
        )
        print("Missing required parameter: Unexpected success")
    except ValueError as e:
        print(f"Missing required parameter correctly caught: {str(e)}")
    except Exception as e:
        print(f"Missing required parameter other error: {str(e)}")
    
    # Clean up
    apinode_mcp_bridge.unregister_workflow_tools(workflow_id)

async def main():
    """Run all tests"""
    print("🧪 Testing APINode MCP Integration")
    print("=" * 50)
    
    try:
        await test_apinode_discovery()
        await test_tool_registration()
        await test_tool_execution()
        await test_validation()
        
        print("\n✅ All tests completed!")
        print("Note: API execution tests are expected to fail with connection errors")
        print("since they attempt real API calls. The important part is that the")
        print("tool registration, discovery, and parameter validation work correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())