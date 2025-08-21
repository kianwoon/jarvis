#!/usr/bin/env python3
"""
Test script for Google Search MCP Tool Schema Fix Verification

This script comprehensively tests the google_search MCP tool to verify:
1. Tool configuration is correctly loaded from database/cache
2. Parameter injection is working with proper camelCase (dateRestrict, num)
3. Schema validation and parameter enhancement
4. Direct API calls to verify tool execution
5. Integration with MCP parameter injector system

Requirements:
- Backend services running (FastAPI on port 8000)
- Google Search MCP tool properly configured in database
- MCP server running on port 3001 (if using local bridge)
"""

import asyncio
import json
import logging
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_google_search_working.log')
    ]
)
logger = logging.getLogger(__name__)

class GoogleSearchMCPTester:
    """Comprehensive tester for Google Search MCP tool functionality"""
    
    def __init__(self, backend_url: str = "http://localhost:8000", mcp_bridge_url: str = "http://localhost:3001"):
        self.backend_url = backend_url.rstrip('/')
        self.mcp_bridge_url = mcp_bridge_url.rstrip('/')
        self.tool_name = "google_search"
        
    def print_separator(self, title: str, char: str = "="):
        """Print a formatted separator with title"""
        width = 80
        title_padded = f" {title} "
        padding = (width - len(title_padded)) // 2
        line = char * padding + title_padded + char * (width - padding - len(title_padded))
        print(f"\n{line}")
        
    def print_results(self, section: str, data: Any, success: bool = True):
        """Print formatted test results"""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"\n[{status}] {section}")
        if isinstance(data, dict):
            print(json.dumps(data, indent=2, default=str))
        else:
            print(str(data))
            
    async def test_backend_connectivity(self) -> bool:
        """Test if backend services are accessible"""
        self.print_separator("BACKEND CONNECTIVITY TEST")
        
        try:
            # Test main backend
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            backend_status = response.status_code == 200
            self.print_results("Backend Health Check", {
                "url": f"{self.backend_url}/health",
                "status_code": response.status_code,
                "accessible": backend_status
            }, backend_status)
            
            # Test MCP bridge (optional)
            try:
                bridge_response = requests.get(f"{self.mcp_bridge_url}/health", timeout=3)
                bridge_status = bridge_response.status_code == 200
                self.print_results("MCP Bridge Health Check", {
                    "url": f"{self.mcp_bridge_url}/health", 
                    "status_code": bridge_response.status_code,
                    "accessible": bridge_status
                }, bridge_status)
            except Exception as e:
                self.print_results("MCP Bridge Health Check", {
                    "url": f"{self.mcp_bridge_url}/health",
                    "error": str(e),
                    "accessible": False,
                    "note": "MCP bridge optional - tool may work through backend"
                }, False)
                
            return backend_status
            
        except Exception as e:
            self.print_results("Backend Connectivity", {"error": str(e)}, False)
            return False
    
    async def test_mcp_tools_cache(self) -> Optional[Dict[str, Any]]:
        """Test MCP tools cache and retrieve google_search configuration"""
        self.print_separator("MCP TOOLS CACHE TEST")
        
        try:
            # Get tools from cache
            response = requests.get(f"{self.backend_url}/api/v1/mcp-tools/from-db", timeout=10)
            
            if response.status_code != 200:
                self.print_results("MCP Tools Cache", {
                    "status_code": response.status_code,
                    "error": response.text
                }, False)
                return None
                
            data = response.json()
            tools = data.get("tools", [])
            manifests = data.get("manifests", [])
            
            # Find google_search tool
            google_search_tool = None
            for tool in tools:
                if tool.get("name") == self.tool_name:
                    google_search_tool = tool
                    break
                    
            self.print_results("MCP Tools Cache Overview", {
                "total_tools": len(tools),
                "total_manifests": len(manifests),
                "google_search_found": google_search_tool is not None
            })
            
            if google_search_tool:
                self.print_results("Google Search Tool Configuration", google_search_tool)
                return google_search_tool
            else:
                available_tools = [tool.get("name") for tool in tools]
                self.print_results("Google Search Tool Not Found", {
                    "available_tools": available_tools[:10],  # Show first 10
                    "total_available": len(available_tools)
                }, False)
                return None
                
        except Exception as e:
            self.print_results("MCP Tools Cache", {
                "error": str(e),
                "traceback": traceback.format_exc()
            }, False)
            return None
    
    async def test_parameter_injection(self, tool_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test MCP parameter injection functionality"""
        self.print_separator("PARAMETER INJECTION TEST")
        
        try:
            # Import the parameter injector
            sys.path.insert(0, '/Users/kianwoonwong/Downloads/jarvis')
            from app.core.mcp_parameter_injector import MCPParameterInjector, analyze_tool_capabilities
            
            injector = MCPParameterInjector()
            
            # Test basic query parameters
            test_params = {"query": "artificial intelligence 2025"}
            
            # Extract tool info for injection
            tool_info = {
                "inputSchema": tool_config.get("parameters", {}),
                "parameters": tool_config.get("parameters", {})
            }
            
            # Test parameter injection
            enhanced_params = injector.inject_parameters(self.tool_name, test_params, tool_info)
            
            # Analyze tool capabilities
            capabilities = analyze_tool_capabilities(tool_info)
            
            self.print_results("Original Parameters", test_params)
            self.print_results("Enhanced Parameters", enhanced_params)
            self.print_results("Tool Capabilities", capabilities)
            
            # Verify expected enhancements
            expected_enhancements = {
                "dateRestrict_added": "dateRestrict" in enhanced_params,
                "num_added": "num" in enhanced_params,
                "dateRestrict_value": enhanced_params.get("dateRestrict"),
                "num_value": enhanced_params.get("num")
            }
            
            self.print_results("Enhancement Verification", expected_enhancements)
            
            return enhanced_params
            
        except Exception as e:
            self.print_results("Parameter Injection", {
                "error": str(e),
                "traceback": traceback.format_exc()
            }, False)
            return test_params
    
    async def test_direct_mcp_execution(self, enhanced_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Test direct execution via MCP tools endpoint"""
        self.print_separator("DIRECT MCP EXECUTION TEST")
        
        try:
            # Test via backend MCP tools execute endpoint
            payload = {
                "tool_name": self.tool_name,
                "parameters": enhanced_params
            }
            
            self.print_results("Execution Request", payload)
            
            response = requests.post(
                f"{self.backend_url}/api/v1/mcp-tools/execute",
                json=payload,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            
            self.print_results("Response Status", {
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type"),
                "response_size": len(response.content)
            })
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    success = result.get("success", False)
                    
                    self.print_results("Execution Result", {
                        "success": success,
                        "tool": result.get("tool"),
                        "has_result": "result" in result and result["result"] is not None,
                        "has_error": "error" in result and result["error"] is not None
                    }, success)
                    
                    if success and result.get("result"):
                        # Show sample of results if available
                        search_result = result["result"]
                        if isinstance(search_result, dict) and "results" in search_result:
                            results = search_result["results"]
                            self.print_results("Search Results Sample", {
                                "total_results": len(results) if isinstance(results, list) else "unknown",
                                "first_result": results[0] if isinstance(results, list) and results else None
                            })
                        
                    return result
                    
                except json.JSONDecodeError as e:
                    self.print_results("JSON Decode Error", {
                        "error": str(e),
                        "response_text": response.text[:500]
                    }, False)
                    return None
            else:
                self.print_results("Execution Failed", {
                    "status_code": response.status_code,
                    "response_text": response.text[:500]
                }, False)
                return None
                
        except Exception as e:
            self.print_results("Direct MCP Execution", {
                "error": str(e),
                "traceback": traceback.format_exc()
            }, False)
            return None
    
    async def test_mcp_bridge_direct(self, enhanced_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Test direct call to MCP bridge if available"""
        self.print_separator("MCP BRIDGE DIRECT TEST")
        
        try:
            # Try direct call to MCP bridge
            response = requests.post(
                f"{self.mcp_bridge_url}/tools/{self.tool_name}",
                json=enhanced_params,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            
            self.print_results("Bridge Response Status", {
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type"),
                "response_size": len(response.content)
            })
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    self.print_results("Bridge Result", result)
                    return result
                except json.JSONDecodeError:
                    self.print_results("Bridge Raw Response", response.text[:500])
                    return {"raw_response": response.text}
            else:
                self.print_results("Bridge Error", {
                    "status_code": response.status_code,
                    "response": response.text[:500]
                }, False)
                return None
                
        except Exception as e:
            self.print_results("MCP Bridge Direct", {
                "error": str(e),
                "note": "Bridge may not be available - this is optional"
            }, False)
            return None
    
    async def test_schema_validation(self, tool_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test schema validation and parameter compatibility"""
        self.print_separator("SCHEMA VALIDATION TEST")
        
        try:
            schema = tool_config.get("parameters", {})
            properties = schema.get("properties", {})
            
            # Check for critical properties
            schema_analysis = {
                "has_query_param": any(param in properties for param in ["query", "q", "search"]),
                "supports_dateRestrict": "dateRestrict" in properties,
                "supports_num": "num" in properties,
                "supports_date_restrict_snake": "date_restrict" in properties,
                "supports_num_results": "num_results" in properties,
                "total_properties": len(properties),
                "property_names": list(properties.keys())
            }
            
            # Check specific parameter schemas
            if "dateRestrict" in properties:
                date_restrict_schema = properties["dateRestrict"]
                schema_analysis["dateRestrict_schema"] = date_restrict_schema
                
            if "num" in properties:
                num_schema = properties["num"]
                schema_analysis["num_schema"] = num_schema
            
            self.print_results("Schema Analysis", schema_analysis)
            
            # Test parameter validation
            test_cases = [
                {"query": "test", "dateRestrict": "d", "num": 10},
                {"query": "test", "date_restrict": "m6", "num_results": 5},
                {"q": "test"},
                {}
            ]
            
            validation_results = []
            for i, test_case in enumerate(test_cases):
                try:
                    # Basic validation - check if parameters exist in schema
                    valid_params = {}
                    invalid_params = {}
                    
                    for param, value in test_case.items():
                        if param in properties:
                            valid_params[param] = value
                        else:
                            invalid_params[param] = value
                    
                    validation_results.append({
                        "test_case": i + 1,
                        "input": test_case,
                        "valid_params": valid_params,
                        "invalid_params": invalid_params,
                        "is_valid": len(invalid_params) == 0
                    })
                except Exception as e:
                    validation_results.append({
                        "test_case": i + 1,
                        "input": test_case,
                        "error": str(e)
                    })
            
            self.print_results("Parameter Validation Tests", validation_results)
            
            return schema_analysis
            
        except Exception as e:
            self.print_results("Schema Validation", {
                "error": str(e),
                "traceback": traceback.format_exc()
            }, False)
            return {}
    
    async def generate_test_report(self, results: Dict[str, Any]):
        """Generate comprehensive test report"""
        self.print_separator("COMPREHENSIVE TEST REPORT", "=")
        
        # Overall status
        critical_tests = ["backend_connectivity", "tools_cache", "tool_found"]
        critical_passed = all(results.get(test, False) for test in critical_tests)
        
        functionality_tests = ["parameter_injection", "execution_success"]
        functionality_passed = all(results.get(test, False) for test in functionality_tests)
        
        overall_status = "WORKING" if critical_passed and functionality_passed else "ISSUES DETECTED"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "critical_systems": {
                "backend_accessible": results.get("backend_connectivity", False),
                "tools_cache_working": results.get("tools_cache", False),
                "google_search_configured": results.get("tool_found", False)
            },
            "functionality": {
                "parameter_injection_working": results.get("parameter_injection", False),
                "schema_validation_passed": results.get("schema_validation", False),
                "execution_successful": results.get("execution_success", False),
                "uses_camelCase_params": results.get("uses_camelcase", False)
            },
            "optional_features": {
                "mcp_bridge_accessible": results.get("bridge_connectivity", False),
                "direct_bridge_execution": results.get("bridge_execution", False)
            },
            "issues_found": results.get("issues", []),
            "recommendations": results.get("recommendations", [])
        }
        
        self.print_results("FINAL TEST REPORT", report)
        
        # Save report to file
        report_filename = f"google_search_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📊 Detailed report saved to: {report_filename}")
        
        return report
    
    async def run_all_tests(self):
        """Run all tests and generate comprehensive report"""
        print("🚀 Starting Google Search MCP Tool Verification Tests")
        print(f"⏰ Test started at: {datetime.now().isoformat()}")
        
        results = {
            "issues": [],
            "recommendations": []
        }
        
        try:
            # Test 1: Backend connectivity
            results["backend_connectivity"] = await self.test_backend_connectivity()
            if not results["backend_connectivity"]:
                results["issues"].append("Backend services not accessible")
                results["recommendations"].append("Start backend services with ./run_local.sh")
            
            # Test 2: MCP tools cache
            tool_config = await self.test_mcp_tools_cache()
            results["tools_cache"] = tool_config is not None
            results["tool_found"] = tool_config is not None
            
            if not tool_config:
                results["issues"].append("Google Search tool not found in MCP tools cache")
                results["recommendations"].append("Configure Google Search MCP tool in the database")
                return await self.generate_test_report(results)
            
            # Test 3: Schema validation
            schema_analysis = await self.test_schema_validation(tool_config)
            results["schema_validation"] = len(schema_analysis) > 0
            results["uses_camelcase"] = schema_analysis.get("supports_dateRestrict", False) and schema_analysis.get("supports_num", False)
            
            if not results["uses_camelcase"]:
                results["issues"].append("Tool schema does not support camelCase parameters (dateRestrict, num)")
                results["recommendations"].append("Update tool schema to support Google's camelCase parameter format")
            
            # Test 4: Parameter injection
            enhanced_params = await self.test_parameter_injection(tool_config)
            results["parameter_injection"] = "dateRestrict" in enhanced_params or "date_restrict" in enhanced_params
            
            if not results["parameter_injection"]:
                results["issues"].append("Parameter injection not adding temporal parameters")
                results["recommendations"].append("Check MCP parameter injector configuration")
            
            # Test 5: Direct execution
            execution_result = await self.test_direct_mcp_execution(enhanced_params)
            results["execution_success"] = execution_result is not None and execution_result.get("success", False)
            
            if not results["execution_success"]:
                if execution_result and execution_result.get("error"):
                    results["issues"].append(f"Tool execution failed: {execution_result['error']}")
                else:
                    results["issues"].append("Tool execution failed with unknown error")
                results["recommendations"].append("Check MCP server configuration and Google API credentials")
            
            # Test 6: Optional bridge test
            bridge_result = await self.test_mcp_bridge_direct(enhanced_params)
            results["bridge_connectivity"] = bridge_result is not None
            results["bridge_execution"] = bridge_result is not None
            
            # Generate final report
            return await self.generate_test_report(results)
            
        except Exception as e:
            results["issues"].append(f"Critical test failure: {str(e)}")
            logger.error(f"Test suite failed: {e}", exc_info=True)
            return await self.generate_test_report(results)

async def main():
    """Main test execution"""
    tester = GoogleSearchMCPTester()
    report = await tester.run_all_tests()
    
    # Exit with appropriate code
    if report["overall_status"] == "WORKING":
        print("\n🎉 All tests passed! Google Search MCP tool is working correctly.")
        sys.exit(0)
    else:
        print("\n⚠️  Issues detected. Check the report above for details.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())