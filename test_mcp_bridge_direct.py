#!/usr/bin/env python3
"""
Direct MCP Bridge Server Test Script

This script tests the MCP bridge server at localhost:3001 directly to isolate:
1. Whether the 400 error is from the MCP bridge server itself
2. Parameter transformation issues (snake_case vs camelCase)
3. Whether the bridge correctly forwards to the Google API

The goal is to identify if the issue is in:
- The MCP bridge layer processing requests
- Parameter naming/formatting issues
- The actual Google API integration
"""

import json
import logging
import requests
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_mcp_bridge_direct.log')
    ]
)
logger = logging.getLogger(__name__)

class MCPBridgeDirectTester:
    """Direct tester for the MCP bridge server to isolate issues"""
    
    def __init__(self, bridge_url: str = "http://localhost:3001"):
        self.bridge_url = bridge_url.rstrip('/')
        self.test_results = []
        
    def log_section(self, title: str, level: str = "INFO"):
        """Log a section header with visual formatting"""
        separator = "=" * 60
        logger.info(f"\n{separator}")
        logger.info(f"{title}")
        logger.info(f"{separator}")
        
    def log_request_details(self, method: str, url: str, headers: Dict[str, Any], data: Any):
        """Log detailed request information"""
        logger.info(f"REQUEST: {method} {url}")
        logger.info(f"Headers: {json.dumps(headers, indent=2)}")
        if data:
            logger.info(f"Body: {json.dumps(data, indent=2)}")
        else:
            logger.info("Body: None")
            
    def log_response_details(self, response: requests.Response):
        """Log detailed response information"""
        logger.info(f"RESPONSE: Status {response.status_code}")
        logger.info(f"Headers: {json.dumps(dict(response.headers), indent=2)}")
        
        # Try to parse response as JSON first
        try:
            response_json = response.json()
            logger.info(f"JSON Body: {json.dumps(response_json, indent=2)}")
        except json.JSONDecodeError:
            logger.info(f"Raw Body: {response.text[:1000]}...")
            
    def test_health_endpoint(self) -> bool:
        """Test the bridge server health endpoint"""
        self.log_section("TESTING BRIDGE HEALTH ENDPOINT")
        
        try:
            url = f"{self.bridge_url}/health"
            headers = {"Content-Type": "application/json"}
            
            self.log_request_details("GET", url, headers, None)
            
            response = requests.get(url, headers=headers, timeout=10)
            
            self.log_response_details(response)
            
            success = response.status_code == 200
            self.test_results.append({
                "test": "health_endpoint",
                "success": success,
                "status_code": response.status_code,
                "response": response.text[:500] if not success else "OK"
            })
            
            if success:
                logger.info("✅ Bridge server is accessible")
            else:
                logger.error(f"❌ Bridge server health check failed: {response.status_code}")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Health check exception: {e}")
            self.test_results.append({
                "test": "health_endpoint",
                "success": False,
                "error": str(e)
            })
            return False
    
    def test_tools_list(self) -> bool:
        """Test the bridge server tools list endpoint"""
        self.log_section("TESTING BRIDGE TOOLS LIST")
        
        try:
            url = f"{self.bridge_url}/tools"
            headers = {"Content-Type": "application/json"}
            
            self.log_request_details("GET", url, headers, None)
            
            response = requests.get(url, headers=headers, timeout=10)
            
            self.log_response_details(response)
            
            success = response.status_code == 200
            
            if success:
                try:
                    tools_data = response.json()
                    total_tools = tools_data.get("total", 0)
                    tools = tools_data.get("tools", [])
                    
                    # Check if google_search is available
                    google_search_available = any(
                        tool.get("name") == "google_search" for tool in tools
                    )
                    
                    logger.info(f"✅ Bridge has {total_tools} tools available")
                    logger.info(f"Google Search available: {google_search_available}")
                    
                    # Show first few tools
                    for tool in tools[:5]:
                        logger.info(f"  - {tool.get('name')}: {tool.get('description', 'No description')}")
                    
                    self.test_results.append({
                        "test": "tools_list",
                        "success": True,
                        "total_tools": total_tools,
                        "google_search_available": google_search_available,
                        "sample_tools": [t.get("name") for t in tools[:5]]
                    })
                    
                    return google_search_available
                    
                except json.JSONDecodeError:
                    logger.error("❌ Failed to parse tools list response")
                    success = False
            
            if not success:
                logger.error(f"❌ Tools list failed: {response.status_code}")
            
            self.test_results.append({
                "test": "tools_list",
                "success": success,
                "status_code": response.status_code,
                "response": response.text[:500] if not success else "OK"
            })
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Tools list exception: {e}")
            self.test_results.append({
                "test": "tools_list",
                "success": False,
                "error": str(e)
            })
            return False
    
    def test_google_search_snake_case(self) -> Dict[str, Any]:
        """Test Google Search with snake_case parameters"""
        self.log_section("TESTING GOOGLE SEARCH - SNAKE_CASE PARAMETERS")
        
        url = f"{self.bridge_url}/tools/google_search"
        headers = {"Content-Type": "application/json"}
        
        # Snake_case parameters (what the original system might send)
        data = {
            "query": "test",
            "num_results": 5,
            "date_restrict": "d"
        }
        
        try:
            self.log_request_details("POST", url, headers, data)
            
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            self.log_response_details(response)
            
            result = {
                "test": "google_search_snake_case",
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "parameters_sent": data
            }
            
            if response.status_code == 200:
                try:
                    response_json = response.json()
                    result["response_success"] = response_json.get("success", False)
                    result["has_results"] = bool(response_json.get("result"))
                    result["execution_time"] = response_json.get("execution_time", 0)
                    logger.info("✅ Snake_case parameters accepted")
                except json.JSONDecodeError:
                    result["json_error"] = "Failed to parse JSON response"
                    logger.error("❌ Failed to parse response JSON")
            else:
                result["error_response"] = response.text[:500]
                logger.error(f"❌ Snake_case parameters failed: {response.status_code}")
                
            self.test_results.append(result)
            return result
            
        except Exception as e:
            logger.error(f"❌ Snake_case test exception: {e}")
            result = {
                "test": "google_search_snake_case",
                "success": False,
                "error": str(e),
                "parameters_sent": data
            }
            self.test_results.append(result)
            return result
    
    def test_google_search_camel_case(self) -> Dict[str, Any]:
        """Test Google Search with camelCase parameters"""
        self.log_section("TESTING GOOGLE SEARCH - CAMELCASE PARAMETERS")
        
        url = f"{self.bridge_url}/tools/google_search"
        headers = {"Content-Type": "application/json"}
        
        # CamelCase parameters (what Google API expects)
        data = {
            "query": "test",
            "num": 5,
            "dateRestrict": "d"
        }
        
        try:
            self.log_request_details("POST", url, headers, data)
            
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            self.log_response_details(response)
            
            result = {
                "test": "google_search_camel_case",
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "parameters_sent": data
            }
            
            if response.status_code == 200:
                try:
                    response_json = response.json()
                    result["response_success"] = response_json.get("success", False)
                    result["has_results"] = bool(response_json.get("result"))
                    result["execution_time"] = response_json.get("execution_time", 0)
                    logger.info("✅ CamelCase parameters accepted")
                except json.JSONDecodeError:
                    result["json_error"] = "Failed to parse JSON response"
                    logger.error("❌ Failed to parse response JSON")
            else:
                result["error_response"] = response.text[:500]
                logger.error(f"❌ CamelCase parameters failed: {response.status_code}")
                
            self.test_results.append(result)
            return result
            
        except Exception as e:
            logger.error(f"❌ CamelCase test exception: {e}")
            result = {
                "test": "google_search_camel_case",
                "success": False,
                "error": str(e),
                "parameters_sent": data
            }
            self.test_results.append(result)
            return result
    
    def test_google_search_minimal(self) -> Dict[str, Any]:
        """Test Google Search with minimal parameters"""
        self.log_section("TESTING GOOGLE SEARCH - MINIMAL PARAMETERS")
        
        url = f"{self.bridge_url}/tools/google_search"
        headers = {"Content-Type": "application/json"}
        
        # Minimal parameters (just query)
        data = {"query": "test"}
        
        try:
            self.log_request_details("POST", url, headers, data)
            
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            self.log_response_details(response)
            
            result = {
                "test": "google_search_minimal",
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "parameters_sent": data
            }
            
            if response.status_code == 200:
                try:
                    response_json = response.json()
                    result["response_success"] = response_json.get("success", False)
                    result["has_results"] = bool(response_json.get("result"))
                    result["execution_time"] = response_json.get("execution_time", 0)
                    logger.info("✅ Minimal parameters accepted")
                except json.JSONDecodeError:
                    result["json_error"] = "Failed to parse JSON response"
                    logger.error("❌ Failed to parse response JSON")
            else:
                result["error_response"] = response.text[:500]
                logger.error(f"❌ Minimal parameters failed: {response.status_code}")
                
            self.test_results.append(result)
            return result
            
        except Exception as e:
            logger.error(f"❌ Minimal test exception: {e}")
            result = {
                "test": "google_search_minimal",
                "success": False,
                "error": str(e),
                "parameters_sent": data
            }
            self.test_results.append(result)
            return result
    
    def test_google_search_invalid_params(self) -> Dict[str, Any]:
        """Test Google Search with invalid parameters to see how it handles errors"""
        self.log_section("TESTING GOOGLE SEARCH - INVALID PARAMETERS")
        
        url = f"{self.bridge_url}/tools/google_search"
        headers = {"Content-Type": "application/json"}
        
        # Invalid parameters (missing query)
        data = {"num_results": 5}
        
        try:
            self.log_request_details("POST", url, headers, data)
            
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            self.log_response_details(response)
            
            result = {
                "test": "google_search_invalid",
                "status_code": response.status_code,
                "parameters_sent": data,
                "expected_failure": True
            }
            
            # We expect this to fail, so 400 is actually success for this test
            if response.status_code == 400:
                try:
                    error_response = response.json()
                    result["success"] = True
                    result["error_handled_properly"] = True
                    result["error_message"] = error_response.get("detail", "No detail")
                    logger.info("✅ Invalid parameters properly rejected with 400")
                except json.JSONDecodeError:
                    result["success"] = False
                    result["json_error"] = "Failed to parse error response"
            else:
                result["success"] = False
                result["unexpected_status"] = response.status_code
                result["response"] = response.text[:500]
                logger.error(f"❌ Unexpected status for invalid params: {response.status_code}")
                
            self.test_results.append(result)
            return result
            
        except Exception as e:
            logger.error(f"❌ Invalid params test exception: {e}")
            result = {
                "test": "google_search_invalid",
                "success": False,
                "error": str(e),
                "parameters_sent": data
            }
            self.test_results.append(result)
            return result
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        self.log_section("ANALYSIS REPORT")
        
        # Analyze patterns in the results
        status_codes = {}
        successful_tests = 0
        
        for result in self.test_results:
            test_name = result.get("test", "unknown")
            success = result.get("success", False)
            status_code = result.get("status_code")
            
            if success:
                successful_tests += 1
                
            if status_code:
                if status_code not in status_codes:
                    status_codes[status_code] = []
                status_codes[status_code].append(test_name)
        
        # Key findings
        logger.info("KEY FINDINGS:")
        logger.info(f"  - {successful_tests}/{len(self.test_results)} tests successful")
        logger.info(f"  - Status codes observed: {list(status_codes.keys())}")
        
        # Analyze 400 errors specifically
        if 400 in status_codes:
            logger.info(f"  - 400 errors in: {status_codes[400]}")
            
            # Check if 400 errors are for both valid and invalid requests
            valid_request_tests = ["google_search_snake_case", "google_search_camel_case", "google_search_minimal"]
            invalid_request_tests = ["google_search_invalid"]
            
            valid_400s = [test for test in status_codes[400] if test in valid_request_tests]
            invalid_400s = [test for test in status_codes[400] if test in invalid_request_tests]
            
            if valid_400s:
                logger.error(f"  ❌ ISSUE: Valid requests getting 400 errors: {valid_400s}")
                logger.error("  This suggests parameter transformation or validation issues in the bridge")
            
            if invalid_400s:
                logger.info(f"  ✅ Good: Invalid requests properly rejected: {invalid_400s}")
        
        # Parameter format analysis
        snake_case_result = next((r for r in self.test_results if r["test"] == "google_search_snake_case"), None)
        camel_case_result = next((r for r in self.test_results if r["test"] == "google_search_camel_case"), None)
        
        if snake_case_result and camel_case_result:
            snake_success = snake_case_result.get("success", False)
            camel_success = camel_case_result.get("success", False)
            
            logger.info("PARAMETER FORMAT ANALYSIS:")
            logger.info(f"  - Snake_case (num_results, date_restrict): {'✅ Works' if snake_success else '❌ Failed'}")
            logger.info(f"  - CamelCase (num, dateRestrict): {'✅ Works' if camel_success else '❌ Failed'}")
            
            if not snake_success and not camel_success:
                logger.error("  ❌ CRITICAL: Neither parameter format works!")
                logger.error("  This suggests a fundamental issue with the bridge or API integration")
            elif snake_success and not camel_success:
                logger.info("  💡 Bridge expects snake_case parameters")
            elif not snake_success and camel_success:
                logger.info("  💡 Bridge expects camelCase parameters (Google API format)")
            else:
                logger.info("  ✅ Bridge handles both parameter formats")
        
        # Final diagnosis
        logger.info("DIAGNOSIS:")
        
        health_ok = any(r.get("test") == "health_endpoint" and r.get("success") for r in self.test_results)
        tools_ok = any(r.get("test") == "tools_list" and r.get("success") for r in self.test_results)
        
        if not health_ok:
            logger.error("  ❌ Bridge server is not accessible or healthy")
        elif not tools_ok:
            logger.error("  ❌ Bridge server accessible but tools list failing")
        elif 400 in status_codes and any(test in status_codes[400] for test in valid_request_tests):
            logger.error("  ❌ Bridge server accessible but rejecting valid Google Search requests")
            logger.error("  The issue is likely in parameter validation or transformation in the bridge")
        else:
            logger.info("  ✅ Bridge server appears to be working correctly")
        
        # Save detailed report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "bridge_url": self.bridge_url,
            "test_results": self.test_results,
            "analysis": {
                "total_tests": len(self.test_results),
                "successful_tests": successful_tests,
                "status_codes": status_codes,
                "health_accessible": health_ok,
                "tools_list_working": tools_ok
            }
        }
        
        report_filename = f"mcp_bridge_direct_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"📊 Detailed report saved to: {report_filename}")
        
        return report_data
    
    def run_all_tests(self):
        """Run all tests and generate analysis"""
        logger.info("🚀 Starting MCP Bridge Direct Testing")
        logger.info(f"Target: {self.bridge_url}")
        logger.info(f"Time: {datetime.now().isoformat()}")
        
        try:
            # Test 1: Health check
            health_ok = self.test_health_endpoint()
            
            if not health_ok:
                logger.error("❌ Bridge server not accessible. Stopping tests.")
                return self.generate_analysis_report()
            
            # Test 2: Tools list
            tools_ok = self.test_tools_list()
            
            if not tools_ok:
                logger.warning("⚠️ Tools list not working, but continuing with direct tests")
            
            # Test 3-6: Google Search tests
            self.test_google_search_snake_case()
            self.test_google_search_camel_case()
            self.test_google_search_minimal()
            self.test_google_search_invalid_params()
            
            # Generate final analysis
            return self.generate_analysis_report()
            
        except Exception as e:
            logger.error(f"❌ Test suite failed with exception: {e}")
            return self.generate_analysis_report()

def main():
    """Main execution"""
    print("=" * 80)
    print("MCP BRIDGE DIRECT TEST - Isolating 400 Error Source")
    print("=" * 80)
    print()
    print("This script will test the MCP bridge server at localhost:3001 directly")
    print("to determine if the 400 error is coming from:")
    print("  1. The bridge server itself")
    print("  2. Parameter transformation issues")
    print("  3. Google API integration problems")
    print()
    
    tester = MCPBridgeDirectTester()
    report = tester.run_all_tests()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE - Check the logs above for detailed analysis")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())