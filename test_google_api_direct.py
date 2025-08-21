#!/usr/bin/env python3
"""
Direct Google Search API Test Script

This script directly tests the Google Custom Search API to identify the source of 400 errors.
It bypasses all MCP layers and tests the raw API directly.

Key Features:
1. Direct HTTP requests to Google's Custom Search API
2. Tests both with and without dateRestrict parameter
3. Validates API credentials and quota status
4. Provides detailed error analysis
5. Tests various parameter combinations

Usage:
    python test_google_api_direct.py

Requirements:
    pip install requests
"""

import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from urllib.parse import urlencode

import requests

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'google_api_direct_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class GoogleAPIDirectTester:
    """Direct tester for Google Custom Search API"""
    
    # API Configuration (from MCP server config)
    API_KEY = "AIzaSyA2U7MBpH7cNDykiZ_OlGsdJJlXumsMps4"
    SEARCH_ENGINE_ID = "d77ac8c3d3e124c3c"
    BASE_URL = "https://www.googleapis.com/customsearch/v1"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GoogleAPIDirectTester/1.0',
            'Accept': 'application/json'
        })
        
    def print_separator(self, title: str, char: str = "=", width: int = 80):
        """Print a formatted separator with title"""
        title_padded = f" {title} "
        padding = (width - len(title_padded)) // 2
        line = char * padding + title_padded + char * (width - padding - len(title_padded))
        print(f"\\n{line}")
        
    def print_results(self, section: str, data: Any, success: bool = True):
        """Print formatted test results"""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"\\n[{status}] {section}")
        if isinstance(data, dict):
            print(json.dumps(data, indent=2, default=str))
        else:
            print(str(data))
    
    def validate_api_credentials(self) -> bool:
        """Validate that API credentials are properly formatted"""
        self.print_separator("API CREDENTIALS VALIDATION")
        
        validation_results = {
            "api_key_length": len(self.API_KEY),
            "api_key_format": self.API_KEY.startswith("AIza"),
            "search_engine_id_length": len(self.SEARCH_ENGINE_ID),
            "search_engine_id_format": len(self.SEARCH_ENGINE_ID) > 10
        }
        
        all_valid = all(validation_results.values())
        self.print_results("Credential Validation", validation_results, all_valid)
        
        return all_valid
    
    def test_basic_api_connectivity(self) -> bool:
        """Test basic connectivity to Google API endpoint"""
        self.print_separator("BASIC API CONNECTIVITY TEST")
        
        try:
            # Just test if the endpoint is reachable (no query)
            response = self.session.get(
                self.BASE_URL,
                params={"key": self.API_KEY},
                timeout=10
            )
            
            connectivity_info = {
                "endpoint": self.BASE_URL,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "response_time_ms": response.elapsed.total_seconds() * 1000,
                "content_length": len(response.content)
            }
            
            # 400 is expected here (no required params), but endpoint should be reachable
            endpoint_reachable = response.status_code in [400, 403]  # 400 = missing params, 403 = auth issue
            
            self.print_results("API Connectivity", connectivity_info, endpoint_reachable)
            
            if response.status_code == 403:
                try:
                    error_data = response.json()
                    self.print_results("Authentication Error Details", error_data, False)
                except:
                    self.print_results("Authentication Error (Raw)", response.text[:500], False)
            
            return endpoint_reachable
            
        except Exception as e:
            self.print_results("API Connectivity", {
                "error": str(e),
                "error_type": type(e).__name__
            }, False)
            return False
    
    def test_minimal_search_request(self) -> Optional[Dict[str, Any]]:
        """Test the most minimal possible search request"""
        self.print_separator("MINIMAL SEARCH REQUEST TEST")
        
        try:
            # Absolute minimal parameters
            params = {
                "key": self.API_KEY,
                "cx": self.SEARCH_ENGINE_ID,
                "q": "test"
            }
            
            self.print_results("Request Parameters", params)
            
            response = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=15
            )
            
            response_info = {
                "status_code": response.status_code,
                "response_time_ms": response.elapsed.total_seconds() * 1000,
                "content_type": response.headers.get('content-type'),
                "content_length": len(response.content)
            }
            
            self.print_results("Response Info", response_info)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    result_summary = {
                        "search_information": data.get("searchInformation", {}),
                        "total_results": len(data.get("items", [])),
                        "has_results": bool(data.get("items"))
                    }
                    self.print_results("Search Results Summary", result_summary, True)
                    return data
                except json.JSONDecodeError:
                    self.print_results("JSON Decode Error", response.text[:500], False)
                    return None
            else:
                # Log detailed error information
                try:
                    error_data = response.json()
                    self.print_results("API Error Response", error_data, False)
                except:
                    self.print_results("Raw Error Response", response.text[:1000], False)
                
                return None
                
        except Exception as e:
            self.print_results("Minimal Search Request", {
                "error": str(e),
                "error_type": type(e).__name__
            }, False)
            return None
    
    def test_date_restrict_parameter(self) -> List[Dict[str, Any]]:
        """Test dateRestrict parameter with various values"""
        self.print_separator("DATE RESTRICT PARAMETER TEST")
        
        test_cases = [
            {"dateRestrict": "d", "description": "Past day"},
            {"dateRestrict": "w", "description": "Past week"}, 
            {"dateRestrict": "m", "description": "Past month"},
            {"dateRestrict": "y", "description": "Past year"},
            {"dateRestrict": "m6", "description": "Past 6 months"},
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"\\n--- Test Case {i+1}: {test_case['description']} ---")
            
            params = {
                "key": self.API_KEY,
                "cx": self.SEARCH_ENGINE_ID,
                "q": "AI developments",
                "dateRestrict": test_case["dateRestrict"]
            }
            
            try:
                response = self.session.get(
                    self.BASE_URL,
                    params=params,
                    timeout=15
                )
                
                test_result = {
                    "test_case": test_case["description"],
                    "dateRestrict_value": test_case["dateRestrict"],
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        test_result.update({
                            "total_results": len(data.get("items", [])),
                            "search_time": data.get("searchInformation", {}).get("searchTime")
                        })
                    except json.JSONDecodeError:
                        test_result["json_decode_error"] = True
                else:
                    try:
                        error_data = response.json()
                        test_result["error"] = error_data
                    except:
                        test_result["raw_error"] = response.text[:300]
                
                self.print_results(f"dateRestrict='{test_case['dateRestrict']}'", test_result, test_result["success"])
                results.append(test_result)
                
                # Small delay between requests to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                error_result = {
                    "test_case": test_case["description"],
                    "dateRestrict_value": test_case["dateRestrict"],
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "success": False
                }
                self.print_results(f"dateRestrict='{test_case['dateRestrict']}'", error_result, False)
                results.append(error_result)
        
        return results
    
    def test_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Test various parameter combinations"""
        self.print_separator("PARAMETER COMBINATIONS TEST")
        
        test_combinations = [
            {
                "name": "Basic query only",
                "params": {"q": "machine learning"}
            },
            {
                "name": "Query + num results",
                "params": {"q": "machine learning", "num": 5}
            },
            {
                "name": "Query + dateRestrict",
                "params": {"q": "machine learning", "dateRestrict": "d"}
            },
            {
                "name": "Query + num + dateRestrict",
                "params": {"q": "machine learning", "num": 3, "dateRestrict": "w"}
            },
            {
                "name": "Query + sort",
                "params": {"q": "machine learning", "sort": "date"}
            },
            {
                "name": "All parameters",
                "params": {"q": "machine learning", "num": 8, "dateRestrict": "m", "sort": "date"}
            }
        ]
        
        results = []
        
        for i, test_combo in enumerate(test_combinations):
            print(f"\\n--- Combination {i+1}: {test_combo['name']} ---")
            
            # Add required API credentials to each test
            full_params = {
                "key": self.API_KEY,
                "cx": self.SEARCH_ENGINE_ID,
                **test_combo["params"]
            }
            
            try:
                response = self.session.get(
                    self.BASE_URL,
                    params=full_params,
                    timeout=15
                )
                
                combo_result = {
                    "combination_name": test_combo["name"],
                    "parameters": test_combo["params"],
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        combo_result.update({
                            "total_results": len(data.get("items", [])),
                            "search_time": data.get("searchInformation", {}).get("searchTime"),
                            "total_results_estimate": data.get("searchInformation", {}).get("totalResults")
                        })
                    except json.JSONDecodeError:
                        combo_result["json_decode_error"] = True
                else:
                    try:
                        error_data = response.json()
                        combo_result["error"] = error_data
                    except:
                        combo_result["raw_error"] = response.text[:300]
                
                self.print_results(test_combo["name"], combo_result, combo_result["success"])
                results.append(combo_result)
                
                # Delay between requests
                time.sleep(0.7)
                
            except Exception as e:
                error_result = {
                    "combination_name": test_combo["name"],
                    "parameters": test_combo["params"],
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "success": False
                }
                self.print_results(test_combo["name"], error_result, False)
                results.append(error_result)
        
        return results
    
    def test_quota_and_limits(self) -> Dict[str, Any]:
        """Test API quota and rate limits"""
        self.print_separator("QUOTA AND LIMITS TEST")
        
        # Make several rapid requests to test limits
        quota_test_results = {
            "rapid_requests": [],
            "quota_exceeded": False,
            "rate_limited": False
        }
        
        for i in range(5):
            try:
                params = {
                    "key": self.API_KEY,
                    "cx": self.SEARCH_ENGINE_ID,
                    "q": f"quota test {i}"
                }
                
                start_time = time.time()
                response = self.session.get(self.BASE_URL, params=params, timeout=10)
                end_time = time.time()
                
                request_result = {
                    "request_number": i + 1,
                    "status_code": response.status_code,
                    "response_time_s": end_time - start_time,
                    "success": response.status_code == 200
                }
                
                if response.status_code == 429:
                    quota_test_results["rate_limited"] = True
                    request_result["rate_limited"] = True
                elif response.status_code == 403:
                    try:
                        error_data = response.json()
                        if "quotaExceeded" in str(error_data):
                            quota_test_results["quota_exceeded"] = True
                            request_result["quota_exceeded"] = True
                    except:
                        pass
                
                quota_test_results["rapid_requests"].append(request_result)
                
                # No delay for quota testing
                
            except Exception as e:
                quota_test_results["rapid_requests"].append({
                    "request_number": i + 1,
                    "error": str(e),
                    "success": False
                })
        
        self.print_results("Quota Test Results", quota_test_results)
        return quota_test_results
    
    def test_invalid_parameters(self) -> List[Dict[str, Any]]:
        """Test with invalid parameters to understand error responses"""
        self.print_separator("INVALID PARAMETERS TEST")
        
        invalid_test_cases = [
            {
                "name": "Invalid dateRestrict value",
                "params": {"q": "test", "dateRestrict": "invalid_value"}
            },
            {
                "name": "Invalid num value (too high)",
                "params": {"q": "test", "num": 100}
            },
            {
                "name": "Invalid sort value",
                "params": {"q": "test", "sort": "invalid_sort"}
            },
            {
                "name": "Missing query parameter",
                "params": {"dateRestrict": "d"}
            },
            {
                "name": "Unknown parameter",
                "params": {"q": "test", "unknown_param": "value"}
            }
        ]
        
        results = []
        
        for test_case in invalid_test_cases:
            print(f"\\n--- Invalid Test: {test_case['name']} ---")
            
            full_params = {
                "key": self.API_KEY,
                "cx": self.SEARCH_ENGINE_ID,
                **test_case["params"]
            }
            
            try:
                response = self.session.get(
                    self.BASE_URL,
                    params=full_params,
                    timeout=10
                )
                
                test_result = {
                    "test_name": test_case["name"],
                    "parameters": test_case["params"],
                    "status_code": response.status_code,
                    "expected_error": True
                }
                
                if response.status_code != 200:
                    try:
                        error_data = response.json()
                        test_result["error_details"] = error_data
                    except:
                        test_result["raw_error"] = response.text[:500]
                else:
                    test_result["unexpected_success"] = True
                
                self.print_results(test_case["name"], test_result)
                results.append(test_result)
                
                time.sleep(0.3)
                
            except Exception as e:
                error_result = {
                    "test_name": test_case["name"],
                    "parameters": test_case["params"],
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                self.print_results(test_case["name"], error_result, False)
                results.append(error_result)
        
        return results
    
    def generate_comprehensive_report(self, all_results: Dict[str, Any]):
        """Generate a comprehensive test report"""
        self.print_separator("COMPREHENSIVE TEST REPORT", "=")
        
        # Analyze all results
        successful_tests = []
        failed_tests = []
        error_patterns = {}
        
        for test_category, results in all_results.items():
            if test_category == "credentials_valid":
                if results:
                    successful_tests.append("Credentials validation")
                else:
                    failed_tests.append("Credentials validation")
            elif test_category == "basic_connectivity":
                if results:
                    successful_tests.append("Basic API connectivity")
                else:
                    failed_tests.append("Basic API connectivity")
            elif test_category == "minimal_search":
                if results:
                    successful_tests.append("Minimal search request")
                else:
                    failed_tests.append("Minimal search request")
            elif isinstance(results, list):
                for result in results:
                    if result.get("success", False):
                        successful_tests.append(f"{test_category}: {result.get('test_case', result.get('combination_name', result.get('test_name', 'unknown')))}")
                    else:
                        failed_tests.append(f"{test_category}: {result.get('test_case', result.get('combination_name', result.get('test_name', 'unknown')))}")
                        
                        # Collect error patterns
                        if "error" in result:
                            error_msg = str(result["error"])[:100]
                            error_patterns[error_msg] = error_patterns.get(error_msg, 0) + 1
                        elif "status_code" in result and result["status_code"] != 200:
                            status_code = result["status_code"]
                            error_patterns[f"HTTP {status_code}"] = error_patterns.get(f"HTTP {status_code}", 0) + 1
        
        # Generate summary
        total_tests = len(successful_tests) + len(failed_tests)
        success_rate = (len(successful_tests) / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "success_rate_percent": round(success_rate, 2)
            },
            "credentials_status": all_results.get("credentials_valid", False),
            "api_connectivity": all_results.get("basic_connectivity", False),
            "minimal_search_works": bool(all_results.get("minimal_search")),
            "successful_tests": successful_tests[:10],  # Limit for readability
            "failed_tests": failed_tests[:10],  # Limit for readability
            "error_patterns": error_patterns,
            "recommendations": []
        }
        
        # Generate recommendations based on results
        if not all_results.get("credentials_valid"):
            report["recommendations"].append("Verify Google API credentials format and validity")
        
        if not all_results.get("basic_connectivity"):
            report["recommendations"].append("Check network connectivity and firewall settings")
        
        if not all_results.get("minimal_search"):
            report["recommendations"].append("Basic search functionality is failing - check API key and search engine ID")
        
        if "HTTP 403" in error_patterns:
            report["recommendations"].append("403 errors suggest authentication/authorization issues - check API key permissions")
        
        if "HTTP 400" in error_patterns:
            report["recommendations"].append("400 errors suggest invalid request parameters - review parameter format")
        
        if "HTTP 429" in error_patterns:
            report["recommendations"].append("429 errors indicate rate limiting - implement request throttling")
        
        self.print_results("FINAL ANALYSIS", report)
        
        # Save detailed report
        report_filename = f"google_api_direct_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump({**report, "detailed_results": all_results}, f, indent=2, default=str)
        
        print(f"\\n📄 Comprehensive report saved to: {report_filename}")
        
        return report
    
    def run_all_tests(self):
        """Run all direct API tests"""
        print("🚀 Starting Google Custom Search API Direct Testing")
        print(f"🕐 Test started at: {datetime.now().isoformat()}")
        print(f"🔑 API Key: {self.API_KEY[:10]}...{self.API_KEY[-4:]}")
        print(f"🔍 Search Engine ID: {self.SEARCH_ENGINE_ID}")
        
        all_results = {}
        
        try:
            # Test 1: Validate credentials format
            all_results["credentials_valid"] = self.validate_api_credentials()
            
            # Test 2: Basic connectivity
            all_results["basic_connectivity"] = self.test_basic_api_connectivity()
            
            # Test 3: Minimal search request
            all_results["minimal_search"] = self.test_minimal_search_request()
            
            # Test 4: Date restrict parameter
            all_results["date_restrict_tests"] = self.test_date_restrict_parameter()
            
            # Test 5: Parameter combinations
            all_results["parameter_combinations"] = self.test_parameter_combinations()
            
            # Test 6: Quota and limits
            all_results["quota_tests"] = self.test_quota_and_limits()
            
            # Test 7: Invalid parameters
            all_results["invalid_parameter_tests"] = self.test_invalid_parameters()
            
            # Generate comprehensive report
            return self.generate_comprehensive_report(all_results)
            
        except Exception as e:
            logger.error(f"Critical test failure: {e}", exc_info=True)
            self.print_results("CRITICAL ERROR", {
                "error": str(e),
                "error_type": type(e).__name__
            }, False)
            return {"error": str(e)}

def main():
    """Main test execution"""
    print("=" * 80)
    print("🔍 GOOGLE CUSTOM SEARCH API DIRECT TESTER")
    print("=" * 80)
    print("This script tests Google's Custom Search API directly to identify 400 error sources")
    print("Bypassing all MCP layers for raw API testing\\n")
    
    tester = GoogleAPIDirectTester()
    report = tester.run_all_tests()
    
    # Exit with appropriate code
    if report.get("test_summary", {}).get("success_rate_percent", 0) >= 70:
        print("\\n🎉 Most tests passed! API appears to be working correctly.")
        sys.exit(0)
    else:
        print("\\n⚠️  Significant issues detected. Check the detailed report above.")
        sys.exit(1)

if __name__ == "__main__":
    main()