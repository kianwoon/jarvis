#!/usr/bin/env python3
"""
Meta Task Template Test Script
Tests that the Meta Tasks template editing properly saves and loads the default_settings configuration.

This script validates:
1. Creating templates with specific default_settings
2. Retrieving templates and verifying all settings are preserved  
3. Updating templates with different settings
4. Verifying updates are correctly saved and retrieved

Usage: python test_meta_task_template.py
"""

import requests
import json
import sys
from typing import Dict, Any, Optional
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1/meta-task"

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def print_result(test_name: str, passed: bool, message: str = ""):
    """Print test result with formatting"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} {test_name}")
    if message:
        print(f"     {message}")

def deep_compare_settings(expected: Dict[str, Any], actual: Dict[str, Any], path: str = "") -> tuple[bool, str]:
    """Deep compare two settings dictionaries"""
    for key, expected_value in expected.items():
        current_path = f"{path}.{key}" if path else key
        
        if key not in actual:
            return False, f"Missing key: {current_path}"
        
        actual_value = actual[key]
        
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict):
                return False, f"Type mismatch at {current_path}: expected dict, got {type(actual_value)}"
            result, error = deep_compare_settings(expected_value, actual_value, current_path)
            if not result:
                return False, error
        elif isinstance(expected_value, list):
            if not isinstance(actual_value, list):
                return False, f"Type mismatch at {current_path}: expected list, got {type(actual_value)}"
            if len(expected_value) != len(actual_value):
                return False, f"List length mismatch at {current_path}"
            for i, (exp_item, act_item) in enumerate(zip(expected_value, actual_value)):
                if exp_item != act_item:
                    return False, f"List item mismatch at {current_path}[{i}]"
        else:
            if expected_value != actual_value:
                return False, f"Value mismatch at {current_path}: expected {expected_value}, got {actual_value}"
    
    return True, ""

class MetaTaskTemplateTest:
    """Test suite for Meta Task Template default_settings functionality"""
    
    def __init__(self):
        self.session = requests.Session()
        self.test_template_id: Optional[str] = None
        self.test_results = []
    
    def run_test(self, test_name: str, test_func) -> bool:
        """Run a test and record results"""
        try:
            result, message = test_func()
            print_result(test_name, result, message)
            self.test_results.append((test_name, result, message))
            return result
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            print_result(test_name, False, error_msg)
            self.test_results.append((test_name, False, error_msg))
            return False
    
    def test_api_health(self) -> tuple[bool, str]:
        """Test that the API is accessible"""
        try:
            response = self.session.get(f"{API_BASE}/health")
            if response.status_code == 200:
                return True, f"API healthy - status: {response.json().get('status', 'unknown')}"
            else:
                return False, f"API unhealthy - status code: {response.status_code}"
        except Exception as e:
            return False, f"API connection failed: {str(e)}"
    
    def test_create_template_with_settings(self) -> tuple[bool, str]:
        """Test creating a template with specific default_settings"""
        test_default_settings = {
            "analyzer_model": {
                "temperature": 0.8,
                "model_name": "gpt-4",
                "max_tokens": 4096
            },
            "generator_model": {
                "temperature": 0.5,
                "max_tokens": 8192,
                "model_name": "claude-3-sonnet"
            },
            "reviewer_model": {
                "temperature": 0.2,
                "system_prompt": "Test reviewer prompt",
                "model_name": "gpt-4-turbo"
            },
            "execution": {
                "max_phases": 15,
                "timeout_seconds": 300,
                "retry_attempts": 3
            },
            "quality_control": {
                "factuality_threshold": 0.85,
                "relevance_threshold": 0.75,
                "coherence_threshold": 0.80
            }
        }
        
        template_data = {
            "name": f"test_template_{int(datetime.now().timestamp())}",
            "description": "Test template for default_settings validation",
            "template_type": "test_type",
            "template_config": {
                "phases": [
                    {
                        "name": "analysis",
                        "type": "analyzer",
                        "description": "Analyze input data"
                    },
                    {
                        "name": "generation", 
                        "type": "generator",
                        "description": "Generate output"
                    },
                    {
                        "name": "review",
                        "type": "reviewer", 
                        "description": "Review generated content"
                    }
                ]
            },
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"}
                },
                "required": ["content"]
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "result": {"type": "string"}
                }
            },
            "default_settings": test_default_settings
        }
        
        response = self.session.post(f"{API_BASE}/templates", json=template_data)
        
        if response.status_code != 200:
            return False, f"Create failed with status {response.status_code}: {response.text}"
        
        created_template = response.json()
        self.test_template_id = created_template.get("id")
        
        if not self.test_template_id:
            return False, "No template ID returned"
        
        # Verify the default_settings were saved correctly
        if "default_settings" not in created_template:
            return False, "default_settings missing from response"
        
        result, error = deep_compare_settings(test_default_settings, created_template["default_settings"])
        if not result:
            return False, f"Settings mismatch: {error}"
        
        return True, f"Template created with ID: {self.test_template_id}"
    
    def test_retrieve_template_settings(self) -> tuple[bool, str]:
        """Test retrieving a template and verifying default_settings are preserved"""
        if not self.test_template_id:
            return False, "No template ID available"
        
        response = self.session.get(f"{API_BASE}/templates/{self.test_template_id}")
        
        if response.status_code != 200:
            return False, f"Retrieve failed with status {response.status_code}: {response.text}"
        
        template = response.json()
        
        if "default_settings" not in template:
            return False, "default_settings missing from retrieved template"
        
        # Verify the specific test values
        settings = template["default_settings"]
        expected_values = {
            "analyzer_model.temperature": 0.8,
            "generator_model.max_tokens": 8192,
            "reviewer_model.system_prompt": "Test reviewer prompt",
            "execution.max_phases": 15,
            "quality_control.factuality_threshold": 0.85
        }
        
        for path, expected_value in expected_values.items():
            keys = path.split('.')
            current = settings
            try:
                for key in keys:
                    current = current[key]
                if current != expected_value:
                    return False, f"Value mismatch at {path}: expected {expected_value}, got {current}"
            except (KeyError, TypeError):
                return False, f"Missing or invalid path: {path}"
        
        return True, "All expected values verified in retrieved template"
    
    def test_update_template_settings(self) -> tuple[bool, str]:
        """Test updating a template with different default_settings"""
        if not self.test_template_id:
            return False, "No template ID available"
        
        updated_settings = {
            "analyzer_model": {
                "temperature": 0.3,  # Changed from 0.8
                "model_name": "claude-3-haiku",  # Changed
                "max_tokens": 2048  # Changed from 4096
            },
            "generator_model": {
                "temperature": 0.7,  # Changed from 0.5
                "max_tokens": 12288,  # Changed from 8192
                "model_name": "gpt-4-turbo"  # Changed
            },
            "reviewer_model": {
                "temperature": 0.1,  # Changed from 0.2
                "system_prompt": "Updated reviewer prompt for testing",  # Changed
                "model_name": "claude-3-opus"  # Changed
            },
            "execution": {
                "max_phases": 25,  # Changed from 15
                "timeout_seconds": 600,  # Changed from 300
                "retry_attempts": 5,  # Changed from 3
                "parallel_execution": True  # New field
            },
            "quality_control": {
                "factuality_threshold": 0.90,  # Changed from 0.85
                "relevance_threshold": 0.80,  # Changed from 0.75
                "coherence_threshold": 0.85,  # Changed from 0.80
                "enable_scoring": True  # New field
            },
            "new_section": {  # Completely new section
                "feature_enabled": True,
                "configuration": {
                    "option_a": "value_a",
                    "option_b": 42
                }
            }
        }
        
        update_data = {
            "default_settings": updated_settings
        }
        
        response = self.session.put(f"{API_BASE}/templates/{self.test_template_id}", json=update_data)
        
        if response.status_code != 200:
            return False, f"Update failed with status {response.status_code}: {response.text}"
        
        updated_template = response.json()
        
        if "default_settings" not in updated_template:
            return False, "default_settings missing from updated template"
        
        result, error = deep_compare_settings(updated_settings, updated_template["default_settings"])
        if not result:
            return False, f"Updated settings mismatch: {error}"
        
        return True, "Template updated successfully with new settings"
    
    def test_retrieve_updated_settings(self) -> tuple[bool, str]:
        """Test retrieving the updated template and verifying changes were persisted"""
        if not self.test_template_id:
            return False, "No template ID available"
        
        response = self.session.get(f"{API_BASE}/templates/{self.test_template_id}")
        
        if response.status_code != 200:
            return False, f"Retrieve failed with status {response.status_code}: {response.text}"
        
        template = response.json()
        
        if "default_settings" not in template:
            return False, "default_settings missing from retrieved updated template"
        
        # Verify the updated specific test values
        settings = template["default_settings"]
        expected_updated_values = {
            "analyzer_model.temperature": 0.3,  # Updated value
            "generator_model.max_tokens": 12288,  # Updated value
            "reviewer_model.system_prompt": "Updated reviewer prompt for testing",  # Updated value
            "execution.max_phases": 25,  # Updated value
            "quality_control.factuality_threshold": 0.90,  # Updated value
            "new_section.feature_enabled": True,  # New field
            "new_section.configuration.option_b": 42  # New nested field
        }
        
        for path, expected_value in expected_updated_values.items():
            keys = path.split('.')
            current = settings
            try:
                for key in keys:
                    current = current[key]
                if current != expected_value:
                    return False, f"Updated value mismatch at {path}: expected {expected_value}, got {current}"
            except (KeyError, TypeError):
                return False, f"Missing or invalid updated path: {path}"
        
        return True, "All updated values verified in retrieved template"
    
    def test_complex_nested_settings(self) -> tuple[bool, str]:
        """Test complex nested settings structures"""
        complex_settings = {
            "models": {
                "primary": {
                    "config": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "frequency_penalty": 0.1,
                        "presence_penalty": 0.2
                    },
                    "fallbacks": [
                        {"model": "gpt-4", "priority": 1},
                        {"model": "claude-3-sonnet", "priority": 2}
                    ]
                },
                "secondary": {
                    "config": {
                        "temperature": 0.3,
                        "max_tokens": 1024
                    }
                }
            },
            "workflow": {
                "stages": [
                    {
                        "name": "preprocessing",
                        "enabled": True,
                        "config": {
                            "filters": ["noise_removal", "normalization"],
                            "thresholds": {
                                "min_confidence": 0.6,
                                "max_tokens": 2000
                            }
                        }
                    },
                    {
                        "name": "processing", 
                        "enabled": True,
                        "config": {
                            "parallel_workers": 4,
                            "batch_size": 10
                        }
                    }
                ]
            }
        }
        
        update_data = {
            "default_settings": complex_settings
        }
        
        response = self.session.put(f"{API_BASE}/templates/{self.test_template_id}", json=update_data)
        
        if response.status_code != 200:
            return False, f"Complex update failed with status {response.status_code}: {response.text}"
        
        # Retrieve and verify
        response = self.session.get(f"{API_BASE}/templates/{self.test_template_id}")
        if response.status_code != 200:
            return False, f"Complex retrieve failed with status {response.status_code}"
        
        template = response.json()
        result, error = deep_compare_settings(complex_settings, template["default_settings"])
        if not result:
            return False, f"Complex settings mismatch: {error}"
        
        return True, "Complex nested settings saved and retrieved successfully"
    
    def cleanup(self):
        """Clean up test template"""
        if self.test_template_id:
            try:
                response = self.session.delete(f"{API_BASE}/templates/{self.test_template_id}")
                if response.status_code == 200:
                    print(f"\n🧹 Cleanup: Template {self.test_template_id} deleted successfully")
                else:
                    print(f"\n⚠️  Cleanup warning: Failed to delete template {self.test_template_id}")
            except Exception as e:
                print(f"\n⚠️  Cleanup error: {str(e)}")
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        print_section("Meta Task Template Default Settings Test Suite")
        print(f"Testing API at: {API_BASE}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Test sequence
        tests = [
            ("API Health Check", self.test_api_health),
            ("Create Template with Settings", self.test_create_template_with_settings),
            ("Retrieve Template Settings", self.test_retrieve_template_settings), 
            ("Update Template Settings", self.test_update_template_settings),
            ("Retrieve Updated Settings", self.test_retrieve_updated_settings),
            ("Complex Nested Settings", self.test_complex_nested_settings)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            if self.run_test(test_name, test_func):
                passed += 1
        
        # Print summary
        print_section("Test Results Summary")
        
        for test_name, result, message in self.test_results:
            status = "✅" if result else "❌"
            print(f"{status} {test_name}")
            if message:
                print(f"    {message}")
        
        print(f"\nOverall Result: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED!")
            return True
        else:
            print(f"❌ {total - passed} tests failed")
            return False

def main():
    """Main test runner"""
    print("Starting Meta Task Template Default Settings Test")
    
    test_suite = MetaTaskTemplateTest()
    
    try:
        success = test_suite.run_all_tests()
        test_suite.cleanup()
        
        print_section("Test Completion")
        if success:
            print("✅ All tests completed successfully!")
            print("The Meta Task template system correctly saves and loads default_settings configuration.")
            sys.exit(0)
        else:
            print("❌ Some tests failed!")
            print("There may be issues with the Meta Task template default_settings functionality.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        test_suite.cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        test_suite.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()