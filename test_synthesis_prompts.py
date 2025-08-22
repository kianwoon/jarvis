#!/usr/bin/env python3
"""
Test script for synthesis prompt management system
This script tests the basic functionality of the synthesis prompt API endpoints
"""

import requests
import json
import sys

# Configuration
BASE_URL = "http://localhost:8000/api/v1/settings"
TEST_CATEGORY = "synthesis_prompts"
TEST_TEMPLATE = "test_template"

def test_synthesis_prompt_system():
    """Test the synthesis prompt management system"""
    print("🧪 Testing Synthesis Prompt Management System")
    print("=" * 50)
    
    # Test 1: Get synthesis prompts status
    print("\n1. Testing status endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/synthesis-prompts/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {data['status']}")
            print(f"   Total categories: {data['statistics']['total_categories']}")
            print(f"   Total templates: {data['statistics']['total_templates']}")
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Status check error: {e}")
        return False
    
    # Test 2: Get default templates
    print("\n2. Testing template retrieval...")
    try:
        response = requests.get(f"{BASE_URL}/synthesis-prompts/{TEST_CATEGORY}/templates")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Retrieved {data['count']} templates from {TEST_CATEGORY}")
            template_names = list(data['templates'].keys())
            print(f"   Templates: {', '.join(template_names[:3])}...")
        else:
            print(f"❌ Template retrieval failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Template retrieval error: {e}")
        return False
    
    # Test 3: Create a new template
    print("\n3. Testing template creation...")
    test_template_data = {
        "content": "This is a test template with {variable1} and {variable2}.",
        "description": "Test template for API validation",
        "variables": ["variable1", "variable2"],
        "active": True
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/synthesis-prompts/{TEST_CATEGORY}/templates/{TEST_TEMPLATE}",
            json=test_template_data
        )
        if response.status_code == 200:
            print(f"✅ Created test template: {TEST_TEMPLATE}")
        else:
            print(f"❌ Template creation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Template creation error: {e}")
        return False
    
    # Test 4: Preview the template
    print("\n4. Testing template preview...")
    preview_data = {
        "sample_variables": {
            "variable1": "sample_value_1",
            "variable2": "sample_value_2"
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/synthesis-prompts/{TEST_CATEGORY}/templates/{TEST_TEMPLATE}/preview",
            json=preview_data
        )
        if response.status_code == 200:
            data = response.json()
            preview = data['preview']
            print(f"✅ Template preview generated")
            print(f"   Original: {preview['original_content']}")
            print(f"   Rendered: {preview['rendered_content']}")
        else:
            print(f"❌ Template preview failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Template preview error: {e}")
        return False
    
    # Test 5: Update the template
    print("\n5. Testing template update...")
    update_data = {
        "description": "Updated test template description",
        "active": False
    }
    
    try:
        response = requests.put(
            f"{BASE_URL}/synthesis-prompts/{TEST_CATEGORY}/templates/{TEST_TEMPLATE}",
            json=update_data
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Template updated: {', '.join(data['updated_fields'])}")
        else:
            print(f"❌ Template update failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Template update error: {e}")
        return False
    
    # Test 6: Validate all settings
    print("\n6. Testing validation...")
    try:
        response = requests.post(f"{BASE_URL}/synthesis-prompts/validate")
        if response.status_code == 200:
            data = response.json()
            if data['valid']:
                print(f"✅ Validation passed: {data['message']}")
            else:
                print(f"⚠️  Validation warning: {data['message']}")
        else:
            print(f"❌ Validation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False
    
    # Test 7: Clean up - delete test template
    print("\n7. Testing template deletion...")
    try:
        response = requests.delete(
            f"{BASE_URL}/synthesis-prompts/{TEST_CATEGORY}/templates/{TEST_TEMPLATE}"
        )
        if response.status_code == 200:
            print(f"✅ Test template deleted successfully")
        else:
            print(f"❌ Template deletion failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Template deletion error: {e}")
        return False
    
    print("\n🎉 All tests passed! Synthesis prompt management system is working correctly.")
    return True

def test_standard_settings_integration():
    """Test integration with standard settings endpoints"""
    print("\n🔗 Testing Standard Settings Integration")
    print("=" * 40)
    
    # Test getting synthesis_prompts via standard settings endpoint
    for category in ['synthesis_prompts', 'formatting_templates', 'system_behaviors']:
        print(f"\nTesting standard GET /{category}...")
        try:
            response = requests.get(f"{BASE_URL}/{category}")
            if response.status_code == 200:
                data = response.json()
                template_count = len(data['settings'])
                print(f"✅ {category}: {template_count} templates")
            else:
                print(f"❌ Failed to get {category}: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error getting {category}: {e}")
            return False
    
    print("\n✅ Standard settings integration working correctly.")
    return True

if __name__ == "__main__":
    print("🚀 Starting Synthesis Prompt API Tests")
    print("Make sure the Jarvis server is running on localhost:8000")
    
    # Test the system
    success = test_synthesis_prompt_system()
    if success:
        success = test_standard_settings_integration()
    
    if success:
        print("\n🎊 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Check the error messages above.")
        sys.exit(1)