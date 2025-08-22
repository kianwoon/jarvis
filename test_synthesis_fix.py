#!/usr/bin/env python3
"""
Test script to verify the synthesis prompts API endpoint and data structure.
This helps validate the backend is working before testing frontend fixes.
"""

import requests
import json
import sys

def test_synthesis_prompts_api():
    """Test the synthesis prompts API endpoint"""
    try:
        # Test the API endpoint
        response = requests.get('http://localhost:8000/api/v1/settings/synthesis_prompts')
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return False
        
        data = response.json()
        print("✅ API Response received")
        
        # Validate data structure
        if not isinstance(data, dict):
            print("❌ Response is not a dictionary")
            return False
        
        if 'category' not in data:
            print("❌ Missing 'category' field")
            return False
        
        if 'settings' not in data:
            print("❌ Missing 'settings' field")
            return False
        
        if data['category'] != 'synthesis_prompts':
            print(f"❌ Wrong category: {data['category']}")
            return False
        
        settings = data['settings']
        if not isinstance(settings, dict):
            print("❌ Settings is not a dictionary")
            return False
        
        print(f"✅ Found {len(settings)} templates")
        
        # Validate each template structure
        for template_name, template in settings.items():
            if not isinstance(template, dict):
                print(f"❌ Template {template_name} is not a dictionary")
                return False
            
            required_fields = ['content', 'description', 'variables']
            for field in required_fields:
                if field not in template:
                    print(f"❌ Template {template_name} missing field: {field}")
                    return False
            
            # Check that variables is a list
            if not isinstance(template['variables'], list):
                print(f"❌ Template {template_name} variables is not a list")
                return False
            
            print(f"✅ Template {template_name}: {len(template['content'])} chars, {len(template['variables'])} vars")
        
        # Test frontend data transformation logic
        print("\n🔄 Testing frontend data transformation...")
        
        # Simulate what the frontend component does
        templates_array = []
        for key, template in settings.items():
            ui_template = {
                'id': key,
                'name': template.get('description', key.replace('_', ' ').title()),
                'content': template.get('content', ''),
                'variables': template.get('variables', []),
                'active': template.get('active', True),
                'version': template.get('version', '1.0'),
                'metadata': template.get('metadata', {})
            }
            templates_array.append(ui_template)
        
        print(f"✅ Frontend transformation successful: {len(templates_array)} templates")
        
        # Print sample template for verification
        if templates_array:
            sample = templates_array[0]
            print(f"📋 Sample template: {sample['name']}")
            print(f"   ID: {sample['id']}")
            print(f"   Content length: {len(sample['content'])}")
            print(f"   Variables: {sample['variables']}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the backend running on port 8000?")
        return False
    except json.JSONDecodeError:
        print("❌ Invalid JSON response from API")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing Synthesis Prompts API and Data Structure...")
    print("=" * 60)
    
    success = test_synthesis_prompts_api()
    
    print("=" * 60)
    if success:
        print("✅ All tests passed! The API and data structure are working correctly.")
        print("📌 If the frontend still shows 'No templates found', the issue is in:")
        print("   - Component data loading timing")
        print("   - React state management")
        print("   - Browser console errors")
        sys.exit(0)
    else:
        print("❌ Tests failed! Fix the API issues before debugging the frontend.")
        sys.exit(1)
