#!/usr/bin/env python3
"""
Debug script to test synthesis prompts UI data flow
"""
import requests
import json

def test_synthesis_prompts_api():
    """Test the synthesis prompts API endpoint"""
    try:
        print("🧪 Testing Synthesis Prompts API")
        print("=" * 50)
        
        # Test the API endpoint
        response = requests.get('http://localhost:8000/api/v1/settings/synthesis_prompts')
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: {response.status_code}")
            print(f"📊 Data structure: {list(data.keys())}")
            
            if 'settings' in data:
                settings = data['settings']
                print(f"📝 Templates found: {len(settings)}")
                
                for template_name, template_data in settings.items():
                    print(f"  📄 {template_name}:")
                    print(f"    Content length: {len(template_data.get('content', ''))}")
                    print(f"    Variables: {template_data.get('variables', [])}")
                    print(f"    Active: {template_data.get('active', False)}")
                
                # Show what the frontend component should receive
                print("\n🔧 Frontend Data Transformation:")
                ui_format = []
                for template_id, template in settings.items():
                    ui_item = {
                        'id': template_id,
                        'name': template.get('description', template_id.replace('_', ' ').title()),
                        'content': template.get('content', ''),
                        'variables': template.get('variables', []),
                        'active': template.get('active', True)
                    }
                    ui_format.append(ui_item)
                
                print(f"📋 UI should show {len(ui_format)} templates:")
                for item in ui_format:
                    print(f"  • {item['name']} ({len(item['content'])} chars)")
                
            else:
                print("❌ No 'settings' key in response")
                print(f"Response: {json.dumps(data, indent=2)[:500]}")
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Is the Jarvis server running on localhost:8000?")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_synthesis_prompts_api()