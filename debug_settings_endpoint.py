#!/usr/bin/env python3

import requests
import json

def test_settings_endpoint():
    """Test the settings endpoint directly"""
    
    url = "http://localhost:8000/api/v1/settings/knowledge_graph"
    
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response Data:")
            print(json.dumps(data, indent=2))
            
            # Check specific paths
            settings = data.get('settings', {})
            model_config = settings.get('model_config', {})
            model = model_config.get('model', 'NOT FOUND')
            
            print(f"\nParsed paths:")
            print(f"data.settings exists: {bool(settings)}")
            print(f"data.settings.model_config exists: {bool(model_config)}")
            print(f"data.settings.model_config.model: '{model}'")
            
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_settings_endpoint()