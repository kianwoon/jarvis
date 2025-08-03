#!/usr/bin/env python3
"""
Test Settings Save - Verify no hardcoded values are reintroduced
"""

import requests
import json
import sys

def test_settings_save():
    """Test that saving settings doesn't reintroduce hardcoded values"""
    print("🧪 Testing Knowledge Graph Settings Save...")
    
    # First get current settings
    print("1. Getting current settings...")
    response = requests.get("http://localhost:8000/api/v1/settings/knowledge_graph")
    if response.status_code != 200:
        print(f"❌ Failed to get settings: {response.status_code}")
        return False
    
    current_settings = response.json()["settings"]
    print("✅ Retrieved current settings")
    
    # Make a small change to trigger save
    print("2. Making small change to settings...")
    test_settings = current_settings.copy()
    test_settings["entity_discovery"]["confidence_threshold"] = 0.36  # Small change
    
    # Save the settings
    print("3. Saving modified settings...")
    save_response = requests.put(
        "http://localhost:8000/api/v1/settings/knowledge_graph",
        json={"settings": test_settings}
    )
    
    if save_response.status_code != 200:
        print(f"❌ Failed to save settings: {save_response.status_code}")
        print(f"Response: {save_response.text}")
        return False
    
    print("✅ Settings saved successfully")
    
    # Get settings again and verify no hardcoded values
    print("4. Verifying saved settings are clean...")
    verify_response = requests.get("http://localhost:8000/api/v1/settings/knowledge_graph")
    if verify_response.status_code != 200:
        print(f"❌ Failed to get settings after save: {verify_response.status_code}")
        return False
    
    saved_settings = verify_response.json()["settings"]
    
    # Check for hardcoded values
    issues = []
    
    if "extraction_settings" in saved_settings:
        issues.append("extraction_settings found in saved settings")
    
    if "prompts" in saved_settings:
        for prompt in saved_settings["prompts"]:
            if "parameters" in prompt and "types" in prompt["parameters"]:
                issues.append(f"hardcoded types found in prompt {prompt.get('name')}")
    
    if issues:
        print("❌ Issues found after save:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("✅ Saved settings are clean - no hardcoded values")
    print(f"✅ Confidence threshold successfully updated to: {saved_settings['entity_discovery']['confidence_threshold']}")
    
    return True

def main():
    print("🔬 Knowledge Graph Settings Save Test")
    print("=" * 40)
    
    try:
        success = test_settings_save()
        
        print("\n📊 Test Results:")
        print("=" * 20)
        if success:
            print("✅ PASS - Settings save/load cycle is clean")
            print("   No hardcoded values reintroduced")
            return 0
        else:
            print("❌ FAIL - Issues detected in save/load cycle")
            return 1
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return 1

if __name__ == "__main__":
    exit(main())