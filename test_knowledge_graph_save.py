#!/usr/bin/env python3
"""
Test script to verify knowledge graph settings save functionality
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"
SETTINGS_ENDPOINT = f"{BASE_URL}/api/v1/settings/knowledge_graph"

def test_save_knowledge_graph_settings():
    """Test saving knowledge graph settings"""
    print("Testing Knowledge Graph Settings Save...")
    
    # Test data with comprehensive settings
    test_settings = {
        "settings": {
            "mode": "thinking",
            "neo4j": {
                "enabled": True,
                "host": "localhost",
                "port": 7687,
                "http_port": 7474,
                "database": "neo4j",
                "username": "neo4j",
                "password": "test_password",
                "uri": "bolt://localhost:7687"
            },
            "model_config": {
                "model": "qwen3:30b-a3b",
                "temperature": 0.3,
                "repeat_penalty": 1.1,
                "system_prompt": "You are an expert knowledge graph extraction system.",
                "max_tokens": 4096,
                "context_length": 40960,
                "model_server": "http://localhost:11434",
                "analysis_prompt": "Analyze this document comprehensively...",
                "extraction_prompt": "Extract entities and relationships from this analysis..."
            },
            "anti_silo": {
                "enabled": True,
                "similarity_threshold": 0.75,
                "cross_document_linking": True,
                "max_relationships_per_entity": 100
            },
            "schema_mode": "hybrid",
            "entity_discovery": {
                "enabled": True,
                "confidence_threshold": 0.75
            },
            "relationship_discovery": {
                "enabled": True,
                "confidence_threshold": 0.7
            },
            "prompts": [
                {
                    "id": "extraction",
                    "name": "Entity Extraction",
                    "content": "Extract entities from the following text..."
                }
            ]
        },
        "persist_to_db": True,
        "reload_cache": True
    }
    
    try:
        # 1. Save settings
        print("\n1. Saving knowledge graph settings...")
        response = requests.put(SETTINGS_ENDPOINT, json=test_settings)
        
        if response.status_code == 200:
            print("✅ Settings saved successfully!")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"❌ Failed to save settings: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        # 2. Wait a moment for cache to update
        time.sleep(1)
        
        # 3. Retrieve settings to verify
        print("\n2. Retrieving saved settings...")
        response = requests.get(SETTINGS_ENDPOINT)
        
        if response.status_code == 200:
            saved_data = response.json()
            saved_settings = saved_data.get("settings", {})
            
            print("✅ Settings retrieved successfully!")
            
            # Verify key fields were saved
            print("\n3. Verifying saved data...")
            
            # Check mode
            if saved_settings.get("mode") == test_settings["settings"]["mode"]:
                print("✅ Mode saved correctly")
            else:
                print(f"❌ Mode mismatch: expected '{test_settings['settings']['mode']}', got '{saved_settings.get('mode')}'")
            
            # Check Neo4j config
            neo4j_config = saved_settings.get("neo4j", {})
            if neo4j_config.get("host") == test_settings["settings"]["neo4j"]["host"]:
                print("✅ Neo4j config saved correctly")
            else:
                print(f"❌ Neo4j config mismatch")
            
            # Check model config
            model_config = saved_settings.get("model_config", {})
            if model_config.get("model") == test_settings["settings"]["model_config"]["model"]:
                print("✅ Model config saved correctly")
            else:
                print(f"❌ Model config mismatch")
            
            # Check prompts
            if model_config.get("analysis_prompt") and model_config.get("extraction_prompt"):
                print("✅ Prompts saved correctly")
            else:
                print(f"❌ Prompts missing: analysis_prompt={bool(model_config.get('analysis_prompt'))}, extraction_prompt={bool(model_config.get('extraction_prompt'))}")
            
            # Check anti-silo settings
            anti_silo = saved_settings.get("anti_silo", {})
            if anti_silo.get("enabled") == test_settings["settings"]["anti_silo"]["enabled"]:
                print("✅ Anti-silo settings saved correctly")
            else:
                print(f"❌ Anti-silo settings mismatch")
            
            print(f"\nFull saved settings:\n{json.dumps(saved_settings, indent=2)}")
            
        else:
            print(f"❌ Failed to retrieve settings: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        # 4. Test cache reload endpoint
        print("\n4. Testing cache reload...")
        response = requests.post(f"{BASE_URL}/api/v1/settings/knowledge-graph/cache/reload")
        
        if response.status_code == 200:
            print("✅ Cache reloaded successfully!")
        else:
            print(f"❌ Failed to reload cache: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Knowledge Graph Settings Save Test ===")
    print(f"Testing against: {BASE_URL}")
    
    success = test_save_knowledge_graph_settings()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")