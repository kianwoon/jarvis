#!/usr/bin/env python3
"""
Test script to verify IDC model configuration fix
Tests that model names are being saved correctly, not IDs
"""

import httpx
import json
import asyncio

async def test_idc_model_fix():
    """Test the IDC model configuration fix"""
    base_url = "http://localhost:8000"
    
    print("Testing IDC Model Configuration Fix")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Get current configuration
        print("\n1. Getting current IDC configuration...")
        response = await client.get(f"{base_url}/api/v1/idc/configuration")
        if response.status_code != 200:
            print(f"   ERROR: Failed to get configuration: {response.status_code}")
            return
        
        config_data = response.json()
        current_config = config_data.get("configuration", {})
        available_models = config_data.get("available_models", [])
        
        print(f"   Current extraction model: {current_config.get('extraction', {}).get('model')}")
        print(f"   Current validation model: {current_config.get('validation', {}).get('model')}")
        print(f"   Available models: {len(available_models)}")
        
        # Check if current models look like IDs (12-char hex strings)
        extraction_model = current_config.get('extraction', {}).get('model', '')
        validation_model = current_config.get('validation', {}).get('model', '')
        
        is_extraction_id = len(extraction_model) == 12 and all(c in '0123456789abcdef' for c in extraction_model.lower())
        is_validation_id = len(validation_model) == 12 and all(c in '0123456789abcdef' for c in validation_model.lower())
        
        if is_extraction_id or is_validation_id:
            print("\n   ⚠️  WARNING: Found model IDs instead of names!")
            if is_extraction_id:
                print(f"      Extraction model looks like ID: {extraction_model}")
            if is_validation_id:
                print(f"      Validation model looks like ID: {validation_model}")
            
            # Try to find the actual model names
            if available_models:
                print("\n   Attempting to map IDs to model names...")
                for model in available_models:
                    if model.get('id') == extraction_model:
                        print(f"      Found extraction model: {model.get('name')}")
                    if model.get('id') == validation_model:
                        print(f"      Found validation model: {model.get('name')}")
        else:
            print("\n   ✅ Models are using proper names (not IDs)")
        
        # Step 2: Test saving configuration with a model name
        print("\n2. Testing configuration save with proper model name...")
        
        # Pick the first available model or use a default
        test_model = available_models[0]['name'] if available_models else "qwen3:30b-a3b-q4_K_M"
        print(f"   Using test model: {test_model}")
        
        update_payload = {
            "extraction_model": test_model,
            "validation_model": test_model,
            "extraction_temperature": 0.3,
            "validation_temperature": 0.3
        }
        
        response = await client.post(
            f"{base_url}/api/v1/idc/configuration",
            json=update_payload
        )
        
        if response.status_code != 200:
            print(f"   ERROR: Failed to update configuration: {response.status_code}")
            print(f"   Response: {response.text}")
            return
        
        print("   ✅ Configuration saved successfully")
        
        # Step 3: Verify the saved configuration
        print("\n3. Verifying saved configuration...")
        response = await client.get(f"{base_url}/api/v1/idc/configuration")
        if response.status_code != 200:
            print(f"   ERROR: Failed to get configuration: {response.status_code}")
            return
        
        updated_config = response.json().get("configuration", {})
        saved_extraction = updated_config.get('extraction', {}).get('model')
        saved_validation = updated_config.get('validation', {}).get('model')
        
        print(f"   Saved extraction model: {saved_extraction}")
        print(f"   Saved validation model: {saved_validation}")
        
        # Check if saved values are model names (not IDs)
        is_saved_extraction_id = len(saved_extraction) == 12 and all(c in '0123456789abcdef' for c in saved_extraction.lower())
        is_saved_validation_id = len(saved_validation) == 12 and all(c in '0123456789abcdef' for c in saved_validation.lower())
        
        if is_saved_extraction_id or is_saved_validation_id:
            print("\n   ❌ FAILED: Still saving model IDs instead of names!")
            print("      Frontend fix may not be applied yet.")
        else:
            print("\n   ✅ SUCCESS: Models are being saved with proper names!")
        
        # Step 4: Check if Ollama can use the model
        print("\n4. Testing if Ollama recognizes the model...")
        try:
            ollama_url = "http://host.docker.internal:11434"
            test_prompt = "Say 'hello' in one word."
            
            response = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": saved_extraction,
                    "prompt": test_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 10
                    }
                },
                timeout=15.0
            )
            
            if response.status_code == 200:
                print(f"   ✅ Ollama successfully used model: {saved_extraction}")
                result = response.json()
                print(f"      Response: {result.get('response', '').strip()[:50]}")
            else:
                print(f"   ❌ Ollama failed to use model: {response.status_code}")
                print(f"      This likely means the model name is incorrect")
                
        except httpx.ConnectError:
            print("   ⚠️  Could not connect to Ollama (may not be running)")
        except httpx.TimeoutException:
            print("   ⚠️  Ollama request timed out")
        except Exception as e:
            print(f"   ⚠️  Error testing Ollama: {e}")
    
    print("\n" + "=" * 50)
    print("Test complete!")
    print("\nSummary:")
    print("- Frontend should use model.name, not model.id as Select value")
    print("- Backend generates IDs from digest[:12] for display only")
    print("- Database should store full model names like 'qwen3:30b-a3b-q4_K_M'")
    print("- The fix ensures Ollama receives proper model names, not IDs")

if __name__ == "__main__":
    asyncio.run(test_idc_model_fix())