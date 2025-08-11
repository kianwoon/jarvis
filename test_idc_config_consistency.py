#!/usr/bin/env python3
"""
Test script to verify IDC configuration consistency
Ensures extraction and validation both have max_context_usage and confidence_threshold
"""

import asyncio
import httpx
import json
from typing import Dict, Any

async def test_idc_configuration():
    """Test IDC configuration consistency"""
    
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        # Get current IDC configuration
        print("\n📊 Getting current IDC configuration...")
        response = await client.get(f"{base_url}/api/v1/idc/configuration")
        
        if response.status_code != 200:
            print(f"❌ Failed to get configuration: {response.status_code}")
            return
        
        config = response.json()
        idc_config = config.get('configuration', {})
        
        print("\n✅ Current IDC Configuration:")
        print(json.dumps(idc_config, indent=2))
        
        # Check extraction configuration
        extraction = idc_config.get('extraction', {})
        print("\n🔍 Extraction Configuration:")
        print(f"  - Model: {extraction.get('model', 'NOT SET')}")
        print(f"  - Max Context Usage: {extraction.get('max_context_usage', 'NOT SET')}")
        print(f"  - Confidence Threshold: {extraction.get('confidence_threshold', 'NOT SET')}")
        print(f"  - Temperature: {extraction.get('temperature', 'NOT SET')}")
        print(f"  - Max Tokens: {extraction.get('max_tokens', 'NOT SET')}")
        
        # Check validation configuration
        validation = idc_config.get('validation', {})
        print("\n🔍 Validation Configuration:")
        print(f"  - Model: {validation.get('model', 'NOT SET')}")
        print(f"  - Max Context Usage: {validation.get('max_context_usage', 'NOT SET')}")
        print(f"  - Confidence Threshold: {validation.get('confidence_threshold', 'NOT SET')}")
        print(f"  - Temperature: {validation.get('temperature', 'NOT SET')}")
        print(f"  - Max Tokens: {validation.get('max_tokens', 'NOT SET')}")
        
        # Test updating configuration
        print("\n📝 Testing configuration update...")
        update_payload = {
            "extraction_model": extraction.get('model', 'llama3.1:8b'),
            "extraction_max_context_usage": 0.40,  # 40%
            "extraction_confidence_threshold": 0.85,  # 85%
            "validation_model": validation.get('model', 'llama3.1:8b'),
            "max_context_usage": 0.40,  # 40% for validation
            "quality_threshold": 0.85  # 85% for validation
        }
        
        response = await client.post(
            f"{base_url}/api/v1/idc/configuration",
            json=update_payload
        )
        
        if response.status_code == 200:
            print("✅ Configuration updated successfully")
            
            # Get updated configuration
            response = await client.get(f"{base_url}/api/v1/idc/configuration")
            if response.status_code == 200:
                updated_config = response.json().get('configuration', {})
                
                print("\n📊 Updated Configuration:")
                extraction = updated_config.get('extraction', {})
                validation = updated_config.get('validation', {})
                
                print("\nExtraction:")
                print(f"  - Max Context Usage: {extraction.get('max_context_usage', 'NOT SET')}")
                print(f"  - Confidence Threshold: {extraction.get('confidence_threshold', 'NOT SET')}")
                
                print("\nValidation:")
                print(f"  - Max Context Usage: {validation.get('max_context_usage', 'NOT SET')}")
                print(f"  - Confidence Threshold: {validation.get('confidence_threshold', 'NOT SET')}")
                
                # Check consistency
                print("\n🔍 Consistency Check:")
                issues = []
                
                if 'max_context_usage' not in extraction:
                    issues.append("❌ Extraction missing max_context_usage")
                if 'confidence_threshold' not in extraction:
                    issues.append("❌ Extraction missing confidence_threshold")
                if 'max_context_usage' not in validation:
                    issues.append("❌ Validation missing max_context_usage")
                if 'confidence_threshold' not in validation:
                    issues.append("❌ Validation missing confidence_threshold")
                
                if issues:
                    for issue in issues:
                        print(issue)
                else:
                    print("✅ All required fields present in both extraction and validation")
                    print("✅ Configuration is architecturally consistent!")
        else:
            print(f"❌ Failed to update configuration: {response.status_code}")
            print(response.text)

if __name__ == "__main__":
    print("🚀 IDC Configuration Consistency Test")
    print("=" * 50)
    asyncio.run(test_idc_configuration())