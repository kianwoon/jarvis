#!/usr/bin/env python3
"""Test script to debug prompt formatting issues"""

import sys
import os
sys.path.append('/app')

from app.services.settings_prompt_service import get_prompt_service

def test_prompt_formatting():
    prompt_service = get_prompt_service()
    
    # Test getting the knowledge_extraction prompt
    variables = {
        'text': 'Sample analysis text',
        'context_info': '',
        'domain_guidance': '',
        'entity_types': 'PERSON, ORGANIZATION, TECHNOLOGY',
        'relationship_types': 'WORKS_FOR, EVALUATES, COMPETES_WITH'
    }
    
    try:
        prompt = prompt_service.get_prompt('knowledge_extraction', variables)
        print(f"✅ Prompt retrieved successfully")
        print(f"📏 Length: {len(prompt)} characters")
        print(f"🔤 First 500 chars: {prompt[:500]}")
        print(f"🔤 Last 200 chars: {prompt[-200:]}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"🔍 Error type: {type(e)}")
        
        # Try without variables
        try:
            prompt_no_vars = prompt_service.get_prompt('knowledge_extraction')
            print(f"✅ Prompt without variables: {len(prompt_no_vars)} chars")
        except Exception as e2:
            print(f"❌ Error even without variables: {e2}")

if __name__ == "__main__":
    test_prompt_formatting()