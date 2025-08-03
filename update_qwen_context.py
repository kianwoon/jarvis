#!/usr/bin/env python3
"""
Emergency fix to update Qwen3:30B model context length to its ACTUAL capacity
"""

import asyncio
import httpx
import json

async def update_knowledge_graph_settings():
    """Update the knowledge graph settings with correct context length"""
    
    # Use the knowledge graph settings cache directly
    from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings, set_knowledge_graph_settings
    
    try:
        # Get current settings
        current_settings = get_knowledge_graph_settings()
        print(f"📊 Current settings loaded")
        
        # Update the context_length to the ACTUAL value for Qwen3:30B
        if 'model_config' not in current_settings:
            current_settings['model_config'] = {}
        
        # Set the ACTUAL context length for the Qwen model
        current_settings['model_config']['model'] = 'qwen3:30b-a3b-instruct-2507-q4_k_m'
        current_settings['model_config']['context_length'] = 262144  # 256k tokens!
        
        # Also update the top-level context_length
        current_settings['context_length'] = 262144
        
        print(f"✅ Setting context_length to 262,144 tokens (256k)")
        
        # Save updated settings
        set_knowledge_graph_settings(current_settings)
        
        print(f"✅ Successfully updated knowledge graph settings with correct context length!")
        print(f"📊 The model will now use its FULL 256k context capacity")
        
        # Force reload the cache
        async with httpx.AsyncClient() as client:
            reload_response = await client.post("http://localhost:8000/api/v1/settings/knowledge-graph/cache/reload")
            if reload_response.status_code == 200:
                print(f"✅ Cache reloaded successfully!")
            else:
                print(f"⚠️  Cache reload failed but settings are saved: {reload_response.text}")
                
    except Exception as e:
        print(f"❌ Error updating settings: {str(e)}")

if __name__ == "__main__":
    print("🚀 Updating Qwen3:30B context length to use FULL capacity...")
    asyncio.run(update_knowledge_graph_settings())