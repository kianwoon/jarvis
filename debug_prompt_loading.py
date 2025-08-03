#!/usr/bin/env python3
"""Debug why enhanced prompt is not loading"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_prompt_loading():
    """Debug the prompt loading process"""
    
    print("🔍 Debugging Enhanced Prompt Loading")
    print("=" * 60)
    
    try:
        # Check database directly
        import psycopg2
        import json
        
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="llm_platform", 
            user="postgres",
            password="postgres"
        )
        
        cur = conn.cursor()
        cur.execute("SELECT settings FROM settings WHERE category = 'knowledge_graph'")
        result = cur.fetchone()
        
        if result:
            settings = result[0]
            system_prompt = settings.get('model_config', {}).get('system_prompt', '')
            
            print(f"📊 Database Check:")
            print(f"   System prompt length: {len(system_prompt)} characters")
            print(f"   Contains {{text}}: {'✅' if '{text}' in system_prompt else '❌'}")
            print(f"   Contains examples: {'✅' if 'Example 1' in system_prompt else '❌'}")
            
            if '{text}' in system_prompt:
                print(f"   First 200 chars: {system_prompt[:200]}...")
            else:
                print(f"   ❌ Missing {{text}} placeholder!")
        
        conn.close()
        
        # Check settings cache
        print(f"\n🔧 Settings Cache Check:")
        from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings
        
        kg_settings = get_knowledge_graph_settings()
        cached_prompt = kg_settings.get('model_config', {}).get('system_prompt', '')
        
        print(f"   Cached prompt length: {len(cached_prompt)} characters")
        print(f"   Cache contains {{text}}: {'✅' if '{text}' in cached_prompt else '❌'}")
        print(f"   Cache contains examples: {'✅' if 'Example 1' in cached_prompt else '❌'}")
        
        if len(cached_prompt) != len(system_prompt):
            print(f"   ⚠️  Cache/DB mismatch: {len(cached_prompt)} vs {len(system_prompt)} chars")
            
        # Test the actual prompt building process
        print(f"\n🧪 Prompt Building Test:")
        from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor
        
        extractor = LLMKnowledgeExtractor()
        test_text = "DBS Bank uses OceanBase database."
        
        try:
            prompt = extractor._build_extraction_prompt(test_text)
            print(f"   Generated prompt length: {len(prompt)} characters")
            print(f"   Contains examples: {'✅' if 'Example 1' in prompt else '❌'}")
            print(f"   Contains test text: {'✅' if 'DBS Bank' in prompt else '❌'}")
            
            if 'Example 1' not in prompt:
                print(f"   ❌ Enhanced prompt not being used!")
                print(f"   First 300 chars: {prompt[:300]}...")
            else:
                print(f"   ✅ Enhanced prompt is being used correctly")
                
        except Exception as e:
            print(f"   ❌ Error building prompt: {e}")
            import traceback
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_prompt_loading()