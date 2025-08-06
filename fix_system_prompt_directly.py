#!/usr/bin/env python3
"""
Direct update to the LLM system prompt in the database
"""

import logging
from app.core.db import SessionLocal, Settings as SettingsModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_system_prompt_directly():
    """Update system prompt directly in database"""
    
    try:
        db = SessionLocal()
        
        # Get current LLM settings
        settings_row = db.query(SettingsModel).filter(SettingsModel.category == 'llm').first()
        
        if not settings_row:
            print("❌ No LLM settings found")
            return False
        
        current_settings = dict(settings_row.settings)
        current_prompt = current_settings.get('main_llm', {}).get('system_prompt', '')
        
        print(f"📋 Current prompt: {current_prompt[:100]}...")
        print(f"📏 Current length: {len(current_prompt)}")
        
        # Add tool usage guidelines
        tool_guidelines = """

IMPORTANT TOOL USAGE GUIDELINES:
- When using tools, only include parameters that are specifically needed
- Do not hardcode parameter values unless required for your specific use case  
- For RAG searches, omit max_documents parameter to use the optimal default value
- Only specify max_documents if you need more/fewer results than the default"""
        
        # Update the prompt
        new_prompt = current_prompt + tool_guidelines
        current_settings['main_llm']['system_prompt'] = new_prompt
        
        # Update in database
        settings_row.settings = current_settings
        db.commit()
        
        print(f"✅ Updated system prompt")
        print(f"📏 New length: {len(new_prompt)}")
        
        # Verify the update
        db.refresh(settings_row)
        updated_prompt = settings_row.settings['main_llm']['system_prompt']
        
        if "IMPORTANT TOOL USAGE GUIDELINES" in updated_prompt:
            print("✅ Verification: Tool guidelines found in database")
            return True
        else:
            print("❌ Verification: Tool guidelines not found in database") 
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        if 'db' in locals():
            db.rollback()
        return False
    finally:
        if 'db' in locals():
            db.close()

def main():
    success = update_system_prompt_directly()
    return success

if __name__ == "__main__":
    result = main()
    exit(0 if result else 1)