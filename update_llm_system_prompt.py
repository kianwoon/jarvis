#!/usr/bin/env python3
"""
Update the LLM system prompt to include tool usage guidelines
"""

import json
import logging
from app.core.db import SessionLocal, Settings as SettingsModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_llm_system_prompt():
    """Add tool usage guidelines to LLM system prompt"""
    
    try:
        db = SessionLocal()
        
        # Get current LLM settings
        settings_row = db.query(SettingsModel).filter(SettingsModel.category == 'llm').first()
        
        if not settings_row or not settings_row.settings:
            print("❌ No LLM settings found")
            return False
        
        current_settings = settings_row.settings
        print("✅ Found LLM settings")
        
        # Get current system prompt
        current_prompt = current_settings.get('main_llm', {}).get('system_prompt', '')
        
        if not current_prompt:
            print("❌ No main_llm system_prompt found")
            return False
        
        print(f"📋 Current prompt length: {len(current_prompt)} characters")
        
        # Check if tool guidelines already exist
        if "IMPORTANT TOOL USAGE GUIDELINES" in current_prompt:
            print("✅ Tool usage guidelines already present in system prompt")
            return True
        
        # Add tool usage guidelines
        tool_guidelines = """

IMPORTANT TOOL USAGE GUIDELINES:
- When using tools, only include parameters that are specifically needed
- Do not hardcode parameter values unless required for your specific use case  
- For RAG searches, omit max_documents parameter to use the optimal default value
- Only specify max_documents if you need more/fewer results than the default"""
        
        # Add the guidelines to the system prompt
        updated_prompt = current_prompt + tool_guidelines
        
        # Update the settings
        current_settings['main_llm']['system_prompt'] = updated_prompt
        
        # Save back to database
        settings_row.settings = current_settings
        db.commit()
        
        print("✅ Successfully updated LLM system prompt with tool usage guidelines")
        print(f"📋 New prompt length: {len(updated_prompt)} characters")
        print("🎯 Added guidelines about not hardcoding tool parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to update LLM system prompt: {e}")
        if 'db' in locals():
            db.rollback()
        return False
        
    finally:
        if 'db' in locals():
            db.close()

def main():
    """Main function"""
    print("🚀 Updating LLM system prompt with tool usage guidelines")
    print("=" * 60)
    
    success = update_llm_system_prompt()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 SUCCESS: LLM system prompt updated!")
        print("📝 The LLM will now have explicit instructions about tool parameter usage")
    else:
        print("❌ FAILURE: Could not update LLM system prompt")
    
    return success

if __name__ == "__main__":
    result = main()
    exit(0 if result else 1)