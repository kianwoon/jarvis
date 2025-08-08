#!/usr/bin/env python3
"""
Clear the problematic conversation that contains wrong ChatGPT-5 information
"""

import asyncio
import sys
import os

# Add the jarvis directory to Python path
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

from app.core.simple_conversation_manager import conversation_manager

async def clear_conversation():
    """Clear the conversation with wrong ChatGPT-5 info"""
    
    conversation_id = "chat-standard-chat-81c2d3d6-818d-4c99-b6fe-307d919bce9b"
    
    print(f"🧹 Clearing conversation: {conversation_id}")
    
    try:
        # Clear the conversation
        await conversation_manager.clear_conversation(conversation_id)
        print(f"✅ Successfully cleared conversation {conversation_id}")
        
        # Verify it's cleared by trying to retrieve history
        history = await conversation_manager.get_conversation_history(conversation_id)
        if not history:
            print("✅ Verification: Conversation history is now empty")
        else:
            print(f"⚠️  Warning: Still found {len(history)} messages after clearing")
            
    except Exception as e:
        print(f"❌ Error clearing conversation: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("🔧 ChatGPT-5 Conversation History Clear Script")
    print("=" * 50)
    
    # Run the clear operation
    success = asyncio.run(clear_conversation())
    
    if success:
        print("\n🎉 SUCCESS: Conversation cleared!")
        print("💡 Now test your ChatGPT-5 query again - it should work correctly without")
        print("   contradictory conversation history interfering with the search results.")
    else:
        print("\n❌ FAILED: Could not clear conversation")