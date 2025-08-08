"""
Conflict Prevention System Example
Demonstrates how to use the conflict prevention functionality
"""

import asyncio
from app.core.simple_conversation_manager import conversation_manager
from app.core.conflict_prevention_engine import conflict_prevention_engine
from app.langchain.conflict_prevention_integration import (
    enhanced_synthesis_with_prevention,
    get_system_conflict_statistics,
    cleanup_historical_conflicts
)

async def example_conflict_prevention():
    """Example demonstrating conflict prevention in action"""
    
    print("=" * 60)
    print("CONFLICT PREVENTION SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    conversation_id = "example_conversation_123"
    
    # Scenario 1: Add initial message claiming something doesn't exist
    print("\n1. Adding initial message about GPT-5...")
    await conversation_manager.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content="OpenAI has not released GPT-5. The latest model is GPT-4.",
        metadata={"source": "initial_response"}
    )
    
    # Scenario 2: Try to add conflicting message (will be checked for conflicts)
    print("\n2. Attempting to add conflicting message...")
    await conversation_manager.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content="OpenAI just announced GPT-5 is now available with amazing new features!",
        metadata={"source": "updated_response"}
    )
    
    # Scenario 3: Check conversation statistics
    print("\n3. Getting conflict statistics...")
    stats = await conversation_manager.get_conflict_statistics(conversation_id)
    print(f"   Blocked messages: {stats.get('blocked_messages', 0)}")
    print(f"   Messages with conflicts: {stats.get('messages_with_conflicts', 0)}")
    print(f"   Total conflicts: {stats.get('total_conflicts', 0)}")
    print(f"   Conflict types: {stats.get('conflict_types', {})}")
    
    # Scenario 4: Use enhanced synthesis with prevention
    print("\n4. Running enhanced synthesis with conflict prevention...")
    synthesis_result = await enhanced_synthesis_with_prevention(
        question="What is the latest GPT model?",
        query_type="LLM",
        tool_context="Search results show GPT-5 has been released today with multimodal capabilities.",
        conversation_history="Previously discussed that GPT-4 was the latest model.",
        conversation_id=conversation_id,
        enable_prevention=True
    )
    
    print(f"   Conflicts detected: {synthesis_result.get('conflicts_detected', False)}")
    print(f"   Prevention enabled: {synthesis_result.get('conflict_prevention_enabled', False)}")
    if synthesis_result.get('conflict_report'):
        print(f"   Conflict report: {synthesis_result['conflict_report'][:200]}...")
    
    # Scenario 5: Test version conflict detection
    print("\n5. Testing version conflict detection...")
    conflict_check = await conflict_prevention_engine.check_for_conflicts(
        new_content="The latest version is 2.5 with improved performance.",
        conversation_id=conversation_id,
        role="assistant"
    )
    
    print(f"   Has conflicts: {conflict_check['has_conflicts']}")
    print(f"   Volatility score: {conflict_check['volatility_score']:.2f}")
    print(f"   Recommended TTL: {conflict_check['recommended_ttl']} seconds")
    print(f"   Should add: {conflict_check['should_add']}")
    
    # Scenario 6: Get system-wide statistics
    print("\n6. Getting system-wide conflict statistics...")
    system_stats = await get_system_conflict_statistics()
    print(f"   Status: {system_stats.get('status')}")
    print(f"   Total historical conflicts: {system_stats.get('total_historical_conflicts', 0)}")
    print(f"   Pattern frequencies: {system_stats.get('pattern_frequencies', {})}")
    
    # Scenario 7: Cleanup old conflicts
    print("\n7. Cleaning up old conflict history...")
    cleanup_stats = await cleanup_historical_conflicts(days_to_keep=7)
    print(f"   Cleaned entries: {cleanup_stats}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)

async def example_ttl_adjustment():
    """Example showing dynamic TTL adjustment based on content volatility"""
    
    print("\n" + "=" * 60)
    print("DYNAMIC TTL ADJUSTMENT DEMONSTRATION")
    print("=" * 60)
    
    conversation_id = "ttl_example_456"
    
    # Test different types of content with varying volatility
    test_cases = [
        {
            "content": "The Earth orbits around the Sun.",
            "description": "Low volatility (scientific fact)"
        },
        {
            "content": "The latest iPhone model is iPhone 15 Pro.",
            "description": "High volatility (version information)"
        },
        {
            "content": "Today's stock price for AAPL is $195.50.",
            "description": "Very high volatility (temporal/statistical)"
        },
        {
            "content": "Python is a programming language.",
            "description": "Low volatility (general fact)"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['description']}")
        print(f"   Content: {test['content']}")
        
        # Check for conflicts and get TTL recommendation
        conflict_check = await conflict_prevention_engine.check_for_conflicts(
            new_content=test['content'],
            conversation_id=conversation_id,
            role="assistant"
        )
        
        volatility = conflict_check['volatility_score']
        ttl = conflict_check['recommended_ttl']
        ttl_hours = ttl / 3600
        
        print(f"   Volatility score: {volatility:.2f}")
        print(f"   Recommended TTL: {ttl_hours:.1f} hours")
        
        # Add the message with dynamic TTL
        await conversation_manager.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=test['content'],
            metadata={"example_type": test['description']}
        )
    
    print("\n" + "=" * 60)
    print("TTL ADJUSTMENT DEMONSTRATION COMPLETE")
    print("=" * 60)

async def main():
    """Run all examples"""
    
    print("\n🚀 Starting Conflict Prevention System Examples\n")
    
    try:
        # Run conflict prevention example
        await example_conflict_prevention()
        
        # Run TTL adjustment example
        await example_ttl_adjustment()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")

if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())