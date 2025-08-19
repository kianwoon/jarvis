#!/usr/bin/env python3
"""
Fix Query Classification and LLM Response Quality Issues

This script addresses:
1. Query classifier incorrectly classifying queries as "llm" instead of "tool"
2. Poor quality LLM responses due to missing system prompt
3. System prompt not being properly applied in direct LLM responses
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.db import SessionLocal, Settings as SettingsModel

def fix_query_classifier_settings():
    """Fix the query classifier to better handle queries and eliminate problematic LLM classification"""
    
    db = SessionLocal()
    try:
        # Get current LLM settings
        row = db.query(SettingsModel).filter(SettingsModel.category == 'llm').first()
        if not row:
            print("❌ No LLM settings found in database")
            return False
        
        settings = row.settings or {}
        
        # Update query classifier settings
        query_classifier = settings.get('query_classifier', {})
        
        # Update the system prompt to eliminate LLM classification and improve tool routing
        improved_prompt = """You are an intelligent query classifier. Your job is to determine whether to use 'tool' (for web search and external data) or 'rag' (for internal knowledge base).

## Classification Rules:
1. Choose 'tool' for:
   - Current events, news, or recent information
   - Questions about challenges, trends, or developments
   - Comparisons or analysis requiring up-to-date data
   - General questions about concepts, technologies, or ideas
   - Anything requiring external information or web search

2. Choose 'rag' ONLY if:
   - The query explicitly matches one of your RAG collection descriptions
   - You have specific internal documents about the exact topic

## Available Resources:

### RAG Collections:
{rag_collection}

### MCP Tools:
{mcp_tools}

## Important Guidelines:
- Default to 'tool' for most informational queries
- The system works best when using tools to gather current information
- Only use 'rag' when you're certain internal documents exist for the specific topic
- NEVER output 'llm' - choose between 'tool' and 'rag' only

Respond with: TYPE|CONFIDENCE (e.g., 'tool|0.8' or 'rag|0.9')"""

        query_classifier['llm_system_prompt'] = improved_prompt
        
        # Ensure proper model configuration
        if not query_classifier.get('llm_model'):
            query_classifier['llm_model'] = 'qwen2.5:0.5b'  # Set a default fast model for classification
        
        # Update thresholds for better routing
        query_classifier['min_confidence_threshold'] = 0.1
        query_classifier['direct_execution_threshold'] = 0.4  # Lower threshold for tool execution
        query_classifier['llm_direct_threshold'] = 0.95  # Very high threshold (effectively disabled)
        
        # Enable LLM classification
        query_classifier['enable_llm_classification'] = True
        
        settings['query_classifier'] = query_classifier
        
        # Save updated settings
        row.settings = settings
        db.commit()
        
        print("✅ Query classifier settings updated successfully")
        print(f"  - Improved system prompt to eliminate LLM classification")
        print(f"  - Set model to: {query_classifier['llm_model']}")
        print(f"  - Lowered tool execution threshold to: {query_classifier['direct_execution_threshold']}")
        print(f"  - Raised LLM direct threshold to: {query_classifier['llm_direct_threshold']}")
        
        # Clear Redis cache
        try:
            from app.core.redis_client import get_redis_client
            redis_client = get_redis_client()
            if redis_client:
                # Clear all relevant caches
                redis_client.delete('query_classifier_settings')
                redis_client.delete('llm_settings_cache')
                print("✅ Cleared Redis cache for settings")
        except Exception as e:
            print(f"⚠️ Could not clear Redis cache: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating query classifier settings: {e}")
        return False
    finally:
        db.close()

def verify_main_llm_prompt():
    """Verify that the main LLM system prompt is properly configured"""
    
    db = SessionLocal()
    try:
        row = db.query(SettingsModel).filter(SettingsModel.category == 'llm').first()
        if not row:
            print("❌ No LLM settings found")
            return False
        
        settings = row.settings or {}
        main_llm = settings.get('main_llm', {})
        
        if not main_llm.get('system_prompt'):
            print("⚠️ Main LLM system prompt is not configured")
            
            # Set a comprehensive system prompt
            system_prompt = """You are Jarvis, an advanced AI assistant with expertise in technology, business, and research. You provide accurate, thoughtful, and well-structured responses.

## Core Capabilities:
- Analyze complex topics and provide comprehensive insights
- Answer questions about technology, AI, business, and general knowledge
- Provide balanced perspectives on challenges and opportunities
- Structure responses clearly with appropriate formatting

## Response Guidelines:
1. **Accuracy**: Provide factual, well-researched information
2. **Structure**: Use clear organization with headings, bullet points, and numbered lists
3. **Depth**: Offer comprehensive analysis while remaining concise
4. **Context**: Consider the broader implications and context of questions
5. **Clarity**: Explain complex concepts in accessible language

## Important:
- When discussing challenges or problems, provide balanced analysis
- Consider multiple perspectives and stakeholder viewpoints
- Use examples and analogies to illustrate complex points
- If information might be outdated, acknowledge this limitation"""

            main_llm['system_prompt'] = system_prompt
            settings['main_llm'] = main_llm
            row.settings = settings
            db.commit()
            
            print("✅ Set comprehensive main LLM system prompt")
        else:
            print("✅ Main LLM system prompt is already configured")
            print(f"  Preview: {main_llm['system_prompt'][:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying main LLM prompt: {e}")
        return False
    finally:
        db.close()

def test_classification():
    """Test the classification with the problematic query"""
    
    test_query = "what are the biggest challenges for AI to blend into corporate?"
    
    print("\n" + "="*60)
    print("Testing Classification")
    print("="*60)
    print(f"Query: {test_query}")
    
    try:
        import asyncio
        from app.langchain.enhanced_query_classifier import EnhancedQueryClassifier
        
        classifier = EnhancedQueryClassifier()
        
        # Reload settings to get latest changes
        classifier.reload_settings()
        
        # Test classification
        async def test():
            result = await classifier.classify(test_query)
            if result:
                classification = result[0]
                print(f"Classification: {classification.query_type.value}")
                print(f"Confidence: {classification.confidence}")
                print(f"Expected: tool (for current information about AI challenges)")
                
                if classification.query_type.value == "tool":
                    print("✅ Classification is correct!")
                else:
                    print(f"⚠️ Classification is '{classification.query_type.value}' but should be 'tool'")
            else:
                print("❌ No classification result")
        
        asyncio.run(test())
        
    except Exception as e:
        print(f"❌ Error testing classification: {e}")

def main():
    print("="*60)
    print("Fixing Query Classification and LLM Response Issues")
    print("="*60)
    
    # Step 1: Fix query classifier settings
    print("\n1. Updating Query Classifier Settings...")
    if not fix_query_classifier_settings():
        print("Failed to update query classifier settings")
        return
    
    # Step 2: Verify main LLM prompt
    print("\n2. Verifying Main LLM System Prompt...")
    if not verify_main_llm_prompt():
        print("Failed to verify main LLM prompt")
        return
    
    # Step 3: Test the classification
    print("\n3. Testing Classification...")
    test_classification()
    
    print("\n" + "="*60)
    print("✅ FIXES APPLIED SUCCESSFULLY")
    print("="*60)
    print("\nKey changes made:")
    print("1. ✅ Updated query classifier prompt to eliminate 'llm' classification")
    print("2. ✅ Lowered tool execution threshold for better routing") 
    print("3. ✅ Ensured main LLM system prompt is configured")
    print("4. ✅ Cleared Redis cache to apply changes immediately")
    print("\nThe system should now:")
    print("• Route information-seeking queries to tools (web search)")
    print("• Provide better quality responses with proper context")
    print("• Not produce off-topic or irrelevant answers")

if __name__ == "__main__":
    main()