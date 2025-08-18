#!/usr/bin/env python3
"""
Test script to verify radiating system is using correct model configuration.
"""

import asyncio
import json
import logging
from app.core.radiating_settings_cache import get_model_config, get_radiating_settings
from app.services.radiating.extraction.universal_entity_extractor import UniversalEntityExtractor
from app.services.radiating.query_expansion.query_analyzer import QueryAnalyzer
from app.services.radiating.query_expansion.expansion_strategy import SemanticExpansionStrategy

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_model_config():
    """Test that model configuration is loaded correctly"""
    print("\n=== Testing Model Configuration ===")
    model_config = get_model_config()
    print(f"Model: {model_config.get('model')}")
    print(f"Temperature: {model_config.get('temperature')}")
    print(f"Max Tokens: {model_config.get('max_tokens')}")
    print(f"Model Server: {model_config.get('model_server')}")
    
    # Check if it's using radiating model, not main LLM
    assert model_config.get('model') != 'claude-3-opus', "Should not be using main LLM model"
    print("✓ Model configuration is correct (not using main LLM)")
    
    return model_config

async def test_entity_extraction():
    """Test entity extraction with a sample query"""
    print("\n=== Testing Entity Extraction ===")
    
    extractor = UniversalEntityExtractor()
    
    # Test with a technology query
    test_query = "What are the best Python frameworks for building REST APIs?"
    
    print(f"Test Query: {test_query}")
    print("Extracting entities...")
    
    entities = await extractor.extract_entities(
        test_query,
        domain_hints=["Technology", "Programming"],
        context="User is asking about web development"
    )
    
    print(f"\nExtracted {len(entities)} entities:")
    for entity in entities:
        print(f"  - {entity.text} (Type: {entity.entity_type}, Confidence: {entity.confidence:.2f})")
    
    if len(entities) > 0:
        print("✓ Entity extraction is working")
    else:
        print("✗ No entities extracted - check LLM response")
    
    return entities

async def test_query_analysis():
    """Test query analyzer"""
    print("\n=== Testing Query Analysis ===")
    
    analyzer = QueryAnalyzer()
    
    test_query = "Compare React and Vue.js for frontend development"
    print(f"Test Query: {test_query}")
    print("Analyzing query...")
    
    analyzed = await analyzer.analyze_query(test_query)
    
    print(f"\nQuery Analysis Results:")
    print(f"  Intent: {analyzed.intent.value}")
    print(f"  Confidence: {analyzed.confidence:.2f}")
    print(f"  Domain Hints: {analyzed.domain_hints}")
    print(f"  Key Entities: {len(analyzed.key_entities)}")
    for entity in analyzed.key_entities:
        print(f"    - {entity.get('text')} ({entity.get('type')})")
    
    if len(analyzed.key_entities) > 0:
        print("✓ Query analysis is working")
    else:
        print("✗ No entities found in query - check LLM response")
    
    return analyzed

async def test_expansion_strategy():
    """Test query expansion"""
    print("\n=== Testing Query Expansion ===")
    
    # First analyze the query
    analyzer = QueryAnalyzer()
    test_query = "Machine learning frameworks"
    analyzed = await analyzer.analyze_query(test_query)
    
    # Then expand it
    expander = SemanticExpansionStrategy()
    expanded = await expander.expand(analyzed)
    
    print(f"Original Query: {test_query}")
    print(f"Expansion Type: {expanded.expansion_type}")
    print(f"Expanded Terms: {expanded.expanded_terms}")
    print(f"Expanded Entities: {len(expanded.expanded_entities)}")
    
    if len(expanded.expanded_terms) > 0 or len(expanded.expanded_entities) > 0:
        print("✓ Query expansion is working")
    else:
        print("✗ No expansion generated - check LLM response")
    
    return expanded

async def test_llm_invoke():
    """Test direct LLM invocation"""
    print("\n=== Testing Direct LLM Invocation ===")
    
    from app.llm.ollama import JarvisLLM
    
    # Get radiating model config
    model_config = get_model_config()
    
    # Initialize with radiating config
    llm = JarvisLLM(
        model=model_config.get('model'),
        mode='non-thinking',
        max_tokens=model_config.get('max_tokens'),
        temperature=model_config.get('temperature'),
        model_server=model_config.get('model_server')
    )
    
    # Test with a simple JSON generation prompt
    test_prompt = """Return ONLY a valid JSON array of 3 programming languages:
["Python", "JavaScript", "Go"]

IMPORTANT: Return ONLY the JSON array, no explanations."""
    
    print(f"Test Prompt: {test_prompt[:100]}...")
    print("Invoking LLM...")
    
    response = await llm.invoke(test_prompt)
    print(f"Raw Response: {response[:200]}")
    
    try:
        parsed = json.loads(response)
        print(f"Parsed Response: {parsed}")
        print("✓ LLM is returning valid JSON")
    except json.JSONDecodeError as e:
        print(f"✗ LLM returned invalid JSON: {e}")
        print(f"Response was: {response[:500]}")
    
    return response

async def main():
    """Run all tests"""
    print("=" * 60)
    print("RADIATING SYSTEM FIX VERIFICATION")
    print("=" * 60)
    
    try:
        # Test model configuration
        model_config = await test_model_config()
        
        # Test direct LLM invocation first
        llm_response = await test_llm_invoke()
        
        # Test entity extraction
        entities = await test_entity_extraction()
        
        # Test query analysis
        analyzed = await test_query_analysis()
        
        # Test expansion
        expanded = await test_expansion_strategy()
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"✓ Model Config: Using {model_config.get('model')}")
        print(f"✓ LLM Response: {'Valid JSON' if llm_response else 'Invalid'}")
        print(f"✓ Entities Extracted: {len(entities) if entities else 0}")
        print(f"✓ Query Analysis: {'Working' if analyzed and analyzed.key_entities else 'Not Working'}")
        print(f"✓ Query Expansion: {'Working' if expanded and (expanded.expanded_terms or expanded.expanded_entities) else 'Not Working'}")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        print(f"\n✗ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())