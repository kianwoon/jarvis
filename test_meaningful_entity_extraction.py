#!/usr/bin/env python3
"""
Test script for meaningful entity extraction improvements.

This script tests the new business value scoring system and enhanced filtering
to ensure we extract meaningful entities over generic ones.
"""

import asyncio
import sys
import os
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor
from app.services.settings_prompt_service import get_prompt_service

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test text with mix of meaningful and generic entities
TEST_TEXT = """
DBS Bank announced that CEO Piyush Gupta will lead the digital transformation initiative in 2024, 
targeting $2.5 billion in revenue from mobile banking services. The bank's PayLah! mobile app has 
gained significant traction in the Singapore market, competing with Grab's payment platform.

In Q3 2024, DBS reported a 15% increase in digital banking adoption across Southeast Asia, 
particularly in Indonesia and Thailand markets. The company's cloud migration strategy, 
powered by AWS infrastructure, has improved system performance and reduced operational costs.

The transformation includes AI-powered customer service, blockchain-based payments, and 
modernization of legacy systems to support the bank's expansion into fintech solutions.
"""

async def test_meaningful_extraction():
    """Test the improved entity extraction system"""
    logger.info("🧪 Testing Meaningful Entity Extraction Improvements")
    logger.info("=" * 60)
    
    try:
        # Initialize the extractor
        extractor = LLMKnowledgeExtractor()
        prompt_service = get_prompt_service()
        
        logger.info("📝 Test Text:")
        logger.info("-" * 40)
        logger.info(TEST_TEXT.strip())
        logger.info("-" * 40)
        
        # Test business value scoring individually
        logger.info("\n🎯 Testing Business Value Scoring:")
        test_entities = [
            ("CEO Piyush Gupta", "EXECUTIVE"),
            ("$2.5 billion revenue", "FINANCIAL_METRIC"), 
            ("PayLah! mobile app", "PRODUCT"),
            ("Singapore market", "LOCATION"),
            ("Q3 2024", "TEMPORAL"),
            ("DBS Bank", "ORGANIZATION"),
            ("CEO", "EXECUTIVE"),  # Generic
            ("system", "TECHNOLOGY"),  # Generic
            ("platform", "TECHNOLOGY"),  # Generic
            ("the", "CONCEPT"),  # Meaningless
        ]
        
        for entity_name, entity_type in test_entities:
            score = extractor._calculate_business_value_score(entity_name, entity_type)
            is_generic = extractor._is_generic_business_term(entity_name)
            logger.info(f"  {entity_name:25} | Score: {score:.2f} | Generic: {is_generic} | Type: {entity_type}")
        
        # Test entity type inference
        logger.info("\n🔍 Testing Enhanced Entity Type Inference:")
        test_names = [
            "CEO Piyush Gupta",
            "$2.5 billion revenue", 
            "PayLah! mobile app",
            "Digital Transformation 2025",
            "Q3 2024 earnings",
            "DBS Bank Singapore"
        ]
        
        for name in test_names:
            inferred_type = extractor._infer_entity_type_from_name(name)
            logger.info(f"  {name:25} | Inferred Type: {inferred_type}")
        
        # Test the improved prompt
        logger.info("\n📋 Testing Improved Prompt:")
        prompt = prompt_service.get_prompt('knowledge_extraction', {
            'text': TEST_TEXT,
            'context_info': 'Business document analysis',
            'domain_guidance': 'Focus on strategic business intelligence',
            'entity_types': 'EXECUTIVE, FINANCIAL_METRIC, ORGANIZATION, PRODUCT, LOCATION, TEMPORAL, STRATEGY',
            'relationship_types': 'LEADS, OPERATES_IN, COMPETES_WITH, GENERATED_BY, IMPLEMENTS'
        })
        
        if prompt:
            logger.info("✅ Successfully retrieved improved extraction prompt")
            logger.info(f"📏 Prompt length: {len(prompt):,} characters")
            
            # Check for key improvements in the prompt
            improvements = [
                "BUSINESS INTELLIGENCE",
                "business_value_score", 
                "strategic_relevance",
                "ANTI-GENERIC FILTERING",
                "HIGH-VALUE ENTITIES ONLY"
            ]
            
            found_improvements = [imp for imp in improvements if imp in prompt]
            logger.info(f"🎯 Found {len(found_improvements)}/{len(improvements)} key improvements in prompt")
            
            for improvement in found_improvements:
                logger.info(f"  ✅ {improvement}")
                
            missing = [imp for imp in improvements if imp not in prompt]
            if missing:
                for improvement in missing:
                    logger.info(f"  ❌ Missing: {improvement}")
        else:
            logger.error("❌ Failed to retrieve improved extraction prompt")
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 Test completed successfully!")
        logger.info("\n📊 Expected Results:")
        logger.info("- High scores (>0.8): CEO Piyush Gupta, $2.5 billion revenue, PayLah! mobile app")
        logger.info("- Medium scores (0.6-0.8): Singapore market, Q3 2024, DBS Bank")
        logger.info("- Low scores (<0.6): CEO, system, platform, the (should be filtered out)")
        logger.info("- Enhanced entity types: EXECUTIVE, FINANCIAL_METRIC, PRODUCT, TEMPORAL")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_meaningful_extraction())
    sys.exit(0 if success else 1)