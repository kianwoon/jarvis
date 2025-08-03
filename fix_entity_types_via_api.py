#!/usr/bin/env python3
"""
Script to fix existing entity types via API calls
"""

import requests
import json
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fixed_entity_types():
    """Test the entity type fixes via API"""
    
    logger.info("🔧 Testing fixed entity types via API")
    
    # Check current stats
    try:
        logger.info("📊 Getting current knowledge graph stats...")
        response = requests.get("http://localhost:8000/api/v1/knowledge-graph/stats")
        
        if response.status_code == 200:
            stats = response.json()
            logger.info(f"Current stats:")
            logger.info(f"  Total entities: {stats.get('total_entities', 0)}")
            logger.info(f"  Total relationships: {stats.get('total_relationships', 0)}")
            logger.info(f"  Entity types: {stats.get('entity_types', {})}")
            
            entity_types = stats.get('entity_types', {})
            unique_types = len(entity_types)
            
            if unique_types == 1 and 'CONCEPT' in entity_types:
                logger.warning("⚠️  All entities are still CONCEPT type")
                logger.info("🔄 The new classification will apply to newly processed documents")
                logger.info("💡 To see the fix in action, reprocess the DBS document")
            else:
                logger.info(f"✅ Found {unique_types} entity types - diversity achieved!")
                
        else:
            logger.error(f"❌ Failed to get stats: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ API call failed: {e}")
    
    # Test the debug endpoint
    try:
        logger.info("\n🔍 Testing debug endpoint...")
        response = requests.get("http://localhost:8000/api/v1/knowledge-graph/debug/entity-types")
        
        if response.status_code == 200:
            debug_data = response.json()
            logger.info("Debug endpoint working ✅")
            
            type_dist = debug_data.get('entity_type_distribution', [])
            logger.info("📊 Type distribution from debug:")
            for item in type_dist:
                logger.info(f"  {item.get('entity_type', 'Unknown')}: {item.get('count', 0)} entities")
                
        else:
            logger.error(f"❌ Debug endpoint failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ Debug API call failed: {e}")

def recommend_next_steps():
    """Provide recommendations for the user"""
    logger.info("\n🎯 Next Steps to See the Fix:")
    logger.info("1. Reprocess the DBS document to see new entity type classification")
    logger.info("2. Check the knowledge graph visualization - nodes should now show different colors")
    logger.info("3. Run anti-silo analysis to reduce isolated nodes")
    logger.info("4. Use the debug endpoint to verify entity types: GET /api/v1/knowledge-graph/debug/entity-types")

if __name__ == "__main__":
    test_fixed_entity_types()
    recommend_next_steps()