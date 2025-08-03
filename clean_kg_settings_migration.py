#!/usr/bin/env python3
"""
Migration script to clean up knowledge graph settings
Moves knowledge graph configuration from LLM settings to separate category
"""

import asyncio
import asyncpg
import json
from datetime import datetime

async def migrate_knowledge_graph_settings():
    """Migrate knowledge graph settings to separate category"""
    
    # Connect to database
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="postgres", 
        password="postgres",
        database="llm_platform"
    )
    
    try:
        # Get current LLM settings
        llm_settings = await conn.fetchrow(
            "SELECT settings FROM settings WHERE category = 'llm'"
        )
        
        if not llm_settings:
            print("No LLM settings found")
            return
            
        settings_data = llm_settings['settings']
        
        # Extract knowledge graph configuration
        kg_config = settings_data.get('knowledge_graph', {})
        
        if not kg_config:
            print("No knowledge graph configuration found in LLM settings")
            return
            
        print(f"Found knowledge graph config with {len(kg_config.get('prompts', []))} prompts")
        
        # Design clean knowledge graph settings structure
        clean_kg_settings = {
            # Model configuration (separate from main/second LLM)
            "model_config": {
                "model": kg_config.get('model', 'qwen3:30b-a3b-instruct-2507-q4_K_M'),
                "model_server": kg_config.get('model_server', 'http://localhost:11434'),
                "max_tokens": kg_config.get('max_tokens', 4096),
                "temperature": kg_config.get('temperature', 0.1),
                "context_length": kg_config.get('context_length', 32768)
            },
            
            # Entity extraction configuration
            "entity_extraction": {
                "enabled": True,
                "confidence_threshold": 0.6,
                "max_entities_per_chunk": 50,
                "entity_types": ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "EVENT", "TEMPORAL", "NUMERIC"],
                "validation_enabled": True
            },
            
            # Relationship extraction configuration
            "relationship_extraction": {
                "enabled": True,
                "confidence_threshold": 0.5,
                "max_relationships_per_chunk": 100,
                "pattern_extraction_enabled": True,
                "llm_enhancement_enabled": True,
                "llm_confidence_threshold": 0.4  # Our improved threshold
            },
            
            # Prompts (clean, specific to KG)
            "prompts": kg_config.get('prompts', []),
            
            # Anti-silo configuration
            "anti_silo": {
                "enabled": True,
                "similarity_threshold": 0.8,
                "max_cross_document_links": 1000
            },
            
            # Storage configuration
            "storage": {
                "neo4j_enabled": True,
                "create_hub_nodes": True,
                "entity_linking_enabled": True
            },
            
            # Quality settings (our improvements)
            "quality_control": {
                "filter_sentence_fragments": True,
                "max_entity_words": 5,
                "max_entity_length": 30,
                "traditional_extraction_supplement_only": True
            }
        }
        
        # Check if knowledge_graph category already exists
        existing_kg = await conn.fetchrow(
            "SELECT id FROM settings WHERE category = 'knowledge_graph'"
        )
        
        if existing_kg:
            # Update existing
            await conn.execute(
                "UPDATE settings SET settings = $1, updated_at = $2 WHERE category = 'knowledge_graph'",
                json.dumps(clean_kg_settings),
                datetime.now()
            )
            print("✅ Updated existing knowledge_graph settings")
        else:
            # Create new knowledge_graph category
            await conn.execute(
                "INSERT INTO settings (category, settings, updated_at) VALUES ($1, $2, $3)",
                'knowledge_graph',
                json.dumps(clean_kg_settings),
                datetime.now()
            )
            print("✅ Created new knowledge_graph settings category")
        
        # Remove knowledge_graph from LLM settings
        clean_llm_settings = dict(settings_data)
        if 'knowledge_graph' in clean_llm_settings:
            del clean_llm_settings['knowledge_graph']
            
            await conn.execute(
                "UPDATE settings SET settings = $1, updated_at = $2 WHERE category = 'llm'",
                json.dumps(clean_llm_settings),
                datetime.now()
            )
            print("✅ Removed knowledge_graph from LLM settings")
        
        # Verify the migration
        verification = await conn.fetchrow(
            "SELECT jsonb_object_keys(settings) as keys FROM settings WHERE category = 'knowledge_graph'"
        )
        
        kg_keys = await conn.fetch(
            "SELECT jsonb_object_keys(settings) as key FROM settings WHERE category = 'knowledge_graph'"
        )
        
        print(f"✅ Knowledge graph settings now has keys: {[row['key'] for row in kg_keys]}")
        
        # Check that LLM settings no longer has knowledge_graph
        llm_keys = await conn.fetch(
            "SELECT jsonb_object_keys(settings) as key FROM settings WHERE category = 'llm'"
        )
        
        print(f"✅ LLM settings now has keys: {[row['key'] for row in llm_keys]}")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(migrate_knowledge_graph_settings())