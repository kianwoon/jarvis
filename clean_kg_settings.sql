-- Clean up knowledge graph settings migration
-- Move knowledge graph configuration from LLM settings to separate category

-- First, let's create the clean knowledge graph settings
INSERT INTO settings (category, settings, updated_at) 
VALUES (
    'knowledge_graph',
    '{
        "model_config": {
            "model": "qwen3:30b-a3b-instruct-2507-q4_K_M",
            "model_server": "http://localhost:11434",
            "max_tokens": 4096,
            "temperature": 0.1,
            "context_length": 32768
        },
        "entity_extraction": {
            "enabled": true,
            "confidence_threshold": 0.6,
            "max_entities_per_chunk": 50,
            "entity_types": ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "EVENT", "TEMPORAL", "NUMERIC"],
            "validation_enabled": true
        },
        "relationship_extraction": {
            "enabled": true,
            "confidence_threshold": 0.5,
            "max_relationships_per_chunk": 100,
            "pattern_extraction_enabled": true,
            "llm_enhancement_enabled": true,
            "llm_confidence_threshold": 0.4
        },
        "anti_silo": {
            "enabled": true,
            "similarity_threshold": 0.8,
            "max_cross_document_links": 1000
        },
        "storage": {
            "neo4j_enabled": true,
            "create_hub_nodes": true,
            "entity_linking_enabled": true
        },
        "quality_control": {
            "filter_sentence_fragments": true,
            "max_entity_words": 5,
            "max_entity_length": 30,
            "traditional_extraction_supplement_only": true
        }
    }',
    NOW()
) ON CONFLICT (category) DO UPDATE SET 
    settings = EXCLUDED.settings,
    updated_at = EXCLUDED.updated_at;

-- Copy prompts from the knowledge_graph section in LLM settings to the new category
UPDATE settings 
SET settings = settings || jsonb_build_object(
    'prompts', 
    (SELECT settings->'knowledge_graph'->'prompts' FROM settings WHERE category = 'llm')
)
WHERE category = 'knowledge_graph';

-- Remove knowledge_graph section from LLM settings
UPDATE settings 
SET settings = settings - 'knowledge_graph',
    updated_at = NOW()
WHERE category = 'llm';

-- Verify the changes
SELECT 'LLM Settings Keys:' as info, jsonb_object_keys(settings) as keys FROM settings WHERE category = 'llm'
UNION ALL
SELECT 'Knowledge Graph Settings Keys:' as info, jsonb_object_keys(settings) as keys FROM settings WHERE category = 'knowledge_graph';