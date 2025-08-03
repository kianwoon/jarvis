-- Clean Knowledge Graph Hardcoded Data
-- Remove extraction_settings and hardcoded types from PostgreSQL settings

UPDATE settings 
SET settings = settings - 'extraction_settings'
WHERE category = 'knowledge_graph';

-- Clean up hardcoded entity_types and relationship_types from prompts
UPDATE settings 
SET settings = jsonb_set(
    settings,
    '{prompts}',
    (
        SELECT jsonb_agg(
            CASE 
                WHEN prompt_obj ? 'parameters' AND prompt_obj->'parameters' ? 'types' THEN
                    prompt_obj - 'parameters' || jsonb_build_object('parameters', prompt_obj->'parameters' - 'types')
                ELSE 
                    prompt_obj
            END
        )
        FROM jsonb_array_elements(settings->'prompts') AS prompt_obj
    )
)
WHERE category = 'knowledge_graph' 
AND settings ? 'prompts';

-- Verify the cleanup
SELECT 
    category,
    CASE 
        WHEN settings ? 'extraction_settings' THEN 'STILL HAS extraction_settings'
        ELSE 'CLEAN: No extraction_settings'
    END as extraction_status,
    CASE 
        WHEN settings->'prompts' @> '[{"parameters": {"types": []}}]' THEN 'STILL HAS hardcoded types'
        ELSE 'CLEAN: No hardcoded types in prompts'
    END as prompt_status
FROM settings 
WHERE category = 'knowledge_graph';