-- Clean Knowledge Graph Hardcoded Data - Fixed Version
-- Remove extraction_settings and hardcoded types from PostgreSQL settings

-- First, let's see what we're working with
SELECT 
    category,
    jsonb_pretty(settings->'prompts') as current_prompts
FROM settings 
WHERE category = 'knowledge_graph';

-- Clean up hardcoded types from prompts - using explicit jsonb operations
UPDATE settings 
SET settings = jsonb_set(
    settings,
    '{prompts}',
    (
        SELECT jsonb_agg(
            CASE 
                WHEN prompt_obj ? 'parameters' AND (prompt_obj->'parameters')::jsonb ? 'types' THEN
                    prompt_obj || jsonb_build_object('parameters', 
                        (prompt_obj->'parameters')::jsonb - 'types'::text
                    )
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
    jsonb_pretty(settings->'prompts') as cleaned_prompts
FROM settings 
WHERE category = 'knowledge_graph';