#!/bin/bash

# Run the model configs migration inside the Docker postgres container
docker exec -i jarvis-postgres-1 psql -U postgres -d llm_platform << 'EOF'
-- Migration to add model-specific configurations support

-- Add context_length to existing LLM settings if not present
UPDATE settings 
SET settings = jsonb_set(
    settings::jsonb, 
    '{context_length}', 
    CASE 
        WHEN settings::jsonb->>'model' LIKE '%deepseek%' THEN '128000'::jsonb
        WHEN settings::jsonb->>'model' LIKE '%qwen%' THEN '32768'::jsonb
        ELSE '8192'::jsonb
    END
)
WHERE category = 'llm' 
AND NOT (settings::jsonb ? 'context_length');

-- Fix max_tokens if it's set to context window size
UPDATE settings 
SET settings = jsonb_set(
    settings::jsonb, 
    '{max_tokens}', 
    '16384'::jsonb
)
WHERE category = 'llm' 
AND (settings::jsonb->>'max_tokens')::int > 32768;

-- Create model_presets table if not exists (already created)
-- Skip since it was already created

-- Verify the update
SELECT 
    category,
    settings->>'model' as model,
    settings->>'max_tokens' as max_tokens,
    settings->>'context_length' as context_length
FROM settings 
WHERE category = 'llm';
EOF

echo "Migration completed!"