#!/bin/bash

# Run the model configs migration inside the Docker postgres container
docker exec -i jarvis-postgres-1 psql -U postgres -d llm_platform << 'EOF'
-- Migration to add model-specific configurations support
-- This allows storing different settings for different models

-- Add context_length to existing settings if not present
UPDATE settings 
SET value = jsonb_set(
    value::jsonb, 
    '{context_length}', 
    CASE 
        WHEN value::jsonb->>'model' LIKE '%deepseek%' THEN '128000'::jsonb
        WHEN value::jsonb->>'model' LIKE '%qwen%' THEN '32768'::jsonb
        ELSE '8192'::jsonb
    END
)
WHERE key = 'llm' 
AND NOT (value::jsonb ? 'context_length');

-- Fix max_tokens if it's set to context window size
UPDATE settings 
SET value = jsonb_set(
    value::jsonb, 
    '{max_tokens}', 
    '16384'::jsonb
)
WHERE key = 'llm' 
AND (value::jsonb->>'max_tokens')::int > 32768;

-- Add model_specific_configs column for future use
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name='settings' 
        AND column_name='model_specific_configs'
    ) THEN
        ALTER TABLE settings ADD COLUMN model_specific_configs JSONB DEFAULT '{}';
    END IF;
END $$;

-- Create model_presets table for storing presets (optional, for persistence)
CREATE TABLE IF NOT EXISTS model_presets (
    model_name VARCHAR(255) PRIMARY KEY,
    display_name VARCHAR(255) NOT NULL,
    context_length INTEGER NOT NULL,
    recommended_max_tokens INTEGER NOT NULL,
    default_temperature FLOAT NOT NULL,
    default_top_p FLOAT NOT NULL,
    supports_thinking BOOLEAN DEFAULT FALSE,
    notes TEXT,
    recommended_prompts JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default presets (optional, since they're in the Python code)
INSERT INTO model_presets (
    model_name, display_name, context_length, recommended_max_tokens,
    default_temperature, default_top_p, supports_thinking, notes
) VALUES 
(
    'deepseek-r1:8b', 
    'DeepSeek R1 8B', 
    128000, 
    16384, 
    0.8, 
    0.95, 
    true, 
    'Reasoning-optimized model. Uses ~66% tokens for thinking. For detailed visible output, use non-thinking mode or increase max_tokens.'
),
(
    'qwen2.5:8b', 
    'Qwen 2.5 8B', 
    32768, 
    8192, 
    0.7, 
    0.9, 
    false, 
    'Balanced model with good verbosity and accuracy. Excellent for general-purpose tasks.'
),
(
    'llama3.1:8b', 
    'Llama 3.1 8B', 
    8192, 
    4096, 
    0.7, 
    0.9, 
    false, 
    'Fast and efficient for general tasks. Limited context window.'
)
ON CONFLICT (model_name) DO UPDATE SET
    updated_at = CURRENT_TIMESTAMP;

-- Verify the update
SELECT 
    key,
    value->>'model' as model,
    value->>'max_tokens' as max_tokens,
    value->>'context_length' as context_length
FROM settings 
WHERE key = 'llm';
EOF

echo "Migration completed!"