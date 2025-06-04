-- Fix DeepSeek settings for proper output length
-- max_tokens should be for output generation, not context window

-- Update the LLM settings to use proper max_tokens
UPDATE settings 
SET value = jsonb_set(
    value::jsonb, 
    '{max_tokens}', 
    '16384'::jsonb
)
WHERE key = 'llm';

-- Optional: Add context_length as a separate field
UPDATE settings 
SET value = jsonb_set(
    value::jsonb, 
    '{context_length}', 
    '128000'::jsonb
)
WHERE key = 'llm';

-- Verify the update
SELECT key, value->>'max_tokens' as max_tokens, value->>'context_length' as context_length 
FROM settings 
WHERE key = 'llm';