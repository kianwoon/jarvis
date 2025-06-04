-- Increase timeout values for all agents to prevent premature timeouts
-- Default timeout increased from 30s to 60s for standard agents
-- Strategic/complex agents get even more time

-- Update all agents to have at least 60 second timeout
UPDATE langgraph_agents 
SET config = jsonb_set(
    COALESCE(config::jsonb, '{}'::jsonb),
    '{timeout}',
    '60'
)
WHERE (config->>'timeout')::int < 60 OR config->>'timeout' IS NULL;

-- Give document researcher more time for RAG searches
UPDATE langgraph_agents 
SET config = jsonb_set(
    COALESCE(config::jsonb, '{}'::jsonb),
    '{timeout}',
    '90'
)
WHERE name = 'document_researcher' OR role = 'researcher';

-- Strategic agents need more time for complex analysis
UPDATE langgraph_agents 
SET config = jsonb_set(
    COALESCE(config::jsonb, '{}'::jsonb),
    '{timeout}',
    '120'
)
WHERE role IN ('sales_strategist', 'technical_architect', 'financial_analyst', 'service_delivery_manager')
   OR name LIKE '%strategist%' 
   OR name LIKE '%architect%'
   OR name LIKE '%analyst%'
   OR name LIKE '%ceo%'
   OR name LIKE '%cto%'
   OR name LIKE '%cio%';

-- Large generation agents need even more time
UPDATE langgraph_agents 
SET config = jsonb_set(
    COALESCE(config::jsonb, '{}'::jsonb),
    '{timeout}',
    '180'
)
WHERE name LIKE '%continuation_agent%' 
   OR name LIKE '%generator%'
   OR role = 'generator';

-- Show updated timeouts
SELECT name, role, config->>'timeout' as timeout_seconds 
FROM langgraph_agents 
ORDER BY (config->>'timeout')::int DESC;