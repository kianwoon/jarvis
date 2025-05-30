-- Add config column to langgraph_agents table
ALTER TABLE langgraph_agents ADD COLUMN IF NOT EXISTS config JSON DEFAULT '{}';

-- Set default configuration for all agents
UPDATE langgraph_agents 
SET config = '{
  "max_tokens": 4000,
  "timeout": 30,
  "temperature": 0.7,
  "response_mode": "complete"
}'
WHERE config IS NULL OR config = '{}';

-- Service delivery manager needs more tokens for tables
UPDATE langgraph_agents 
SET config = '{
  "max_tokens": 8000,
  "timeout": 60,
  "temperature": 0.6,
  "response_mode": "complete",
  "allow_continuation": true
}'
WHERE role = 'service_delivery_manager';

-- Financial analyst needs precision and more tokens
UPDATE langgraph_agents 
SET config = '{
  "max_tokens": 6000,
  "timeout": 45,
  "temperature": 0.5,
  "response_mode": "complete"
}'
WHERE role = 'financial_analyst';

-- Technical architect needs detailed responses
UPDATE langgraph_agents 
SET config = '{
  "max_tokens": 6000,
  "timeout": 45,
  "temperature": 0.6,
  "response_mode": "complete"
}'
WHERE role = 'technical_architect';

-- Sales strategist
UPDATE langgraph_agents 
SET config = '{
  "max_tokens": 5000,
  "timeout": 40,
  "temperature": 0.7,
  "response_mode": "complete"
}'
WHERE role = 'sales_strategist';

-- Router agent - quick decisions
UPDATE langgraph_agents 
SET config = '{
  "max_tokens": 1000,
  "timeout": 15,
  "temperature": 0.3,
  "response_mode": "complete"
}'
WHERE role = 'router';

-- Document researcher - may need long responses
UPDATE langgraph_agents 
SET config = '{
  "max_tokens": 6000,
  "timeout": 45,
  "temperature": 0.5,
  "response_mode": "complete"
}'
WHERE role = 'document_researcher';

-- Synthesizer - combines all responses
UPDATE langgraph_agents 
SET config = '{
  "max_tokens": 8000,
  "timeout": 60,
  "temperature": 0.6,
  "response_mode": "complete"
}'
WHERE role = 'synthesizer';