-- Add configuration columns to langgraph_agents table
ALTER TABLE langgraph_agents ADD COLUMN IF NOT EXISTS config JSON DEFAULT '{}';

-- Default configuration structure:
-- {
--   "max_tokens": 4000,
--   "timeout": 30,
--   "temperature": 0.7,
--   "response_mode": "complete",  -- or "streaming" or "chunked"
--   "chunk_size": 1000,
--   "allow_continuation": true
-- }

-- Update existing agents with sensible defaults based on their roles
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
  "max_tokens": 6000,
  "timeout": 45,
  "temperature": 0.6,
  "response_mode": "complete"
}'
WHERE role = 'service_delivery_manager';

-- Financial analyst needs precision
UPDATE langgraph_agents 
SET config = '{
  "max_tokens": 5000,
  "timeout": 40,
  "temperature": 0.5,
  "response_mode": "complete"
}'
WHERE role = 'financial_analyst';

-- Technical architect needs detailed responses
UPDATE langgraph_agents 
SET config = '{
  "max_tokens": 5000,
  "timeout": 40,
  "temperature": 0.6,
  "response_mode": "complete"
}'
WHERE role = 'technical_architect';