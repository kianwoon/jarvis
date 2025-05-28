-- Add is_active column to langgraph_agents table if it doesn't exist

-- For PostgreSQL:
ALTER TABLE langgraph_agents 
ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;

-- If the column was added without default, update existing rows
UPDATE langgraph_agents 
SET is_active = TRUE 
WHERE is_active IS NULL;