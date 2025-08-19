-- Migration to add communication_protocol field to mcp_servers table
-- This allows explicit control over the communication protocol independent of server type

-- Add communication_protocol column with default value
ALTER TABLE mcp_servers 
ADD COLUMN IF NOT EXISTS communication_protocol VARCHAR(20) DEFAULT 'stdio';

-- Update existing servers based on their config_type
-- Command servers default to stdio
UPDATE mcp_servers 
SET communication_protocol = 'stdio' 
WHERE config_type = 'command' AND communication_protocol IS NULL;

-- HTTP and remote_http servers default to http
UPDATE mcp_servers 
SET communication_protocol = 'http' 
WHERE config_type IN ('http', 'remote_http') AND communication_protocol IS NULL;

-- Manifest servers default to http (they typically use HTTP endpoints)
UPDATE mcp_servers 
SET communication_protocol = 'http' 
WHERE config_type = 'manifest' AND communication_protocol IS NULL;

-- For servers with SSE in their remote_config, update to sse
UPDATE mcp_servers 
SET communication_protocol = 'sse' 
WHERE config_type = 'remote_http' 
  AND remote_config->>'transport_type' = 'sse' 
  AND communication_protocol = 'http';

-- Add a comment to document the field
COMMENT ON COLUMN mcp_servers.communication_protocol IS 'Communication protocol used by the server: stdio, http, sse, websocket';