-- Fix Local MCP server configuration
-- Change from "command" type to "http" type
-- The Local MCP server is actually an HTTP server running on port 3001

-- Update the server configuration
UPDATE mcp_servers 
SET 
    config_type = 'http',
    hostname = 'host.docker.internal:3001',  -- Use host.docker.internal for Docker access
    command = NULL,  -- Clear the command field
    args = NULL,     -- Clear the args field
    working_directory = NULL  -- Clear working directory as it's not needed for HTTP servers
WHERE 
    name = 'Local MCP' 
    AND id = 9;

-- Verify the update
SELECT 
    id, 
    name, 
    config_type, 
    hostname, 
    command, 
    args,
    env IS NOT NULL as has_env,
    is_active
FROM mcp_servers 
WHERE name = 'Local MCP';

-- Note: Environment variables will be preserved in the env field
-- The HTTP server will still have access to the env vars for configuration