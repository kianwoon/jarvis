-- Migration: Add Enhanced Error Handling Configuration to MCP Servers
-- Description: Adds JSON columns for enhanced error handling and authentication refresh configuration

-- Add enhanced error handling configuration column
ALTER TABLE mcp_servers 
ADD COLUMN enhanced_error_handling_config JSON DEFAULT NULL;

-- Add authentication refresh configuration column  
ALTER TABLE mcp_servers
ADD COLUMN auth_refresh_config JSON DEFAULT NULL;

-- Add index for faster lookups of servers with enhanced error handling enabled
CREATE INDEX IF NOT EXISTS idx_mcp_servers_enhanced_error_handling 
ON mcp_servers USING GIN ((enhanced_error_handling_config->'enabled'));

-- Add comment to the table describing the new columns
COMMENT ON COLUMN mcp_servers.enhanced_error_handling_config IS 'JSON configuration for enhanced error handling including retry policies, timeouts, and circuit breaker settings';
COMMENT ON COLUMN mcp_servers.auth_refresh_config IS 'JSON configuration for automatic authentication token refresh including OAuth endpoints and refresh parameters';

-- Insert default configurations for existing servers to maintain backward compatibility
UPDATE mcp_servers 
SET enhanced_error_handling_config = jsonb_build_object(
    'enabled', true,
    'max_tool_retries', 3,
    'retry_base_delay', 1.0,
    'retry_max_delay', 60.0,
    'retry_backoff_multiplier', 2.0,
    'timeout_seconds', 30,
    'enable_circuit_breaker', true,
    'circuit_failure_threshold', 5,
    'circuit_recovery_timeout', 60
)
WHERE enhanced_error_handling_config IS NULL;

UPDATE mcp_servers
SET auth_refresh_config = jsonb_build_object(
    'enabled', false,
    'server_type', 'custom',
    'auth_type', 'oauth2',
    'refresh_endpoint', '',
    'refresh_method', 'POST',
    'refresh_headers', '{}',
    'refresh_data_template', '{}',
    'token_expiry_buffer_minutes', 5
)  
WHERE auth_refresh_config IS NULL;