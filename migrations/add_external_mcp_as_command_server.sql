-- Migration: Add External MCP Server as a configurable command-based server
-- Date: 2025-08-18
-- Description: Transforms the hardcoded External MCP Server into a configurable database entry

-- Check if the External MCP Server already exists before inserting
DO $$
BEGIN
    -- Only insert if no server named "External MCP Server" exists
    IF NOT EXISTS (
        SELECT 1 FROM mcp_servers 
        WHERE name = 'External MCP Server'
    ) THEN
        -- Insert the External MCP Server as a command-based server
        INSERT INTO mcp_servers (
            name,
            config_type,
            command,
            args,
            working_directory,
            env,
            restart_policy,
            max_restarts,
            is_active,
            enhanced_error_handling_config,
            auth_refresh_config,
            created_at,
            updated_at
        ) VALUES (
            'External MCP Server',
            'command',
            'npm',
            '["start"]'::jsonb,
            '/Users/kianwoonwong/Downloads/MCP',
            '{"MCP_MODE": "http", "MCP_PORT": "3001"}'::jsonb,
            'on-failure',
            3,
            true,
            '{
                "enabled": true,
                "max_tool_retries": 3,
                "retry_base_delay": 1.0,
                "retry_max_delay": 60.0,
                "retry_backoff_multiplier": 2.0,
                "timeout_seconds": 30,
                "enable_circuit_breaker": true,
                "circuit_failure_threshold": 5,
                "circuit_recovery_timeout": 60
            }'::jsonb,
            '{
                "enabled": false,
                "server_type": "custom",
                "auth_type": "oauth2",
                "refresh_method": "POST",
                "token_expiry_buffer_minutes": 5
            }'::jsonb,
            NOW(),
            NOW()
        );
        
        RAISE NOTICE 'External MCP Server added successfully as a command-based server';
    ELSE
        -- If it exists but is not command type, update it
        UPDATE mcp_servers
        SET 
            config_type = 'command',
            command = 'npm',
            args = '["start"]'::jsonb,
            working_directory = '/Users/kianwoonwong/Downloads/MCP',
            env = '{"MCP_MODE": "http", "MCP_PORT": "3001"}'::jsonb,
            restart_policy = 'on-failure',
            max_restarts = 3,
            updated_at = NOW()
        WHERE name = 'External MCP Server' 
        AND config_type != 'command';
        
        IF FOUND THEN
            RAISE NOTICE 'External MCP Server updated to command-based configuration';
        ELSE
            RAISE NOTICE 'External MCP Server already exists with correct configuration';
        END IF;
    END IF;
END $$;

-- Create an index to speed up server lookups by name if it doesn't exist
CREATE INDEX IF NOT EXISTS idx_mcp_servers_name ON mcp_servers(name);

-- Verify the migration
SELECT 
    id,
    name,
    config_type,
    command,
    args,
    working_directory,
    env,
    is_active
FROM mcp_servers 
WHERE name = 'External MCP Server';