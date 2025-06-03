-- PostgreSQL Migration Script for MCP Schema Enhancement
-- This script adds support for command-based MCP servers to the existing schema

-- Step 1: Create the new mcp_servers table
CREATE TABLE IF NOT EXISTS mcp_servers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    config_type VARCHAR(20) NOT NULL CHECK (config_type IN ('manifest', 'command')),
    
    -- Manifest-based configuration
    manifest_url VARCHAR(255),
    hostname VARCHAR(255),
    api_key VARCHAR(255),
    
    -- Command-based configuration
    command VARCHAR(500),
    args JSONB,
    env JSONB,
    working_directory VARCHAR(500),
    
    -- Process management
    process_id INTEGER,
    is_running BOOLEAN DEFAULT false,
    restart_policy VARCHAR(20) DEFAULT 'on-failure' CHECK (restart_policy IN ('always', 'on-failure', 'never')),
    max_restarts INTEGER DEFAULT 3,
    restart_count INTEGER DEFAULT 0,
    
    -- Common fields
    is_active BOOLEAN NOT NULL DEFAULT true,
    last_health_check TIMESTAMP WITH TIME ZONE,
    health_status VARCHAR(20) DEFAULT 'unknown' CHECK (health_status IN ('healthy', 'unhealthy', 'unknown', 'starting', 'stopped')),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Step 2: Add server_id column to mcp_manifests table
ALTER TABLE mcp_manifests 
ADD COLUMN IF NOT EXISTS server_id INTEGER REFERENCES mcp_servers(id);

-- Step 3: Add server_id column to mcp_tools table  
ALTER TABLE mcp_tools 
ADD COLUMN IF NOT EXISTS server_id INTEGER REFERENCES mcp_servers(id);

-- Step 4: Migrate existing manifests to the new server structure
INSERT INTO mcp_servers (name, config_type, manifest_url, hostname, api_key, is_active, health_status, created_at, updated_at)
SELECT 
    COALESCE(
        (content->>'name'), 
        hostname, 
        'Server ' || id::text
    ) as name,
    'manifest' as config_type,
    url as manifest_url,
    hostname,
    api_key,
    true as is_active,
    'unknown' as health_status,
    created_at,
    updated_at
FROM mcp_manifests 
WHERE server_id IS NULL;

-- Step 5: Update mcp_manifests to reference the new servers
UPDATE mcp_manifests 
SET server_id = (
    SELECT s.id 
    FROM mcp_servers s 
    WHERE s.manifest_url = mcp_manifests.url 
    AND s.config_type = 'manifest'
    LIMIT 1
)
WHERE server_id IS NULL;

-- Step 6: Update mcp_tools to reference servers instead of manifests
UPDATE mcp_tools 
SET server_id = (
    SELECT server_id 
    FROM mcp_manifests 
    WHERE mcp_manifests.id = mcp_tools.manifest_id
)
WHERE server_id IS NULL AND manifest_id IS NOT NULL;

-- Step 7: Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_mcp_servers_config_type ON mcp_servers(config_type);
CREATE INDEX IF NOT EXISTS idx_mcp_servers_is_active ON mcp_servers(is_active);
CREATE INDEX IF NOT EXISTS idx_mcp_servers_is_running ON mcp_servers(is_running);
CREATE INDEX IF NOT EXISTS idx_mcp_servers_health_status ON mcp_servers(health_status);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_server_id ON mcp_tools(server_id);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_is_active ON mcp_tools(is_active);
CREATE INDEX IF NOT EXISTS idx_mcp_manifests_server_id ON mcp_manifests(server_id);

-- Step 8: Add updated_at trigger for mcp_servers
CREATE OR REPLACE FUNCTION update_mcp_servers_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER mcp_servers_updated_at
    BEFORE UPDATE ON mcp_servers
    FOR EACH ROW
    EXECUTE FUNCTION update_mcp_servers_updated_at();

-- Verify the migration
SELECT 
    'mcp_servers' as table_name, 
    COUNT(*) as row_count,
    COUNT(CASE WHEN config_type = 'manifest' THEN 1 END) as manifest_servers,
    COUNT(CASE WHEN config_type = 'command' THEN 1 END) as command_servers
FROM mcp_servers
UNION ALL
SELECT 
    'mcp_tools' as table_name,
    COUNT(*) as row_count,
    COUNT(CASE WHEN server_id IS NOT NULL THEN 1 END) as with_server_id,
    COUNT(CASE WHEN server_id IS NULL THEN 1 END) as without_server_id
FROM mcp_tools
UNION ALL
SELECT 
    'mcp_manifests' as table_name,
    COUNT(*) as row_count,
    COUNT(CASE WHEN server_id IS NOT NULL THEN 1 END) as with_server_id,
    COUNT(CASE WHEN server_id IS NULL THEN 1 END) as without_server_id
FROM mcp_manifests;

-- Show the updated schema
\d mcp_servers
\d mcp_tools  
\d mcp_manifests