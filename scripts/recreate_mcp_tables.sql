-- Drop existing tables if they exist
DROP TABLE IF EXISTS mcp_tools CASCADE;
DROP TABLE IF EXISTS mcp_manifests CASCADE;

-- Recreate tables with correct schema
CREATE TABLE mcp_manifests (
    id SERIAL PRIMARY KEY,
    url VARCHAR(255) NOT NULL UNIQUE,
    content JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE TABLE mcp_tools (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL DEFAULT 'POST',
    parameters JSONB,
    headers JSONB,
    is_active BOOLEAN DEFAULT true,
    manifest_id INTEGER REFERENCES mcp_manifests(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    UNIQUE(name, manifest_id)
); 