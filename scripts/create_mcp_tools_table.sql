CREATE TABLE IF NOT EXISTS mcp_tools (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL DEFAULT 'POST',
    parameters JSONB,
    headers JSONB,
    is_active BOOLEAN DEFAULT true,
    manifest_id INTEGER REFERENCES mcp_manifests(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE TABLE IF NOT EXISTS mcp_manifests (
    id SERIAL PRIMARY KEY,
    url VARCHAR(255) NOT NULL UNIQUE,
    content JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
); 