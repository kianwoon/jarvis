#!/bin/bash
# Script to run the PostgreSQL migration in Docker environment

echo "🚀 Running MCP PostgreSQL schema migration..."

# Method 1: Using docker-compose exec to run the Python migration script
echo "Running Python migration script in Docker container..."
docker-compose exec app python run_postgres_migration.py

# Method 2: Alternative - run SQL directly against PostgreSQL
echo ""
echo "Alternative: Running SQL migration directly..."
docker-compose exec postgres psql -U postgres -d llm_platform -f /app/migrate_postgres_schema.sql

echo ""
echo "Verifying migration results..."
docker-compose exec postgres psql -U postgres -d llm_platform -c "
SELECT 
    table_name,
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns 
WHERE table_schema = 'public' 
AND table_name IN ('mcp_servers', 'mcp_tools', 'mcp_manifests')
AND column_name IN ('server_id', 'config_type', 'command', 'args', 'env')
ORDER BY table_name, column_name;
"

echo ""
echo "Checking data migration..."
docker-compose exec postgres psql -U postgres -d llm_platform -c "
SELECT 
    'mcp_servers' as table_name, 
    COUNT(*) as total,
    COUNT(CASE WHEN config_type = 'manifest' THEN 1 END) as manifests,
    COUNT(CASE WHEN config_type = 'command' THEN 1 END) as commands
FROM mcp_servers
UNION ALL
SELECT 
    'mcp_tools',
    COUNT(*),
    COUNT(CASE WHEN server_id IS NOT NULL THEN 1 END),
    COUNT(CASE WHEN server_id IS NULL THEN 1 END)
FROM mcp_tools;
"

echo "✅ Migration verification complete!"