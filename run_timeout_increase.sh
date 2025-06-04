#!/bin/bash

echo "Increasing agent timeout values..."

# Run the SQL migration
docker exec -i jarvis-postgres-1 psql -U postgres -d postgres < increase_agent_timeouts.sql

echo "Timeout values updated successfully!"
echo ""
echo "New timeout values:"
docker exec -i jarvis-postgres-1 psql -U postgres -d postgres -c "SELECT name, role, config->>'timeout' as timeout_seconds FROM langgraph_agents ORDER BY (config->>'timeout')::int DESC;"