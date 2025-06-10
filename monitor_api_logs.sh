#!/bin/bash
echo "Monitoring API logs for langgraph agent updates..."
echo "Update an agent in the LangGraph page and watch the logs below:"
echo "=================================================="
docker logs -f jarvis-app-1 2>&1 | grep -E "(DEBUG|langgraph|agent|26)" | grep -v "mcp_tools_cache"