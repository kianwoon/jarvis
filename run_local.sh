#!/bin/bash
# Run the app locally with M3 GPU support

echo "Starting Jarvis locally with Apple M3 GPU support..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Start MCP server first
echo "==================================="
echo "Starting MCP Server..."
echo "==================================="
"$SCRIPT_DIR/start_mcp_server.sh"
MCP_STATUS=$?

if [ $MCP_STATUS -ne 0 ]; then
    echo "Warning: MCP server failed to start (exit code: $MCP_STATUS)"
    echo "Continuing with Jarvis startup anyway..."
    echo ""
fi

# Load local environment variables
export $(cat .env.local | grep -v '^#' | xargs)

# Run the app
echo ""
echo "==================================="
echo "Starting Jarvis Backend..."
echo "==================================="
echo "Environment: ENABLE_QWEN_RERANKER=$ENABLE_QWEN_RERANKER"
echo "Device: QWEN_RERANKER_DEVICE=$QWEN_RERANKER_DEVICE"
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000