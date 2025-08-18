#!/bin/bash

# Script to run and test the Working MCP HTTP Bridge Server

echo "=================================="
echo "Working MCP HTTP Bridge Server"
echo "=================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Function to check if server is running
check_server() {
    curl -s http://localhost:3001/health > /dev/null 2>&1
    return $?
}

# Function to stop the server
stop_server() {
    echo "Stopping server..."
    if [ ! -z "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
    fi
    # Also try to kill any process on port 3001
    lsof -ti:3001 | xargs kill -9 2>/dev/null
}

# Trap to ensure server stops on script exit
trap stop_server EXIT

# Check if server is already running
if check_server; then
    echo "Server is already running on port 3001"
    echo "Do you want to restart it? (y/n)"
    read -r response
    if [[ "$response" == "y" ]]; then
        echo "Stopping existing server..."
        lsof -ti:3001 | xargs kill -9 2>/dev/null
        sleep 2
    else
        echo "Using existing server..."
    fi
fi

# Start the server if not running
if ! check_server; then
    echo "Starting Working MCP HTTP Bridge Server..."
    echo "----------------------------------------"
    
    # Start server in background
    python3 working_mcp_bridge.py > bridge_server.log 2>&1 &
    SERVER_PID=$!
    
    echo "Server PID: $SERVER_PID"
    echo "Waiting for server to start..."
    
    # Wait for server to be ready (max 30 seconds)
    for i in {1..30}; do
        if check_server; then
            echo "✓ Server is running!"
            break
        fi
        echo -n "."
        sleep 1
    done
    echo ""
    
    if ! check_server; then
        echo "❌ Server failed to start after 30 seconds"
        echo "Check bridge_server.log for errors:"
        tail -20 bridge_server.log
        exit 1
    fi
fi

# Run tests
echo ""
echo "=================================="
echo "Running Tests"
echo "=================================="
echo ""

python3 test_working_bridge.py

# Show server logs if tests failed
if [ $? -ne 0 ]; then
    echo ""
    echo "=================================="
    echo "Server Logs (last 50 lines)"
    echo "=================================="
    tail -50 bridge_server.log
fi

echo ""
echo "=================================="
echo "Test Complete"
echo "=================================="
echo ""
echo "Server is still running on http://localhost:3001"
echo "You can test it manually with:"
echo '  curl http://localhost:3001/health'
echo '  curl -X POST http://localhost:3001/tools/google_search -H "Content-Type: application/json" -d "{\"query\": \"test search\", \"num_results\": 2}"'
echo ""
echo "To stop the server, press Ctrl+C or run:"
echo "  kill $SERVER_PID"
echo ""
echo "Server logs are in: bridge_server.log"
echo ""

# Keep script running to maintain server
echo "Press Ctrl+C to stop the server..."
wait $SERVER_PID