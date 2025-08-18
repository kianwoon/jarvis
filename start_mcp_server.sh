#!/bin/bash
# Start MCP Server - Robust startup script for the MCP server

# Configuration
MCP_DIR="/Users/kianwoonwong/Downloads/MCP"
MCP_PORT=3001
MCP_LOG_FILE="mcp_server.log"
MCP_HEALTH_CHECK_URL="http://localhost:$MCP_PORT/health"
MAX_WAIT_TIME=30  # Maximum seconds to wait for server to become healthy
STARTUP_DELAY=2   # Initial delay before health checks

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_status() {
    echo -e "${GREEN}[MCP Server]${NC} $1"
}

print_error() {
    echo -e "${RED}[MCP Server Error]${NC} $1" >&2
}

print_warning() {
    echo -e "${YELLOW}[MCP Server Warning]${NC} $1"
}

# Function to check if MCP server is running
is_mcp_running() {
    # Check if port is in use
    if lsof -Pi :$MCP_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    fi
    
    # Also check if MCP process exists
    if pgrep -f "MCP_MODE=http.*npm start" >/dev/null 2>&1; then
        return 0
    fi
    
    return 1
}

# Function to check server health
check_health() {
    # Try to connect to the health endpoint
    if curl -s -f -o /dev/null -w "%{http_code}" "$MCP_HEALTH_CHECK_URL" 2>/dev/null | grep -q "200\|204"; then
        return 0
    fi
    
    # Alternative: check if server responds to base URL
    if curl -s -f -o /dev/null -w "%{http_code}" "http://localhost:$MCP_PORT" 2>/dev/null | grep -qE "200|204|404"; then
        return 0
    fi
    
    return 1
}

# Function to wait for server to become healthy
wait_for_health() {
    local elapsed=0
    
    print_status "Waiting for MCP server to become healthy..."
    
    # Initial delay to give the server time to start
    sleep $STARTUP_DELAY
    
    while [ $elapsed -lt $MAX_WAIT_TIME ]; do
        if check_health; then
            print_status "MCP server is healthy and ready!"
            return 0
        fi
        
        # Check if process died during startup
        if ! is_mcp_running; then
            print_error "MCP server process died during startup"
            return 1
        fi
        
        echo -n "."
        sleep 1
        elapsed=$((elapsed + 1))
    done
    
    echo ""
    print_warning "MCP server did not become healthy within $MAX_WAIT_TIME seconds"
    return 1
}

# Function to kill existing MCP server if needed
kill_existing_mcp() {
    print_warning "Stopping existing MCP server..."
    
    # Try to kill by port
    local pid=$(lsof -ti:$MCP_PORT)
    if [ ! -z "$pid" ]; then
        kill -TERM $pid 2>/dev/null
        sleep 2
        # Force kill if still running
        if kill -0 $pid 2>/dev/null; then
            kill -KILL $pid 2>/dev/null
        fi
    fi
    
    # Also kill by process pattern
    pkill -f "MCP_MODE=http.*npm start" 2>/dev/null
    
    sleep 1
}

# Main execution
main() {
    print_status "Starting MCP server management..."
    
    # Check if MCP directory exists
    if [ ! -d "$MCP_DIR" ]; then
        print_error "MCP directory not found: $MCP_DIR"
        exit 1
    fi
    
    # Check if npm is available
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed or not in PATH"
        exit 1
    fi
    
    # Check if server is already running and healthy
    if is_mcp_running; then
        print_status "MCP server appears to be running, checking health..."
        if check_health; then
            print_status "MCP server is already running and healthy on port $MCP_PORT"
            exit 0
        else
            print_warning "MCP server is running but not healthy, restarting..."
            kill_existing_mcp
        fi
    fi
    
    # Change to MCP directory
    cd "$MCP_DIR" || {
        print_error "Failed to change to MCP directory: $MCP_DIR"
        exit 1
    }
    
    # Check if package.json exists
    if [ ! -f "package.json" ]; then
        print_error "package.json not found in $MCP_DIR"
        exit 1
    fi
    
    # Install dependencies if node_modules doesn't exist
    if [ ! -d "node_modules" ]; then
        print_status "Installing MCP dependencies..."
        npm install || {
            print_error "Failed to install dependencies"
            exit 1
        }
    fi
    
    # Create log file if it doesn't exist
    touch "$MCP_LOG_FILE"
    
    # Start the MCP server in background
    print_status "Starting MCP server on port $MCP_PORT..."
    print_status "Logging to: $MCP_DIR/$MCP_LOG_FILE"
    
    # Export environment variables and start server
    export MCP_MODE=http
    export MCP_PORT=$MCP_PORT
    
    # Start server with nohup to prevent it from dying when script exits
    nohup npm start >> "$MCP_LOG_FILE" 2>&1 &
    local mcp_pid=$!
    
    # Check if process started successfully
    sleep 1
    if ! kill -0 $mcp_pid 2>/dev/null; then
        print_error "Failed to start MCP server process"
        print_error "Check the log file for details: $MCP_DIR/$MCP_LOG_FILE"
        tail -n 20 "$MCP_LOG_FILE"
        exit 1
    fi
    
    print_status "MCP server process started with PID: $mcp_pid"
    
    # Wait for server to become healthy
    if wait_for_health; then
        print_status "MCP server successfully started and is ready!"
        print_status "Server URL: http://localhost:$MCP_PORT"
        print_status "Log file: $MCP_DIR/$MCP_LOG_FILE"
        exit 0
    else
        print_error "MCP server failed to become healthy"
        print_error "Last 20 lines of log file:"
        tail -n 20 "$MCP_LOG_FILE"
        
        # Optionally kill the unhealthy server
        print_warning "Killing unhealthy MCP server process..."
        kill $mcp_pid 2>/dev/null
        exit 1
    fi
}

# Run the main function
main "$@"