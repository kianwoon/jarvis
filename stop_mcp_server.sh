#!/bin/bash
# Stop MCP Server - Clean shutdown script for the MCP server

# Configuration
MCP_PORT=3001

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

# Main execution
main() {
    print_status "Stopping MCP server..."
    
    if ! is_mcp_running; then
        print_status "MCP server is not running"
        exit 0
    fi
    
    # Try to find and kill by port
    local pid=$(lsof -ti:$MCP_PORT)
    if [ ! -z "$pid" ]; then
        print_status "Found MCP server process on port $MCP_PORT (PID: $pid)"
        
        # Try graceful shutdown first
        print_status "Sending SIGTERM to process $pid..."
        kill -TERM $pid 2>/dev/null
        
        # Wait up to 5 seconds for graceful shutdown
        local count=0
        while [ $count -lt 5 ]; do
            if ! kill -0 $pid 2>/dev/null; then
                print_status "MCP server stopped gracefully"
                break
            fi
            sleep 1
            count=$((count + 1))
        done
        
        # Force kill if still running
        if kill -0 $pid 2>/dev/null; then
            print_warning "Process did not stop gracefully, forcing shutdown..."
            kill -KILL $pid 2>/dev/null
            sleep 1
            if ! kill -0 $pid 2>/dev/null; then
                print_status "MCP server forcefully stopped"
            else
                print_error "Failed to stop MCP server process"
                exit 1
            fi
        fi
    fi
    
    # Also kill by process pattern (cleanup any orphaned processes)
    local killed_any=false
    for pid in $(pgrep -f "MCP_MODE=http.*npm start"); do
        print_warning "Killing orphaned MCP process: $pid"
        kill -KILL $pid 2>/dev/null
        killed_any=true
    done
    
    if [ "$killed_any" = true ]; then
        print_status "Cleaned up orphaned MCP processes"
    fi
    
    # Final verification
    if is_mcp_running; then
        print_error "MCP server is still running after shutdown attempt"
        exit 1
    else
        print_status "MCP server successfully stopped"
        exit 0
    fi
}

# Run the main function
main "$@"