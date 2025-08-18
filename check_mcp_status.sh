#!/bin/bash
# Check MCP Server Status - Diagnostic script for the MCP server

# Configuration
MCP_PORT=3001
MCP_DIR="/Users/kianwoonwong/Downloads/MCP"
MCP_LOG_FILE="$MCP_DIR/mcp_server.log"
MCP_HEALTH_CHECK_URL="http://localhost:$MCP_PORT/health"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Function to check if MCP server is running
is_mcp_running() {
    if lsof -Pi :$MCP_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    fi
    
    if pgrep -f "MCP_MODE=http.*npm start" >/dev/null 2>&1; then
        return 0
    fi
    
    return 1
}

# Function to check server health
check_health() {
    local response=$(curl -s -w "\n%{http_code}" "$MCP_HEALTH_CHECK_URL" 2>/dev/null)
    local http_code=$(echo "$response" | tail -n1)
    
    if [[ "$http_code" == "200" ]] || [[ "$http_code" == "204" ]]; then
        return 0
    fi
    
    # Try base URL as fallback
    response=$(curl -s -w "\n%{http_code}" "http://localhost:$MCP_PORT" 2>/dev/null)
    http_code=$(echo "$response" | tail -n1)
    
    if [[ "$http_code" == "200" ]] || [[ "$http_code" == "204" ]] || [[ "$http_code" == "404" ]]; then
        return 0
    fi
    
    return 1
}

# Function to get process info
get_process_info() {
    local pid=$(lsof -ti:$MCP_PORT 2>/dev/null)
    if [ ! -z "$pid" ]; then
        echo "$pid"
    else
        pgrep -f "MCP_MODE=http.*npm start" | head -n1
    fi
}

# Main execution
main() {
    echo "==================================="
    echo "    MCP Server Status Check"
    echo "==================================="
    echo ""
    
    # Check port binding
    echo "Port Status:"
    if lsof -Pi :$MCP_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        local pid=$(lsof -ti:$MCP_PORT)
        print_status "Port $MCP_PORT is in use by PID: $pid"
    else
        print_error "Port $MCP_PORT is not in use"
    fi
    echo ""
    
    # Check process
    echo "Process Status:"
    local mcp_pid=$(get_process_info)
    if [ ! -z "$mcp_pid" ]; then
        print_status "MCP server process running (PID: $mcp_pid)"
        
        # Get process details
        if command -v ps &> /dev/null; then
            local cpu_mem=$(ps -p $mcp_pid -o %cpu,%mem,etime --no-headers 2>/dev/null)
            if [ ! -z "$cpu_mem" ]; then
                echo "  CPU: $(echo $cpu_mem | awk '{print $1}')%"
                echo "  Memory: $(echo $cpu_mem | awk '{print $2}')%"
                echo "  Uptime: $(echo $cpu_mem | awk '{print $3}')"
            fi
        fi
    else
        print_error "No MCP server process found"
    fi
    echo ""
    
    # Check health endpoint
    echo "Health Check:"
    if check_health; then
        print_status "Server is responding to health checks"
        
        # Try to get more info from the server
        local response=$(curl -s "$MCP_HEALTH_CHECK_URL" 2>/dev/null)
        if [ ! -z "$response" ] && [ "$response" != "null" ]; then
            echo "  Response: $response"
        fi
    else
        print_error "Server is not responding to health checks"
    fi
    echo ""
    
    # Check log file
    echo "Log File:"
    if [ -f "$MCP_LOG_FILE" ]; then
        print_info "Log file: $MCP_LOG_FILE"
        local log_size=$(du -h "$MCP_LOG_FILE" | cut -f1)
        echo "  Size: $log_size"
        echo "  Last modified: $(date -r "$MCP_LOG_FILE" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || stat -f '%Sm' "$MCP_LOG_FILE" 2>/dev/null)"
        
        # Show last few non-empty lines
        echo ""
        echo "  Last 5 log entries:"
        tail -n 20 "$MCP_LOG_FILE" | grep -v "^$" | tail -n 5 | sed 's/^/    /'
    else
        print_warning "Log file not found: $MCP_LOG_FILE"
    fi
    echo ""
    
    # Overall status
    echo "==================================="
    if is_mcp_running && check_health; then
        print_status "MCP Server Status: HEALTHY"
        echo ""
        echo "Server URL: http://localhost:$MCP_PORT"
    elif is_mcp_running; then
        print_warning "MCP Server Status: RUNNING BUT UNHEALTHY"
        echo ""
        echo "The server process is running but not responding to health checks."
        echo "Check the log file for errors: $MCP_LOG_FILE"
    else
        print_error "MCP Server Status: NOT RUNNING"
        echo ""
        echo "Start the server with: ./start_mcp_server.sh"
    fi
    echo "==================================="
}

# Run the main function
main "$@"