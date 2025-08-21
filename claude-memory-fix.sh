#!/bin/bash

# Claude Code Permanent Memory Fix Setup Script

echo "🔧 Setting up permanent Claude Code memory fix..."

# Detect shell
if [[ $SHELL == *"zsh"* ]]; then
    SHELL_CONFIG="$HOME/.zshrc"
    echo "Detected zsh shell"
elif [[ $SHELL == *"bash"* ]]; then
    SHELL_CONFIG="$HOME/.bashrc"
    echo "Detected bash shell"
else
    echo "Unknown shell: $SHELL"
    echo "Please manually add the following to your shell config:"
    echo 'export NODE_OPTIONS="--max-old-space-size=8192"'
    exit 1
fi

# Create backup of current config
cp "$SHELL_CONFIG" "${SHELL_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
echo "✅ Created backup of $SHELL_CONFIG"

# Check if NODE_OPTIONS already exists
if grep -q "NODE_OPTIONS" "$SHELL_CONFIG"; then
    echo "⚠️  NODE_OPTIONS already exists in $SHELL_CONFIG"
    echo "Current setting:"
    grep "NODE_OPTIONS" "$SHELL_CONFIG"
    read -p "Do you want to replace it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Remove existing NODE_OPTIONS lines
        sed -i.bak '/NODE_OPTIONS/d' "$SHELL_CONFIG"
        echo "🗑️  Removed existing NODE_OPTIONS"
    else
        echo "❌ Skipping NODE_OPTIONS setup"
        exit 0
    fi
fi

# Add Claude Code optimizations
cat >> "$SHELL_CONFIG" << 'EOF'

# ========================================
# Claude Code Memory & Performance Optimizations
# ========================================

# Increase Node.js memory limit for Claude Code (8GB)
export NODE_OPTIONS="--max-old-space-size=8192"

# Optional: Add Claude Code to PATH if installed globally
# export PATH="$PATH:$(npm config get prefix)/bin"

# Claude Code aliases for convenience
alias claude-fresh="claude --clear"
alias claude-continue="claude --continue"
alias claude-memory="node -e \"console.log('Memory usage:', Math.round(process.memoryUsage().heapUsed / 1024 / 1024), 'MB')\""

# Function to check Claude Code memory before starting
claude-check() {
    echo "🔍 System memory check:"
    echo "Available memory: $(vm_stat | grep 'Pages free' | awk '{print $3}' | sed 's/\.//' | awk '{print $1 * 4096 / 1024 / 1024}' | cut -d. -f1) MB"
    echo "Node memory limit: $(node -e \"console.log(Math.round(v8.getHeapStatistics().heap_size_limit / 1024 / 1024))\") MB"
    echo "🚀 Starting Claude Code..."
    claude "$@"
}

EOF

echo "✅ Added Claude Code optimizations to $SHELL_CONFIG"

# Source the config to apply immediately
source "$SHELL_CONFIG"
echo "✅ Applied settings to current session"

# Verify the setup
echo ""
echo "🔍 Verifying setup:"
echo "NODE_OPTIONS: $NODE_OPTIONS"
echo "Node memory limit: $(node -e "console.log(Math.round(v8.getHeapStatistics().heap_size_limit / 1024 / 1024))" 2>/dev/null || echo "Unable to check") MB"

echo ""
echo "🎉 Setup complete! The following has been added:"
echo "   • NODE_OPTIONS set to 8GB memory limit"
echo "   • Useful Claude Code aliases"
echo "   • Memory checking function"
echo ""
echo "🔄 Please restart your terminal or run: source $SHELL_CONFIG"
echo ""
echo "💡 New commands available:"
echo "   • claude-fresh    → Start Claude with cleared history"
echo "   • claude-continue → Continue last conversation"
echo "   • claude-memory   → Check current memory usage"
echo "   • claude-check    → Check system memory before starting Claude"
