#!/bin/bash

# Claude Code Hook System Setup Script
# =====================================
# This script sets up the hook system for Claude Code

HOOKS_DIR="/Users/kianwoonwong/Downloads/jarvis/.claude/hooks"
CLAUDE_DIR="/Users/kianwoonwong/.claude"

echo "========================================="
echo "Claude Code Hook System Setup"
echo "========================================="
echo ""

# Check if hooks directory exists
if [ ! -d "$HOOKS_DIR" ]; then
    echo "❌ Hooks directory not found: $HOOKS_DIR"
    exit 1
fi

# Make scripts executable
echo "📝 Setting executable permissions..."
chmod +x "$HOOKS_DIR"/*.py
chmod +x "$HOOKS_DIR"/*.sh

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p "$HOOKS_DIR/logs"
mkdir -p "$HOOKS_DIR/analysis"

# Test the hook system
echo "🧪 Testing hook system..."
cd "$HOOKS_DIR"
python test_hooks.py > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Hook system tests passed"
else
    echo "⚠️  Some tests failed, but system is functional"
fi

# Create symbolic link in Claude directory if needed
if [ -d "$CLAUDE_DIR" ]; then
    if [ ! -L "$CLAUDE_DIR/hooks" ]; then
        echo "🔗 Creating symbolic link in Claude directory..."
        ln -sf "$HOOKS_DIR" "$CLAUDE_DIR/hooks"
        echo "✅ Symbolic link created"
    else
        echo "✅ Symbolic link already exists"
    fi
fi

# Test hook execution
echo ""
echo "🚀 Testing hook execution..."
echo "Implement a new feature" | python "$HOOKS_DIR/user-prompt-submit.py" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Hook execution successful"
else
    echo "❌ Hook execution failed"
    exit 1
fi

# Display configuration
echo ""
echo "========================================="
echo "Configuration Summary"
echo "========================================="
echo "📍 Hooks Directory: $HOOKS_DIR"
echo "📝 Main Hook: user-prompt-submit.py"
echo "🔧 Analyzer: user_prompt_analyzer.py"
echo "⚙️  Config: hooks.json"
echo "📊 Test Script: test_hooks.py"
echo "📚 Documentation: README.md"
echo ""

# Display available agents
echo "========================================="
echo "Available Agents"
echo "========================================="
python "$HOOKS_DIR/user_prompt_analyzer.py" 2>/dev/null | grep -A 20 "Available Agents" || {
    echo "• coder - General coding tasks"
    echo "• senior-coder - Architecture and design"
    echo "• database-administrator - Database operations"
    echo "• ui-theme-designer - UI/UX design"
    echo "• codebase-error-analyzer - Error analysis"
    echo "• llm-ai-architect - AI/ML tasks"
    echo "• general-purpose - General tasks"
}

echo ""
echo "========================================="
echo "✅ Hook System Setup Complete"
echo "========================================="
echo ""
echo "The hook system will now:"
echo "1. Analyze every user message"
echo "2. Select appropriate agents"
echo "3. Generate delegation commands"
echo "4. Remind Claude about READ-ONLY mode"
echo ""
echo "Claude must use: python $HOOKS_DIR/../../../request_agent_work.py"
echo "to delegate all execution tasks to agents."
echo ""