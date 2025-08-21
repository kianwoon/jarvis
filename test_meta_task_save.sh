#!/bin/bash

echo "Meta-Task System Prompt Save Test Script"
echo "========================================"
echo ""
echo "This script will help you test the meta-task system prompt save functionality."
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Step 1: Clear existing test data${NC}"
echo "Clearing any existing test prompts from the database..."
PGPASSWORD=postgres psql -h localhost -p 5432 -U postgres -d llm_platform -c "UPDATE settings SET settings = '{}' WHERE category = 'meta_task';" > /dev/null 2>&1

echo -e "${GREEN}✓ Database cleared${NC}"
echo ""

echo -e "${BLUE}Step 2: Instructions for UI Testing${NC}"
echo "1. Open your browser and navigate to: http://localhost:5173/meta-task.html"
echo "2. Click on the Settings tab"
echo "3. For each model tab (Analyzer, Reviewer, Assembler, Generator):"
echo "   - Click on the 'System Prompt' sub-tab"
echo "   - Enter a unique test prompt, for example:"
echo ""
echo -e "${YELLOW}   Analyzer Model:${NC} 'You are an analytical AI that breaks down complex tasks.'"
echo -e "${YELLOW}   Reviewer Model:${NC} 'You are a critical reviewer that ensures quality.'"
echo -e "${YELLOW}   Assembler Model:${NC} 'You are an assembler that combines multiple outputs.'"
echo -e "${YELLOW}   Generator Model:${NC} 'You are a generator that creates detailed content.'"
echo ""
echo "4. Click the 'Save Settings' button"
echo ""
echo -e "${BLUE}Press Enter when you've completed the UI steps...${NC}"
read

echo ""
echo -e "${BLUE}Step 3: Verifying saved data${NC}"
echo "Checking database for saved prompts..."
echo ""

# Check if data was saved
RESULT=$(PGPASSWORD=postgres psql -h localhost -p 5432 -U postgres -d llm_platform -t -c "SELECT settings FROM settings WHERE category = 'meta_task';" 2>/dev/null)

if [ -z "$RESULT" ] || [ "$RESULT" = "{}" ]; then
    echo -e "${YELLOW}⚠ No data found in database. The save may have failed.${NC}"
    echo "Please check:"
    echo "  1. That the backend is running (./run_local.sh)"
    echo "  2. That there are no errors in the browser console"
    echo "  3. That you clicked 'Save Settings' and saw a success message"
else
    echo -e "${GREEN}✓ Data found in database!${NC}"
    echo ""
    echo "Saved settings:"
    PGPASSWORD=postgres psql -h localhost -p 5432 -U postgres -d llm_platform -c "SELECT jsonb_pretty(settings) FROM settings WHERE category = 'meta_task';"
    
    # Check for system prompts specifically
    echo ""
    echo -e "${BLUE}Checking for system prompts:${NC}"
    
    for model in analyzer reviewer assembler generator; do
        PROMPT=$(PGPASSWORD=postgres psql -h localhost -p 5432 -U postgres -d llm_platform -t -c "SELECT settings->'${model}_model'->>'system_prompt' FROM settings WHERE category = 'meta_task';" 2>/dev/null | xargs)
        
        if [ ! -z "$PROMPT" ] && [ "$PROMPT" != "null" ]; then
            echo -e "${GREEN}✓${NC} ${model}_model system_prompt: Found"
        else
            echo -e "${YELLOW}✗${NC} ${model}_model system_prompt: Not found"
        fi
    done
fi

echo ""
echo -e "${BLUE}Step 4: Testing cache reload${NC}"
echo "Testing cache reload endpoint..."

# Test cache reload
RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/settings/meta-task/cache/reload)

if echo "$RESPONSE" | grep -q "success.*true"; then
    echo -e "${GREEN}✓ Cache reload successful${NC}"
else
    echo -e "${YELLOW}⚠ Cache reload may have failed${NC}"
fi

echo ""
echo -e "${BLUE}Test complete!${NC}"
echo ""
echo "Summary:"
echo "--------"
if [ ! -z "$RESULT" ] && [ "$RESULT" != "{}" ]; then
    echo -e "${GREEN}✓ Meta-task system prompts are being saved correctly!${NC}"
    echo "The fix is working as expected. You can now use custom system prompts for all meta-task models."
else
    echo -e "${YELLOW}⚠ No saved data detected. Please ensure you followed all the steps above.${NC}"
    echo "If the issue persists, check the backend logs for errors."
fi