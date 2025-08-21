#!/bin/bash

echo "Testing Meta-Task Templates Configuration"
echo "========================================="
echo ""

# Test 1: Check templates in database
echo "1. Checking templates in database..."
TEMPLATES=$(PGPASSWORD=postgres psql -h localhost -p 5432 -U postgres -d llm_platform -t -c "SELECT COUNT(*) FROM meta_task_templates WHERE is_active = true;")
echo "   Active templates: $TEMPLATES"

# Test 2: Check template configuration
echo ""
echo "2. Checking template configuration..."
echo "   Strategic Business Plan:"
PHASES_BP=$(PGPASSWORD=postgres psql -h localhost -p 5432 -U postgres -d llm_platform -t -c "SELECT json_array_length(template_config->'phases') FROM meta_task_templates WHERE name='strategic_business_plan';")
echo "   - Number of phases: $PHASES_BP"

HAS_PROMPTS_BP=$(PGPASSWORD=postgres psql -h localhost -p 5432 -U postgres -d llm_platform -t -c "SELECT COUNT(*) FROM meta_task_templates WHERE name='strategic_business_plan' AND template_config->'phases'->0->>'prompt' IS NOT NULL;")
echo "   - Has prompts: $([ $HAS_PROMPTS_BP -eq 1 ] && echo 'Yes' || echo 'No')"

echo ""
echo "   Comprehensive Research Report:"
PHASES_RR=$(PGPASSWORD=postgres psql -h localhost -p 5432 -U postgres -d llm_platform -t -c "SELECT json_array_length(template_config->'phases') FROM meta_task_templates WHERE name='comprehensive_research_report';")
echo "   - Number of phases: $PHASES_RR"

HAS_PROMPTS_RR=$(PGPASSWORD=postgres psql -h localhost -p 5432 -U postgres -d llm_platform -t -c "SELECT COUNT(*) FROM meta_task_templates WHERE name='comprehensive_research_report' AND template_config->'phases'->0->>'prompt' IS NOT NULL;")
echo "   - Has prompts: $([ $HAS_PROMPTS_RR -eq 1 ] && echo 'Yes' || echo 'No')"

# Test 3: Check API endpoint
echo ""
echo "3. Testing API endpoint..."
RESPONSE=$(curl -s "http://localhost:8000/api/v1/meta-task/templates?active_only=true")
TEMPLATE_COUNT=$(echo "$RESPONSE" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('count', 0))")
echo "   Templates returned by API: $TEMPLATE_COUNT"

# Test 4: Check if templates have all required fields
echo ""
echo "4. Checking required fields..."
echo "   Strategic Business Plan phases:"
PGPASSWORD=postgres psql -h localhost -p 5432 -U postgres -d llm_platform -c "SELECT (template_config->'phases'->jsonb_array_length(template_config->'phases'::jsonb)-1)->>'name' as phase_name, (template_config->'phases'->jsonb_array_length(template_config->'phases'::jsonb)-1)->>'type' as phase_type, LENGTH((template_config->'phases'->jsonb_array_length(template_config->'phases'::jsonb)-1)->>'prompt') as prompt_length FROM meta_task_templates WHERE name='strategic_business_plan';" 2>/dev/null || echo "   - Error checking phases (JSON functions may not be available)"

# Test 5: Summary
echo ""
echo "5. Configuration Summary:"
echo "   ✓ Templates exist in database: $([ $TEMPLATES -gt 0 ] && echo 'YES' || echo 'NO')"
echo "   ✓ Templates have phases: $([ $PHASES_BP -gt 0 ] && [ $PHASES_RR -gt 0 ] && echo 'YES' || echo 'NO')"
echo "   ✓ Templates have prompts: $([ $HAS_PROMPTS_BP -eq 1 ] && [ $HAS_PROMPTS_RR -eq 1 ] && echo 'YES' || echo 'NO')"
echo "   ✓ API returns templates: $([ $TEMPLATE_COUNT -gt 0 ] && echo 'YES' || echo 'NO')"

echo ""
echo "========================================="
echo "Test complete!"