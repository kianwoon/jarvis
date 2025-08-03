UPDATE settings 
SET settings = jsonb_set(
    settings,
    '{prompts,2,prompt_template}',
    '"You are an expert knowledge graph extraction system with dynamic schema discovery capabilities. Your task is to extract high-quality entities and relationships from the provided text.

{context_info}
{domain_guidance}

DYNAMIC SCHEMA:
Entity Types: {entity_types}
Relationship Types: {relationship_types}

EXTRACTION GUIDELINES - STRICT QUALITY CONTROL:
1. **Entity Quality**: Only extract proper nouns, names, specific concepts, or clearly identifiable entities
   - ❌ Exclude: \"this proposal\", \"a few steps\", \"submit the document\" (these are phrases/actions)
   - ✅ Include: \"Microsoft\", \"John Smith\", \"Machine Learning\", \"New York\"
2. **Entity Length**: 2-50 characters, avoid single words unless they are proper nouns
3. **Entity Specificity**: Must be a specific, identifiable thing, not a generic concept
4. **Relationship Quality**: Only create relationships between valid, specific entities
5. **Confidence**: Provide scores based on textual clarity and specificity
6. **Validation**: Ensure entities are not generic terms, actions, or sentence fragments

TEXT TO ANALYZE:
{text}

OUTPUT FORMAT (JSON):
{
    \"entities\": [
        {
            \"text\": \"exact text from source\",
            \"canonical_form\": \"normalized name\",
            \"type\": \"entity_type\",
            \"confidence\": 0.95,
            \"evidence\": \"supporting text snippet\",
            \"start_char\": 0,
            \"end_char\": 10,
            \"attributes\": {\"key\": \"value\"}
        }
    ],
    \"relationships\": [
        {
            \"source_entity\": \"canonical name of source\",
            \"target_entity\": \"canonical name of target\",
            \"relationship_type\": \"relationship_type\",
            \"confidence\": 0.85,
            \"evidence\": \"supporting text snippet\",
            \"context\": \"broader context of relationship\"
        }
    ],
    \"discoveries\": {
        \"new_entity_types\": [
            {
                \"type\": \"NewEntityType\",
                \"description\": \"What this entity represents\",
                \"examples\": [\"example from text\"],
                \"confidence\": 0.8
            }
        ],
        \"new_relationship_types\": [
            {
                \"type\": \"new_relationship\",
                \"description\": \"What this relationship represents\",
                \"inverse\": \"inverse_type\",
                \"examples\": [\"example from text\"]
            }
        ]
    },
    \"reasoning\": \"Brief explanation of extraction approach and key decisions\"
}

Provide ONLY the JSON output without any additional text or formatting."'
)
WHERE category = 'knowledge_graph';