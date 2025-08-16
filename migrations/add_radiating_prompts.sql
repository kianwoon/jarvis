-- Migration to add radiating prompts to settings table
-- This migration moves all hardcoded LLM prompts from the radiating system to database settings

-- First, ensure the settings table exists and has the required structure
CREATE TABLE IF NOT EXISTS settings (
    id SERIAL PRIMARY KEY,
    category VARCHAR(255) NOT NULL,
    settings JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(category)
);

-- Insert or update radiating settings with all prompts
INSERT INTO settings (category, settings, updated_at)
VALUES (
    'radiating',
    '{
        "enabled": true,
        "default_depth": 3,
        "max_depth": 5,
        "relevance_threshold": 0.7,
        "expansion_strategy": "adaptive",
        "cache_ttl": 3600,
        "traversal_strategy": "hybrid",
        "max_entities_per_hop": 10,
        "relationship_weight_threshold": 0.5,
        
        "prompts": {
            "entity_extraction": {
                "discovery_comprehensive": "You are analyzing a query about MODERN LLM-ERA technologies (2023-2024 era).\nFocus on technologies actually used for building LLM-powered applications TODAY.\nIdentify ALL relevant entity types for a COMPREHENSIVE extraction {domain_context}.\n\nQuery/Text: {text}\n\nCRITICAL: Focus on MODERN AI/LLM technologies, NOT legacy ML tools!\nModern means: LangChain, Ollama, vector databases, LLM frameworks, RAG systems\nNOT legacy: scikit-learn, Keras (outdated for LLM apps)\n\nReturn as JSON with MANY specific entity types:\n{{\n    \"entity_types\": [\n        {{\n            \"type\": \"LLMFramework\",\n            \"description\": \"Modern frameworks for building LLM applications\",\n            \"examples\": [\"LangChain\", \"LlamaIndex\", \"Haystack\", \"Semantic Kernel\", \"DSPy\"],\n            \"confidence\": 0.95\n        }},\n        {{\n            \"type\": \"VectorDatabase\",\n            \"description\": \"Vector databases for embeddings and similarity search\",\n            \"examples\": [\"Milvus\", \"Qdrant\", \"Weaviate\", \"Pinecone\", \"Chroma\", \"FAISS\"],\n            \"confidence\": 0.95\n        }},\n        ... (continue with more modern LLM-era types)\n    ]\n}}\n\nFocus on technologies from 2023-2024 that are actively used in modern LLM applications.",
                
                "discovery_regular": "Analyze this text and identify the types of entities present {domain_context}.\nDo not use generic types like \"Person\", \"Organization\", \"Location\" unless they truly fit.\nInstead, discover domain-specific entity types that best describe the content.\n\nText: {text}\n\nReturn as JSON:\n{{\n    \"entity_types\": [\n        {{\n            \"type\": \"SpecificEntityType\",\n            \"description\": \"What this type represents\",\n            \"examples\": [\"example1\", \"example2\"],\n            \"confidence\": 0.8\n        }},\n        ...\n    ]\n}}\n\nBe creative and specific. For example:\n- Instead of \"Person\", use \"Researcher\", \"CEO\", \"Author\"\n- Instead of \"Organization\", use \"TechCompany\", \"University\", \"Government Agency\"\n- Instead of \"Concept\", use \"MachineLearningModel\", \"BusinessStrategy\", \"ScientificTheory\"",
                
                "extraction_comprehensive": "You are an expert on MODERN LLM-ERA technologies (2023-2024 era).\nThe user is asking for a COMPREHENSIVE list of entities related to their query.\n\nCRITICAL CONTEXT: Focus on technologies used for building LLM-powered applications TODAY!\nModern means: LangChain, Ollama, vector databases, LLM frameworks, RAG systems\nNOT legacy: scikit-learn, Keras (these are outdated for LLM apps in 2024)\n\n{domain_context}\n{additional_context}\n\nQuery/Text: {text}\n\nCRITICAL INSTRUCTIONS:\n1. Extract EVERY SINGLE relevant MODERN LLM-era technology, tool, framework, library, platform, or system\n2. Focus on technologies from 2023-2024 that are actively used for LLM applications\n3. Include both popular AND niche modern options\n4. For AI/LLM queries: Focus on LLM frameworks (LangChain, LlamaIndex), vector DBs (Milvus, Qdrant), inference engines (Ollama, vLLM)\n5. For open source preference: Prioritize open source LLMs (Llama, Mistral, Qwen) and tools\n6. Target 30-50+ entities minimum for technology queries\n7. Include entities at different levels: LLM frameworks, vector databases, inference engines, prompt tools, RAG systems\n\nReturn as JSON array with AT LEAST 30 entities (aim for 50+):\n[\n    {{\n        \"text\": \"LangChain\",\n        \"type\": \"LLMFramework\",\n        \"confidence\": 0.95,\n        \"context\": \"modern framework for building LLM applications\",\n        \"reason\": \"leading framework for LLM app development with chains and agents\"\n    }},\n    ... (continue with MANY more MODERN LLM-era entities)\n]\n\nRemember:\n- Extract 30-50+ MODERN entities minimum\n- Focus on 2023-2024 LLM technologies, NOT legacy ML tools\n- Include: LLM frameworks, vector DBs, inference engines, open source LLMs, RAG tools\n- Avoid outdated: scikit-learn, Keras, traditional ML libraries (unless specifically relevant)\n- Use your knowledge of CURRENT LLM ecosystem",
                
                "extraction_regular": "Extract all important entities from this text.\n{domain_context}\n{additional_context}\n{type_guidance}\n\nText: {text}\n\nReturn as JSON array:\n[\n    {{\n        \"text\": \"exact entity text\",\n        \"type\": \"specific entity type\",\n        \"confidence\": 0.8,\n        \"context\": \"surrounding context\",\n        \"reason\": \"why this is an important entity\"\n    }},\n    ...\n]\n\nGuidelines:\n1. Extract entities that are central to understanding the content\n2. Use specific, descriptive entity types\n3. Include confidence scores based on clarity and importance\n4. Preserve exact text as it appears\n5. Don''t extract common words or pronouns"
            },
            
            "relationship_discovery": {
                "llm_discovery": "You are an expert in technology, AI, ML, software systems, cloud computing, databases, and business relationships.\n\nAnalyze these entities and discover ALL meaningful relationships between them based on your comprehensive knowledge:\n\nENTITIES:\n{entity_list}\n\nRELATIONSHIP TYPES TO USE (MUST use these specific types):\n{relationship_types}\n\nCRITICAL INSTRUCTIONS:\n1. Be EXHAUSTIVE - find EVERY real relationship you know about these entities\n2. Use ONLY the specific relationship types listed above (absolutely NO generic \"RELATED_TO\")\n3. Think about multiple aspects:\n   - Technical integrations and APIs\n   - Dependencies and requirements\n   - Competition and alternatives\n   - Business relationships (ownership, partnerships)\n   - Implementation details (what runs on what, what uses what)\n   - Data flow (what stores in what, what queries what)\n4. Include relationships in BOTH directions when different (e.g., A USES B, B USED_BY A)\n5. Confidence scores: 1.0 for certain facts, 0.8 for likely, 0.6 for probable\n6. Provide specific technical context for each relationship\n\nReturn a JSON object with comprehensive relationships:\n{{\n    \"relationships\": [\n        {{\n            \"source\": \"Entity Name 1\",\n            \"target\": \"Entity Name 2\",\n            \"type\": \"RELATIONSHIP_TYPE\",\n            \"confidence\": 0.9,\n            \"context\": \"Specific technical or business context\",\n            \"bidirectional\": false\n        }}\n    ]\n}}\n\nExample comprehensive discovery for [LangChain, OpenAI, Pinecone]:\n- LangChain INTEGRATES_WITH OpenAI (Native OpenAI LLM integration via langchain.llms)\n- LangChain INTEGRATES_WITH Pinecone (Vector store integration via langchain.vectorstores)\n- LangChain DEPENDS_ON Python (Written in Python, requires Python 3.8+)\n- OpenAI COMPETES_WITH Anthropic (Both provide LLM APIs)\n- Pinecone COMPETES_WITH Milvus (Both are vector databases)\n- LangChain USES OpenAI (Can use OpenAI models for chains)\n- LangChain USES Pinecone (Can use Pinecone for embeddings storage)\n- OpenAI PROVIDES GPT-4 (Provides GPT model family)\n- Pinecone STORES_IN AWS (Runs on AWS infrastructure)\n\nFor the given entities, discover ALL relationships comprehensively. Aim for 50+ relationships if dealing with 20-30 entities.",
                
                "relationship_analysis": "Analyze the relationships between these entities {domain_context}:\n\nEntities:\n{entity_list}\n\nText:\n{text}\n\nDiscover all meaningful relationships between the entities.\nBe specific about relationship types - avoid generic \"RELATED_TO\".\n\nReturn as JSON array:\n[\n    {{\n        \"source_index\": 1,\n        \"target_index\": 2,\n        \"relationship_type\": \"SPECIFIC_RELATIONSHIP\",\n        \"confidence\": 0.8,\n        \"bidirectional\": false,\n        \"context\": \"supporting text snippet\",\n        \"reasoning\": \"why this relationship exists\"\n    }},\n    ...\n]\n\nGuidelines:\n1. Use specific, meaningful relationship types\n2. Include confidence based on evidence strength\n3. Mark bidirectional relationships appropriately\n4. Extract context that supports the relationship\n5. Focus on important, non-trivial relationships",
                
                "implicit_relationships": "Analyze these entities and infer implicit relationships {domain_context}:\n\n{entities_json}\n\nInfer relationships based on:\n1. Entity types (e.g., all people might work at same company)\n2. Naming patterns (e.g., similar names might indicate versions)\n3. Logical connections (e.g., cause and effect)\n4. Domain knowledge\n\nReturn as JSON array of implicit relationships.\nOnly include high-confidence inferences."
            },
            
            "query_analysis": {
                "entity_extraction": "Extract key entities from the following query. For each entity, identify:\n1. The entity text\n2. The entity type (Person, Organization, Location, Concept, Event, etc.)\n3. Confidence score (0.0 to 1.0)\n\nQuery: {query}\n\nReturn as JSON array:\n[\n    {{\"text\": \"entity_text\", \"type\": \"entity_type\", \"confidence\": 0.8}},\n    ...\n]\n\nFocus on important entities that would be useful for knowledge exploration.",
                
                "intent_identification": "Identify the primary intent of this query. Choose from:\n- EXPLORATION: Broad discovery queries\n- CONNECTION_FINDING: Finding relationships between things\n- COMPREHENSIVE: Deep, thorough information gathering\n- SPECIFIC: Targeted, narrow queries\n- COMPARISON: Comparing multiple entities\n- TEMPORAL: Time-based queries\n- CAUSAL: Cause-effect relationships\n- HIERARCHICAL: Parent-child or part-whole relationships\n\nQuery: {query}\nEntities found: {entities}\n\nReturn only the intent type name.",
                
                "domain_extraction": "Identify the knowledge domains relevant to this query.\nExamples: Technology, Business, Science, Medicine, Finance, Education, etc.\n\nQuery: {query}\n\nReturn as JSON array of domain names (max 5):\n[\"domain1\", \"domain2\", ...]",
                
                "temporal_extraction": "Extract temporal context from this query if present.\nLook for: time periods, dates, relative times (latest, recent, current), etc.\n\nQuery: {query}\n\nReturn as JSON or null if no temporal context:\n{{\n    \"type\": \"absolute|relative|range\",\n    \"value\": \"extracted time reference\",\n    \"parsed\": \"standardized format if applicable\"\n}}"
            },
            
            "expansion_strategy": {
                "semantic_expansion": "Find semantically related terms and entities for:\nEntity: {entity}\nType: {entity_type}\n{domain_context}\n\nReturn as JSON:\n{{\n    \"terms\": [\"related_term1\", \"related_term2\", ...],\n    \"entities\": [\n        {{\"text\": \"entity_name\", \"type\": \"entity_type\", \"relationship\": \"how_related\"}},\n        ...\n    ]\n}}\n\nInclude synonyms, related concepts, and associated entities.\nLimit to 5 terms and 5 entities.",
                
                "concept_expansion": "Identify related concepts and topics for this query:\nQuery: {query}\n{domain_context}\n\nReturn as JSON array of related concepts (max 5):\n[\"concept1\", \"concept2\", ...]\n\nFocus on conceptually related topics that would help explore the subject.",
                
                "hierarchical_expansion": "Find hierarchical relationships for:\nEntity: {entity}\nType: {entity_type}\n\nReturn as JSON:\n{{\n    \"parents\": [\"broader_concept1\", ...],\n    \"children\": [\"narrower_concept1\", ...],\n    \"siblings\": [\"related_concept1\", ...]\n}}\n\nLimit to 3 items per category."
            }
        },
        
        "query_expansion": {
            "enabled": true,
            "max_expansions": 5,
            "confidence_threshold": 0.6,
            "preserve_context": true,
            "expansion_method": "semantic",
            "intent_detection": true,
            "domain_hints": true,
            "synonym_expansion": true,
            "concept_expansion": true,
            "temporal_expansion": false,
            "geographic_expansion": false,
            "hierarchical_expansion": true,
            "cross_domain_expansion": false
        },
        
        "extraction": {
            "entity_confidence_threshold": 0.6,
            "relationship_confidence_threshold": 0.65,
            "enable_universal_discovery": true,
            "max_entities_per_query": 20,
            "max_relationships_per_query": 30,
            "enable_pattern_detection": true,
            "enable_semantic_inference": true,
            "enable_context_preservation": true,
            "bidirectional_relationships": true,
            "extract_implicit_relationships": false,
            "extract_temporal_context": true,
            "extract_spatial_context": false,
            "extract_causal_relationships": true,
            "extract_hierarchical_relationships": true,
            "extract_part_whole_relationships": true,
            "extract_comparison_relationships": false
        }
    }'::jsonb
)
ON CONFLICT (category) 
DO UPDATE SET 
    settings = EXCLUDED.settings,
    updated_at = CURRENT_TIMESTAMP
WHERE settings.category = 'radiating';

-- Add index for faster queries
CREATE INDEX IF NOT EXISTS idx_settings_category ON settings(category);

-- Function to reload radiating prompts (can be called from application)
CREATE OR REPLACE FUNCTION reload_radiating_prompts()
RETURNS void AS $$
BEGIN
    -- This function can be called to trigger a cache refresh
    -- The actual refresh is handled by the application
    NOTIFY radiating_settings_changed;
END;
$$ LANGUAGE plpgsql;

-- Grant necessary permissions
GRANT SELECT, UPDATE ON settings TO PUBLIC;

-- Add comment to document the migration
COMMENT ON TABLE settings IS 'Stores application settings including LLM prompts for various subsystems';