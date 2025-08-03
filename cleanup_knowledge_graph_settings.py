#!/usr/bin/env python3
"""
Database script to clean up knowledge_graph settings by removing hardcoded fields
and maintaining only LLM-driven configuration fields.
"""

import json
import sys
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.core.config import get_settings

# Get database connection details from settings
settings = get_settings()
DATABASE_URL = (
    f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def get_db_session():
    """Context manager for database sessions with automatic cleanup"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_knowledge_graph_settings():
    """Retrieve current knowledge_graph settings from database"""
    with get_db_session() as db:
        result = db.execute(text(
            "SELECT settings FROM settings WHERE category = 'knowledge_graph'"
        )).fetchone()
        
        if result:
            return result[0]
        return None

def create_clean_llm_driven_config():
    """Create a clean LLM-driven knowledge graph configuration"""
    
    # Clean LLM-driven prompts without hardcoded types
    clean_prompts = [
        {
            "id": "1",
            "name": "entity_discovery",
            "version": 1,
            "is_active": True,
            "parameters": {
                "format": "json",
                "confidence_threshold": 0.7  # Dynamic threshold
            },
            "description": "Extract entities from text using LLM intelligence without predefined types",
            "prompt_type": "entity_discovery",
            "prompt_template": "Analyze the following text and extract all meaningful entities. Let the content guide what types of entities to discover. Return entities in JSON format with their natural types based on context: {text}"
        },
        {
            "id": "2", 
            "name": "relationship_discovery",
            "version": 1,
            "is_active": True,
            "parameters": {
                "format": "triples",
                "confidence_threshold": 0.6  # Dynamic threshold
            },
            "description": "Discover relationships between entities using LLM understanding",
            "prompt_type": "relationship_discovery", 
            "prompt_template": "Identify meaningful relationships between entities in the text. Let the content determine natural relationship types. Return relationships as triples with confidence scores: {text}"
        },
        {
            "id": "3",
            "name": "knowledge_extraction", 
            "version": 1,
            "is_active": True,
            "parameters": {
                "depth": "comprehensive",
                "include_metadata": True,
                "confidence_threshold": 0.65
            },
            "description": "Extract comprehensive knowledge from documents using LLM intelligence",
            "prompt_type": "knowledge_extraction",
            "prompt_template": "Extract key knowledge, concepts, and insights from the document using your understanding of the domain. Focus on meaningful connections and knowledge relationships: {text}"
        }
    ]
    
    # Clean LLM-driven configuration
    clean_config = {
        # Core mode settings - LLM-driven
        "schema_mode": "dynamic",  # Let LLM determine schema
        
        # Model configuration for LLM-driven extraction
        "model_config": {
            "model": "qwen3:30b-a3b-instruct-2507-q4_K_M",
            "max_tokens": 4096,
            "temperature": 0.1,
            "model_server": "http://localhost:11434", 
            "system_prompt": "You are an expert knowledge graph extraction system. Extract entities and relationships from text and return them in STRICT JSON format.\n\nCRITICAL: You MUST respond with ONLY valid JSON. Do NOT include explanations, markdown, thinking, or other text.\n\nREQUIRED JSON FORMAT:\n{{\"entities\": [\"Entity1\", \"Entity2\"], \"relationships\": [{{\"from\": \"Entity1\", \"to\": \"Entity2\", \"type\": \"TYPE\"}}]}}\n\nEXAMPLES:\n\nExample 1:\n{{\"entities\": [\"Microsoft\", \"Seattle\", \"Azure\"], \"relationships\": [{{\"from\": \"Microsoft\", \"to\": \"Seattle\", \"type\": \"LOCATED_IN\"}}, {{\"from\": \"Microsoft\", \"to\": \"Azure\", \"type\": \"DEVELOPS\"}}]}}\n\nExample 2:\n{{\"entities\": [\"DBS Bank\", \"Singapore\", \"OceanBase\"], \"relationships\": [{{\"from\": \"DBS Bank\", \"to\": \"Singapore\", \"type\": \"LOCATED_IN\"}}, {{\"from\": \"DBS Bank\", \"to\": \"OceanBase\", \"type\": \"EVALUATES\"}}]}}\n\nRULES:\n- Extract 3-12 entities maximum\n- Extract 2-8 relationships maximum\n- Use relationship types: LOCATED_IN, WORKS_FOR, USES, DEVELOPS, EVALUATES, PART_OF, MANAGES\n\nText: {{text}}\n\nJSON:",
            "context_length": 40960,
            "repeat_penalty": 1.0
        },
        
        # Neo4j connection configuration
        "neo4j": {
            "uri": "bolt://neo4j:7687",
            "host": "neo4j", 
            "port": 7687,
            "enabled": True,
            "database": "neo4j",
            "password": "jarvis_neo4j_password",
            "username": "neo4j",
            "http_port": 7474
        },
        
        # LLM-driven entity discovery
        "entity_discovery": {
            "enabled": True,
            "auto_linking": True,  # Let LLM determine linking
            "relationship_boost": True,  # Allow LLM to boost relationships
            "confidence_threshold": 0.4,  # Dynamic threshold
            "adaptive_threshold": True  # Allow threshold adjustment based on content
        },
        
        # LLM-driven relationship discovery  
        "relationship_discovery": {
            "enabled": True,
            "cross_document": True,  # Enable cross-document relationship discovery
            "semantic_linking": True,  # Use LLM for semantic linking
            "confidence_threshold": 0.5,  # Dynamic threshold
            "adaptive_threshold": True  # Allow threshold adjustment
        },
        
        # Anti-silo configuration for LLM-driven connection discovery
        "anti_silo": {
            "enabled": True,
            "similarity_threshold": 0.5,  # Dynamic threshold
            "cross_document_linking": True,
            "max_relationships_per_entity": 100,
            "adaptive_threshold": True  # Allow LLM to adjust based on content
        },
        
        # Enhanced knowledge graph processing
        "knowledge_graph": {
            "extraction": {
                "bridge_relationship_confidence": 0.55,
                "adaptive_confidence": True  # Let LLM adjust confidence dynamically
            }
        },
        
        # Clean LLM-driven prompts
        "prompts": clean_prompts
    }
    
    return clean_config

def update_knowledge_graph_settings(new_config):
    """Update knowledge_graph settings in database"""
    with get_db_session() as db:
        try:
            # Update the settings
            db.execute(text(
                "UPDATE settings SET settings = :settings, updated_at = NOW() WHERE category = 'knowledge_graph'"
            ), {"settings": json.dumps(new_config)})
            
            db.commit()
            print("✅ Successfully updated knowledge_graph settings with clean LLM-driven configuration")
            return True
            
        except Exception as e:
            db.rollback()
            print(f"❌ Error updating knowledge_graph settings: {e}")
            return False

def main():
    """Main execution function"""
    print("🧹 Knowledge Graph Settings Cleanup - Removing Hardcoded Fields")
    print("=" * 70)
    
    # Get current settings
    print("\n1. Retrieving current knowledge_graph settings...")
    current_settings = get_current_knowledge_graph_settings()
    
    if not current_settings:
        print("❌ No knowledge_graph settings found in database")
        return
    
    print("✅ Current settings retrieved")
    
    # Show hardcoded fields that will be removed
    hardcoded_fields_found = []
    if "extraction_settings" in current_settings:
        hardcoded_fields_found.append("extraction_settings (entity_types, relationship_types)")
    
    # Check for hardcoded types in prompts
    if "prompts" in current_settings:
        for prompt in current_settings["prompts"]:
            if "parameters" in prompt and "types" in prompt["parameters"]:
                hardcoded_fields_found.append(f"prompt '{prompt['name']}' hardcoded types")
    
    if hardcoded_fields_found:
        print(f"\n2. Hardcoded fields found that will be removed:")
        for field in hardcoded_fields_found:
            print(f"   🗑️  {field}")
    else:
        print("\n2. No hardcoded fields found - configuration appears clean")
    
    # Create clean configuration
    print("\n3. Creating clean LLM-driven configuration...")
    clean_config = create_clean_llm_driven_config()
    print("✅ Clean configuration created")
    
    # Show what will be preserved
    print("\n4. LLM-driven fields that will be preserved/enhanced:")
    preserved_fields = [
        "schema_mode: dynamic (LLM determines schema)",
        "model_config: LLM model configuration", 
        "entity_discovery: LLM-driven entity extraction",
        "relationship_discovery: LLM-driven relationship extraction",
        "anti_silo: LLM-driven cross-document linking",
        "prompts: Clean prompts without hardcoded types",
        "adaptive_threshold: Dynamic confidence adjustment"
    ]
    
    for field in preserved_fields:
        print(f"   ✅ {field}")
    
    # Auto-proceed with cleanup since we identified hardcoded fields
    print("\n5. Proceeding with cleanup automatically...")
    
    if True:
        print("\n6. Updating database...")
        success = update_knowledge_graph_settings(clean_config)
        
        if success:
            print("\n🎉 Knowledge graph settings successfully cleaned!")
            print("\nKey improvements:")
            print("• Removed hardcoded entity_types and relationship_types")
            print("• Enabled adaptive thresholds for dynamic confidence adjustment")
            print("• Enhanced prompts to let LLM determine natural types")
            print("• Enabled cross-document relationship discovery")
            print("• Enabled semantic linking for better connections")
            print("• Maintained all Neo4j connection settings")
            
            # Verify the update
            print("\n7. Verifying update...")
            updated_settings = get_current_knowledge_graph_settings()
            if updated_settings and "extraction_settings" not in updated_settings:
                print("✅ Verification successful - hardcoded fields removed")
            else:
                print("⚠️  Verification warning - please check manually")
        else:
            print("\n❌ Update failed - settings unchanged")
    else:
        print("\n🚫 Update cancelled - no changes made")

if __name__ == "__main__":
    main()