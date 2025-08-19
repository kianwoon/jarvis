#!/usr/bin/env python3
"""
Fix Hardcoded Prompts Migration Script
This script moves all hardcoded prompts from the codebase to the PostgreSQL settings table
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.db import SessionLocal, Settings as SettingsModel

def migrate_prompts_to_database():
    """Move all hardcoded prompts to the database settings table"""
    
    db = SessionLocal()
    try:
        # Define all prompt templates that should be in the database
        prompt_templates = {
            'synthesis_prompts': {
                'knowledge_base_synthesis': """Based on the search results below, provide a comprehensive answer to the user's question.

{enhanced_question}

📚 Internal Knowledge Base Results:
{documents_text}

Please provide a detailed, helpful response based on the information found in the knowledge base.""",
                'default_synthesis': """Based on the following information, provide a comprehensive answer to the user's question.

Question: {question}

Information:
{context}

Please provide a detailed and helpful response.""",
                'radiating_synthesis': """Based on the radiating coverage analysis below, provide a comprehensive answer.

Query: {query}

Radiating Coverage Analysis:
{analysis}

Please provide a thorough response incorporating the discovered relationships and patterns.""",
                'hybrid_synthesis': """User Question: {question}

{context}

Based on the above information, provide a comprehensive answer that synthesizes all available information."""
            },
            'formatting_templates': {
                'document_context': "**📄 {file_info}:**\n{content}",
                'search_result': "**Source:** {source}\n**Content:** {content}\n",
                'entity_reference': "[{entity_name}]({entity_type})",
            },
            'system_behaviors': {
                'answer_first_approach': True,
                'include_sources': True,
                'format_with_markdown': True,
                'max_source_attributions': 5,
                'include_synthesis_instructions': True
            }
        }
        
        # Check if synthesis_prompts settings already exist
        existing = db.query(SettingsModel).filter(
            SettingsModel.category == 'synthesis_prompts'
        ).first()
        
        if existing:
            print("Updating existing synthesis_prompts settings...")
            existing.settings = prompt_templates['synthesis_prompts']
        else:
            print("Creating new synthesis_prompts settings...")
            new_settings = SettingsModel(
                category='synthesis_prompts',
                settings=prompt_templates['synthesis_prompts']
            )
            db.add(new_settings)
        
        # Add formatting templates
        formatting_existing = db.query(SettingsModel).filter(
            SettingsModel.category == 'formatting_templates'
        ).first()
        
        if formatting_existing:
            print("Updating existing formatting_templates settings...")
            formatting_existing.settings = prompt_templates['formatting_templates']
        else:
            print("Creating new formatting_templates settings...")
            new_formatting = SettingsModel(
                category='formatting_templates',
                settings=prompt_templates['formatting_templates']
            )
            db.add(new_formatting)
        
        # Add system behaviors
        behaviors_existing = db.query(SettingsModel).filter(
            SettingsModel.category == 'system_behaviors'
        ).first()
        
        if behaviors_existing:
            print("Updating existing system_behaviors settings...")
            behaviors_existing.settings = prompt_templates['system_behaviors']
        else:
            print("Creating new system_behaviors settings...")
            new_behaviors = SettingsModel(
                category='system_behaviors',
                settings=prompt_templates['system_behaviors']
            )
            db.add(new_behaviors)
        
        db.commit()
        print("✅ Successfully migrated all prompts to database!")
        
        # Clear Redis cache to force reload
        try:
            from app.core.redis_client import get_redis_client
            redis_client = get_redis_client()
            if redis_client:
                # Clear all settings caches
                for key in ['synthesis_prompts_cache', 'formatting_templates_cache', 
                           'system_behaviors_cache', 'llm_settings_cache']:
                    redis_client.delete(key)
                print("✅ Cleared Redis cache for settings")
        except Exception as e:
            print(f"⚠️ Could not clear Redis cache: {e}")
        
    except Exception as e:
        print(f"❌ Error migrating prompts: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    print("Starting prompt migration to database...")
    migrate_prompts_to_database()
    print("Migration complete!")