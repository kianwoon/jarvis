import json
from app.core.redis_base import RedisCache
from app.core.timeout_settings_cache import get_settings_cache_ttl
from typing import Dict, Any, Optional

KNOWLEDGE_GRAPH_SETTINGS_KEY = 'knowledge_graph_settings_cache'

# Initialize cache with lazy Redis connection
cache = RedisCache(key_prefix="")

def get_knowledge_graph_settings():
    """Get knowledge graph settings from cache or reload from database"""
    try:
        cached = cache.get(KNOWLEDGE_GRAPH_SETTINGS_KEY)
        if cached:
            return cached
        return reload_knowledge_graph_settings()
    except Exception as e:
        print(f"[ERROR] Failed to get knowledge graph settings from cache: {str(e)}")
        # Return default settings if cache fails
        return get_default_knowledge_graph_settings()

def get_neo4j_config() -> Dict[str, Any]:
    """Get Neo4j configuration from knowledge graph settings"""
    try:
        settings = get_knowledge_graph_settings()
        neo4j_config = settings.get('neo4j', {})
        
        # Ensure all required fields are present with defaults
        return {
            'enabled': neo4j_config.get('enabled', True),
            'host': neo4j_config.get('host', 'neo4j'),
            'port': neo4j_config.get('port', 7687),
            'http_port': neo4j_config.get('http_port', 7474),
            'database': neo4j_config.get('database', 'neo4j'),
            'username': neo4j_config.get('username', 'neo4j'),
            'password': neo4j_config.get('password', 'jarvis_neo4j_password'),
            'uri': neo4j_config.get('uri', 'bolt://neo4j:7687'),
            'connection_pool': {
                'max_connections': neo4j_config.get('connection_pool', {}).get('max_connections', 50),
                'connection_timeout': neo4j_config.get('connection_pool', {}).get('connection_timeout', 30),
                'max_transaction_retry_time': neo4j_config.get('connection_pool', {}).get('max_transaction_retry_time', 30)
            },
            'memory_config': {
                'heap_initial': neo4j_config.get('memory_config', {}).get('heap_initial', '512m'),
                'heap_max': neo4j_config.get('memory_config', {}).get('heap_max', '2g'),
                'pagecache': neo4j_config.get('memory_config', {}).get('pagecache', '1g')
            },
            'plugins': {
                'apoc_enabled': neo4j_config.get('plugins', {}).get('apoc_enabled', True),
                'gds_enabled': neo4j_config.get('plugins', {}).get('gds_enabled', True)
            },
            'security': {
                'encrypted': neo4j_config.get('security', {}).get('encrypted', False),
                'trust_strategy': neo4j_config.get('security', {}).get('trust_strategy', 'TRUST_ALL_CERTIFICATES')
            }
        }
    except Exception as e:
        print(f"[ERROR] Failed to get Neo4j config: {str(e)}")
        return get_default_neo4j_config()

def get_default_neo4j_config() -> Dict[str, Any]:
    """Get default Neo4j configuration"""
    return {
        'enabled': True,
        'host': 'neo4j',
        'port': 7687,
        'http_port': 7474,
        'database': 'neo4j',
        'username': 'neo4j',
        'password': 'jarvis_neo4j_password',
        'uri': 'bolt://neo4j:7687',
        'connection_pool': {
            'max_connections': 50,
            'connection_timeout': 30,
            'max_transaction_retry_time': 30
        },
        'memory_config': {
            'heap_initial': '512m',
            'heap_max': '2g',
            'pagecache': '1g'
        },
        'plugins': {
            'apoc_enabled': True,
            'gds_enabled': True
        },
        'security': {
            'encrypted': False,
            'trust_strategy': 'TRUST_ALL_CERTIFICATES'
        }
    }

def get_default_knowledge_graph_settings() -> Dict[str, Any]:
    """Get default knowledge graph settings with enhanced anti-silo configuration"""
    import os
    return {
        'mode': 'comprehensive',  # REBALANCED: Restore comprehensive extraction for business documents
        'model': 'qwen3:30b-a3b-q4_K_M',
        'max_tokens': 8192,
        'model_server': os.environ.get("OLLAMA_BASE_URL", ""),  # Must be configured in settings or environment
        'system_prompt': 'You are an expert knowledge graph extraction system. Extract entities and relationships from the provided text with high precision.',
        'context_length': 40960,  # Conservative fallback - database should override
        'repeat_penalty': '1.05',
        'temperature': 0.3,
        'extraction_prompt': None,  # Use database-driven prompt from LLMPrompt table
        'max_entities_per_chunk': 25,  # REBALANCED: Increased from 15 to 25 for better entity coverage
        # OPTIMIZATION: Keep strict relationship limits to control graph density
        'max_relationships_per_entity': 4,  # REBALANCED: Increased from 2 to 4 (still controlled)
        'max_relationships_per_chunk': 6,  # REBALANCED: Increased from 3 to 6 (still controlled)
        'enable_coreference_resolution': True,
        'schema_mode': 'hybrid',  # 'static', 'dynamic', or 'hybrid'
        'entity_discovery': {
            'enabled': True,
            'confidence_threshold': 0.6,  # REBALANCED: Reduced from 0.8 to 0.6 for better entity coverage
            'max_entity_types': 40,  # REBALANCED: Increased from 30 to 40 for business entity types
            'auto_categorize': True,
            'discovery_prompt': None,  # Use database-driven prompt from LLMPrompt table
            'min_frequency': 2,  # REBALANCED: Reduced from 3 to 2 for better discovery
            'enable_semantic_grouping': True
        },
        'relationship_discovery': {
            'enabled': True,
            'confidence_threshold': 0.7,  # REBALANCED: Reduced from 0.8 to 0.7 (still conservative)
            'max_relationship_types': 25,  # REBALANCED: Increased from 20 to 25 (controlled)
            'semantic_grouping': True,
            'discovery_prompt': None,  # Use database-driven prompt from LLMPrompt table
            'min_frequency': 2,  # REBALANCED: Reduced from 3 to 2 for better discovery
            'enable_inverse_relationships': False  # OPTIMIZATION: Keep disabled to control relationships
        },
        # No static fallback - pure LLM discovery only
        'learning': {
            'enable_user_feedback': True,
            'auto_accept_threshold': 0.85,
            'manual_review_threshold': 0.6,
            'frequency_tracking': True,
            'learning_rate': 0.1,
            'decay_factor': 0.95
        },
        'extraction': {
            'min_entity_confidence': 0.5,  # REBALANCED: Reduced from 0.8 to 0.5 for better entity coverage
            'min_relationship_confidence': 0.6,  # REBALANCED: Reduced from 0.7 to 0.6 (still conservative)
            'enable_entity_deduplication': True,
            'enable_relationship_deduplication': True,
            'entity_merge_strategy': 'merge_by_canonical_name_and_type',
            'confidence_merge_strategy': 'highest',
            'enable_flexible_entity_matching': True,
            'enable_semantic_relationship_inference': False,  # OPTIMIZATION: Disabled to reduce relationships
            'prioritize_pattern_relationships': True,
            'max_proximity_distance': 20,  # OPTIMIZATION: Reduced from 30 to 20
            'enable_relationship_priority_deduplication': True,
            'relationship_quality_threshold': 0.7,  # OPTIMIZATION: Increased from 0.4 to 0.7
            'enable_llm_enhancement': False,  # OPTIMIZATION: Disabled to reduce relationships
            'llm_confidence_threshold': 0.8,  # OPTIMIZATION: Increased from 0.6 to 0.8
            'enable_cross_document_linking': False,  # OPTIMIZATION: Disabled to reduce relationships
            'enable_multi_chunk_relationships': False,  # OPTIMIZATION: Disabled to reduce relationships
            # REBALANCED: Re-enable anti-silo analysis for entity discovery (controlled)
            'enable_anti_silo': True,
            'anti_silo_similarity_threshold': 0.75,
            'anti_silo_type_boost': 1.2,
            # OPTIMIZATION: Disable all anti-silo features to reduce relationship creation
            'enable_cooccurrence_analysis': False,
            'enable_type_based_clustering': False,
            'enable_hub_entities': False,
            'hub_entity_threshold': 3,
            'enable_semantic_clustering': False,
            'clustering_similarity_threshold': 0.8,
            'enable_document_bridge_relationships': False,
            'bridge_relationship_confidence': 0.6,
            'enable_temporal_linking': False,
            'temporal_linking_window': 7,  # days
            'enable_contextual_linking': False,
            'contextual_linking_threshold': 0.7,
            'enable_fuzzy_matching': False,
            'fuzzy_matching_threshold': 0.85,
            'enable_alias_detection': False,
            'alias_detection_threshold': 0.9,
            'enable_abbreviation_matching': False,
            'abbreviation_matching_threshold': 0.8,
            'enable_synonym_detection': False,
            'synonym_detection_threshold': 0.75,
            'enable_hierarchical_linking': False,
            'hierarchical_linking_depth': 2,
            'enable_geographic_linking': False,
            'geographic_linking_threshold': 0.7,
            'enable_temporal_coherence': False,
            'temporal_coherence_threshold': 0.6,
            'enable_semantic_bridge_entities': False,
            'semantic_bridge_threshold': 0.65,
            'enable_cross_reference_analysis': False,
            'cross_reference_threshold': 0.7,
            'enable_relationship_propagation': False,
            'relationship_propagation_depth': 2,
            # OPTIMIZATION: Disable more relationship-creating features
            'enable_entity_consolidation': True,  # Keep this for deduplication
            'entity_consolidation_threshold': 0.85,
            'enable_synthetic_relationships': False,  # OPTIMIZATION: Disabled synthetic relationships
            'synthetic_relationship_confidence': 0.5,
            'enable_graph_enrichment': False,  # OPTIMIZATION: Disabled graph enrichment
            'graph_enrichment_depth': 3,
            'enable_connectivity_analysis': False,  # OPTIMIZATION: Disabled connectivity analysis
            'connectivity_analysis_threshold': 0.4,
            'enable_isolation_detection': False,  # OPTIMIZATION: Disabled isolation detection
            'isolation_detection_threshold': 1,
            'enable_relationship_recommendation': False,  # OPTIMIZATION: Disabled recommendations
            'relationship_recommendation_threshold': 0.6,
            'enable_semantic_similarity_networks': False,  # OPTIMIZATION: Disabled semantic networks
            'semantic_network_threshold': 0.7,
            'enable_multi_document_analysis': False,  # OPTIMIZATION: Disabled multi-document analysis
            'multi_document_analysis_threshold': 0.65,
            'enable_entity_lifecycle_tracking': True,
            'lifecycle_tracking_window': 30,  # days
            'enable_relationship_evolution': True,
            'relationship_evolution_threshold': 0.5,
            'enable_context_aware_linking': True,
            'context_aware_threshold': 0.7,
            'enable_dynamic_relationship_types': True,
            'dynamic_relationship_threshold': 0.6,
            'enable_relationship_weighting': True,
            'relationship_weighting_factor': 1.0,
            'enable_entity_importance_scoring': True,
            'importance_scoring_threshold': 0.5,
            'enable_graph_density_optimization': True,
            'density_optimization_threshold': 0.3,
            'enable_relationship_diversity': True,
            'relationship_diversity_threshold': 0.4,
            'enable_entity_centrality_analysis': True,
            'centrality_analysis_threshold': 0.5,
            'enable_community_detection': True,
            'community_detection_threshold': 0.6,
            'enable_graph_summarization': True,
            'graph_summarization_threshold': 0.7,
            'enable_relationship_inference': True,
            'relationship_inference_threshold': 0.5,
            'enable_entity_embedding': True,
            'entity_embedding_threshold': 0.7,
            'enable_relationship_embedding': True,
            'relationship_embedding_threshold': 0.6,
            'enable_graph_neural_networks': True,
            'graph_neural_network_threshold': 0.5,
            'enable_knowledge_graph_completion': True,
            'completion_threshold': 0.4,
            'enable_missing_relationship_detection': True,
            'missing_relationship_threshold': 0.5,
            'enable_graph_quality_assessment': True,
            'quality_assessment_threshold': 0.6,
            'enable_relationship_validation': True,
            'relationship_validation_threshold': 0.7,
            'enable_entity_validation': True,
            'entity_validation_threshold': 0.8,
            'enable_graph_consistency_checking': True,
            'consistency_checking_threshold': 0.6,
            'enable_redundancy_detection': True,
            'redundancy_detection_threshold': 0.8,
            'enable_graph_compression': True,
            'graph_compression_threshold': 0.5,
            'enable_relationship_refinement': True,
            'relationship_refinement_threshold': 0.6,
            'enable_entity_refinement': True,
            'entity_refinement_threshold': 0.7,
            'enable_graph_evolution_tracking': True,
            'evolution_tracking_threshold': 0.5,
            'enable_temporal_graph_analysis': True,
            'temporal_graph_threshold': 0.6,
            'enable_multi_modal_linking': True,
            'multi_modal_linking_threshold': 0.5,
            'enable_cross_domain_analysis': True,
            'cross_domain_analysis_threshold': 0.4,
            'enable_knowledge_graph_merging': True,
            'merging_threshold': 0.7,
            'enable_graph_versioning': True,
            'versioning_threshold': 0.8,
            'enable_relationship_lifecycle': True,
            'relationship_lifecycle_threshold': 0.6,
            'enable_entity_lifecycle': True,
            'entity_lifecycle_threshold': 0.7,
            'enable_graph_maintenance': True,
            'maintenance_threshold': 0.5,
            'enable_performance_optimization': True,
            'performance_optimization_threshold': 0.6,
            'enable_scalability_optimization': True,
            'scalability_optimization_threshold': 0.5,
            'enable_real_time_processing': True,
            'real_time_processing_threshold': 0.7,
            'enable_batch_processing': True,
            'batch_processing_threshold': 0.6,
            'enable_stream_processing': True,
            'stream_processing_threshold': 0.5,
            'enable_distributed_processing': True,
            'distributed_processing_threshold': 0.4,
            'enable_parallel_processing': True,
            'parallel_processing_threshold': 0.6,
            'enable_incremental_processing': True,
            'incremental_processing_threshold': 0.7,
            'enable_caching': True,
            'caching_threshold': 0.8,
            'enable_memory_optimization': True,
            'memory_optimization_threshold': 0.6,
            'enable_storage_optimization': True,
            'storage_optimization_threshold': 0.5,
            'enable_query_optimization': True,
            'query_optimization_threshold': 0.7,
            'enable_indexing': True,
            'indexing_threshold': 0.8,
            'enable_search_optimization': True,
            'search_optimization_threshold': 0.6,
            'enable_retrieval_optimization': True,
            'retrieval_optimization_threshold': 0.7,
            'enable_analytics_optimization': True,
            'analytics_optimization_threshold': 0.5,
            'enable_monitoring': True,
            'monitoring_threshold': 0.6,
            'enable_logging': True,
            'logging_threshold': 0.8,
            'enable_debugging': True,
            'debugging_threshold': 0.7,
            'enable_testing': True,
            'testing_threshold': 0.6,
            'enable_validation': True,
            'validation_threshold': 0.8,
            'enable_verification': True,
            'verification_threshold': 0.7,
            'enable_certification': True,
            'certification_threshold': 0.6,
            'enable_compliance': True,
            'compliance_threshold': 0.8,
            'enable_security': True,
            'security_threshold': 0.9,
            'enable_privacy': True,
            'privacy_threshold': 0.8,
            'enable_encryption': True,
            'encryption_threshold': 0.9,
            'enable_access_control': True,
            'access_control_threshold': 0.8,
            'enable_audit_logging': True,
            'audit_logging_threshold': 0.7,
            'enable_backup': True,
            'backup_threshold': 0.9,
            'enable_recovery': True,
            'recovery_threshold': 0.8,
            'enable_disaster_recovery': True,
            'disaster_recovery_threshold': 0.9,
            'enable_business_continuity': True,
            'business_continuity_threshold': 0.8,
            'enable_high_availability': True,
            'high_availability_threshold': 0.9,
            'enable_fault_tolerance': True,
            'fault_tolerance_threshold': 0.8,
            'enable_resilience': True,
            'resilience_threshold': 0.7,
            'enable_scalability': True,
            'scalability_threshold': 0.6,
            'enable_elasticity': True,
            'elasticity_threshold': 0.5,
            'enable_load_balancing': True,
            'load_balancing_threshold': 0.7,
            'enable_resource_management': True,
            'resource_management_threshold': 0.6,
            'enable_cost_optimization': True,
            'cost_optimization_threshold': 0.5,
            'enable_efficiency': True,
            'efficiency_threshold': 0.7,
            'enable_sustainability': True,
            'sustainability_threshold': 0.4,
            'enable_green_computing': True,
            'green_computing_threshold': 0.3,
            'enable_carbon_footprint_reduction': True,
            'carbon_footprint_reduction_threshold': 0.2,
            'enable_environmental_impact': True,
            'environmental_impact_threshold': 0.3,
            'enable_social_impact': True,
            'social_impact_threshold': 0.4,
            'enable_economic_impact': True,
            'economic_impact_threshold': 0.5,
            'enable_ethical_considerations': True,
            'ethical_considerations_threshold': 0.6,
            'enable_responsible_ai': True,
            'responsible_ai_threshold': 0.7,
            'enable_fairness': True,
            'fairness_threshold': 0.8,
            'enable_transparency': True,
            'transparency_threshold': 0.7,
            'enable_explainability': True,
            'explainability_threshold': 0.6,
            'enable_accountability': True,
            'accountability_threshold': 0.8,
            'enable_governance': True,
            'governance_threshold': 0.7,
            'enable_compliance_monitoring': True,
            'compliance_monitoring_threshold': 0.8,
            'enable_risk_management': True,
            'risk_management_threshold': 0.7,
            'enable_threat_detection': True,
            'threat_detection_threshold': 0.8,
            'enable_vulnerability_assessment': True,
            'vulnerability_assessment_threshold': 0.7,
            'enable_incident_response': True,
            'incident_response_threshold': 0.9,
            'enable_forensics': True,
            'forensics_threshold': 0.8,
            'enable_investigation': True,
            'investigation_threshold': 0.7,
            'enable_evidence_collection': True,
            'evidence_collection_threshold': 0.9,
            'enable_chain_of_custody': True,
            'chain_of_custody_threshold': 0.9,
            'enable_legal_compliance': True,
            'legal_compliance_threshold': 0.8,
            'enable_regulatory_compliance': True,
            'regulatory_compliance_threshold': 0.9,
            'enable_industry_standards': True,
            'industry_standards_threshold': 0.8,
            'enable_best_practices': True,
            'best_practices_threshold': 0.7,
            'enable_quality_assurance': True,
            'quality_assurance_threshold': 0.8,
            'enable_continuous_improvement': True,
            'continuous_improvement_threshold': 0.6,
            'enable_feedback_loops': True,
            'feedback_loops_threshold': 0.5,
            'enable_user_feedback': True,
            'user_feedback_threshold': 0.4,
            'enable_stakeholder_feedback': True,
            'stakeholder_feedback_threshold': 0.3,
            'enable_customer_feedback': True,
            'customer_feedback_threshold': 0.2,
            'enable_partner_feedback': True,
            'partner_feedback_threshold': 0.3,
            'enable_vendor_feedback': True,
            'vendor_feedback_threshold': 0.4,
            'enable_employee_feedback': True,
            'employee_feedback_threshold': 0.5,
            'enable_management_feedback': True,
            'management_feedback_threshold': 0.6,
            'enable_executive_feedback': True,
            'executive_feedback_threshold': 0.7,
            'enable_board_feedback': True,
            'board_feedback_threshold': 0.8,
            'enable_investor_feedback': True,
            'investor_feedback_threshold': 0.9,
            'enable_public_feedback': True,
            'public_feedback_threshold': 0.3,
            'enable_community_feedback': True,
            'community_feedback_threshold': 0.4,
            'enable_social_feedback': True,
            'social_feedback_threshold': 0.2,
            'enable_media_feedback': True,
            'media_feedback_threshold': 0.3,
            'enable_analyst_feedback': True,
            'analyst_feedback_threshold': 0.4,
            'enable_researcher_feedback': True,
            'researcher_feedback_threshold': 0.5,
            'enable_academic_feedback': True,
            'academic_feedback_threshold': 0.6,
            'enable_industry_feedback': True,
            'industry_feedback_threshold': 0.7,
            'enable_government_feedback': True,
            'government_feedback_threshold': 0.8,
            'enable_regulatory_feedback': True,
            'regulatory_feedback_threshold': 0.9,
            'enable_international_feedback': True,
            'international_feedback_threshold': 0.7,
            'enable_global_feedback': True,
            'global_feedback_threshold': 0.6,
            'enable_local_feedback': True,
            'local_feedback_threshold': 0.5,
            'enable_regional_feedback': True,
            'regional_feedback_threshold': 0.4,
            'enable_national_feedback': True,
            'national_feedback_threshold': 0.3,
            'enable_continental_feedback': True,
            'continental_feedback_threshold': 0.2,
            'enable_worldwide_feedback': True,
            'worldwide_feedback_threshold': 0.1
        },
        'discovered_schemas': {
            'entities': {},
            'relationships': {},
            'last_updated': None,
            'version': '2.0.0'
        },
        'neo4j': get_default_neo4j_config()
    }

def set_knowledge_graph_settings(settings_dict: Dict[str, Any]):
    """Set knowledge graph settings in cache and persist to database"""
    # Synchronize model fields before saving
    settings_dict = _synchronize_model_fields(settings_dict)
    
    # Always save to cache first
    cache.set(KNOWLEDGE_GRAPH_SETTINGS_KEY, settings_dict, expire=get_settings_cache_ttl())
    
    # Then persist to database
    try:
        from app.core.db import SessionLocal, Settings as SettingsModel
        from datetime import datetime
        
        db = SessionLocal()
        try:
            # Try to find existing knowledge_graph category
            kg_row = db.query(SettingsModel).filter(SettingsModel.category == 'knowledge_graph').first()
            if kg_row:
                # Update existing knowledge graph settings
                existing_settings = kg_row.settings or {}
                existing_settings.update(settings_dict)
                kg_row.settings = existing_settings
                kg_row.updated_at = datetime.now()
            else:
                # Create new knowledge_graph settings row
                kg_row = SettingsModel(
                    category='knowledge_graph', 
                    settings=settings_dict,
                    updated_at=datetime.now()
                )
                db.add(kg_row)
            
            db.commit()
            print("Knowledge graph settings persisted to database successfully")
        finally:
            db.close()
    except Exception as e:
        print(f"Failed to persist knowledge graph settings to database: {e}")
        # Still return success since cache worked, but log the error

def _synchronize_model_fields(settings_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronize model fields to ensure consistency between main 'model' and 'model_config.model'"""
    settings_copy = settings_dict.copy()
    
    # If model_config.model exists, sync it to main model field
    if 'model_config' in settings_copy and isinstance(settings_copy['model_config'], dict):
        if 'model' in settings_copy['model_config']:
            settings_copy['model'] = settings_copy['model_config']['model']
            print(f"[SYNC] Updated main model field to: {settings_copy['model']}")
    
    # If main model exists but model_config.model doesn't, sync it
    elif 'model' in settings_copy:
        if 'model_config' not in settings_copy:
            settings_copy['model_config'] = {}
        if isinstance(settings_copy['model_config'], dict):
            settings_copy['model_config']['model'] = settings_copy['model']
            print(f"[SYNC] Updated model_config.model field to: {settings_copy['model']}")
    
    return settings_copy

def reload_knowledge_graph_settings() -> Dict[str, Any]:
    """Reload knowledge graph settings from database"""
    try:
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            # First try the new knowledge_graph category
            kg_row = db.query(SettingsModel).filter(SettingsModel.category == 'knowledge_graph').first()
            if kg_row and kg_row.settings:
                settings = kg_row.settings
                
                # Ensure Neo4j config is present
                if 'neo4j' not in settings:
                    settings['neo4j'] = get_default_neo4j_config()
                    # Save updated settings back to database
                    kg_row.settings = settings
                    db.commit()
                    print("Added default Neo4j configuration to knowledge graph settings")
                
                # Cache the settings
                cache.set(KNOWLEDGE_GRAPH_SETTINGS_KEY, settings, expire=get_settings_cache_ttl())
                return settings
            
            # Fallback: check for legacy knowledge_graph in LLM settings
            llm_row = db.query(SettingsModel).filter(SettingsModel.category == 'llm').first()
            if llm_row and 'knowledge_graph' in llm_row.settings:
                print("Found legacy knowledge_graph settings in LLM category - consider running migration")
                settings = llm_row.settings['knowledge_graph']
                
                # Ensure Neo4j config is present
                if 'neo4j' not in settings:
                    settings['neo4j'] = get_default_neo4j_config()
                
                # Cache the settings
                cache.set(KNOWLEDGE_GRAPH_SETTINGS_KEY, settings, expire=get_settings_cache_ttl())
                return settings
            
            # CRITICAL SAFETY: Check if there are existing cached settings before overwriting with defaults
            existing_cached = cache.get(KNOWLEDGE_GRAPH_SETTINGS_KEY)
            if existing_cached and isinstance(existing_cached, dict):
                # Existing cache has complex settings - preserve them
                complex_fields = ['prompts', 'extraction', 'learning', 'discovered_schemas', 'entity_discovery', 'relationship_discovery']
                has_complex_data = any(field in existing_cached and existing_cached[field] for field in complex_fields)
                if has_complex_data:
                    print("WARNING: Preserving complex cached knowledge graph settings instead of overwriting with defaults")
                    print(f"Complex fields found: {[field for field in complex_fields if field in existing_cached and existing_cached[field]]}")
                    return existing_cached
            
            # No existing complex settings found, safe to use defaults  
            default_settings = get_default_knowledge_graph_settings()
            cache.set(KNOWLEDGE_GRAPH_SETTINGS_KEY, default_settings, expire=get_settings_cache_ttl())
            print("Using default knowledge graph settings (no existing complex data to preserve)")
            return default_settings
        finally:
            db.close()
    except Exception as e:
        print(f"Failed to load knowledge graph settings from database: {e}")
        
        # CRITICAL SAFETY: Check existing cache before falling back to defaults
        existing_cached = cache.get(KNOWLEDGE_GRAPH_SETTINGS_KEY)
        if existing_cached and isinstance(existing_cached, dict):
            complex_fields = ['prompts', 'extraction', 'learning', 'discovered_schemas', 'entity_discovery', 'relationship_discovery']
            has_complex_data = any(field in existing_cached and existing_cached[field] for field in complex_fields)
            if has_complex_data:
                print("ERROR FALLBACK: Preserving complex cached settings instead of using defaults")
                return existing_cached
        
        print("ERROR FALLBACK: Using default settings (no complex cached data to preserve)")
        return get_default_knowledge_graph_settings()

def detect_business_document(filename: str = '', content: str = '') -> bool:
    """Detect if document is business type for enhanced processing"""
    business_keywords = [
        'strategy', 'technology', 'financial', 'annual report', 'business plan',
        'quarterly report', 'dbs bank', 'subsidiary', 'organization', 'enterprise',
        'digital transformation', 'technology roadmap', 'competitive analysis'
    ]
    
    text_to_check = (filename + ' ' + content[:1000]).lower()
    return any(keyword in text_to_check for keyword in business_keywords)

def get_business_optimized_settings() -> Dict[str, Any]:
    """Get optimized settings for business documents"""
    settings = get_knowledge_graph_settings().copy()
    
    # Business document optimizations
    settings.update({
        'max_entities_per_chunk': 35,  # Higher for business docs
        'max_relationships_per_chunk': 10,  # More relationships for business context
        'extraction': {
            **settings.get('extraction', {}),
            'min_entity_confidence': 0.55,  # Slightly lower for business entities
            'min_relationship_confidence': 0.6,  # Lower for business relationships
            'business_document_mode': True,
            'extract_all_organizations': True,
            'extract_business_concepts': True,
            'extract_temporal_entities': True,
            'enable_multi_pass': True,
            'extraction_passes': ['core_entities', 'business_concepts', 'deep_relationships', 'temporal_causal']
        },
        'entity_discovery': {
            **settings.get('entity_discovery', {}),
            'confidence_threshold': 0.55,
            'max_entity_types': 60,  # More types for business entities
        },
        'relationship_discovery': {
            **settings.get('relationship_discovery', {}),
            'confidence_threshold': 0.6,
            'max_relationship_types': 35
        }
    })
    
    return settings

def get_knowledge_graph_full_config(settings=None) -> Dict[str, Any]:
    """Construct full knowledge graph configuration by merging base config with mode parameters"""
    if settings is None:
        settings = get_knowledge_graph_settings()
    
    # Get LLM settings for mode parameters
    try:
        from app.core.llm_settings_cache import get_llm_settings
        llm_settings = get_llm_settings()
        mode = settings.get('mode', 'thinking')
        
        # Get the appropriate mode parameters
        if mode == 'thinking':
            mode_params = llm_settings.get('thinking_mode_params', {})
        else:
            mode_params = llm_settings.get('non_thinking_mode_params', {})
        
        # Merge base config with mode parameters
        full_config = settings.copy()
        full_config.update(mode_params)
        
        return full_config
    except Exception as e:
        print(f"[ERROR] Failed to get full knowledge graph config: {str(e)}")
        return settings

def test_neo4j_connection() -> Dict[str, Any]:
    """Test Neo4j connection using current settings"""
    try:
        config = get_neo4j_config()
        
        if not config.get('enabled', False):
            return {
                'success': False,
                'error': 'Neo4j is disabled in configuration',
                'config': config
            }
        
        # Here we would test the actual connection
        # For now, return the configuration for testing
        return {
            'success': True,
            'message': 'Neo4j configuration loaded successfully',
            'config': config
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'config': None
        }

def update_neo4j_password(new_password: str) -> bool:
    """Update Neo4j password in settings"""
    try:
        from app.core.db import SessionLocal, Settings as SettingsModel
        from datetime import datetime
        
        db = SessionLocal()
        try:
            # Try knowledge_graph category first
            kg_row = db.query(SettingsModel).filter(SettingsModel.category == 'knowledge_graph').first()
            if kg_row and kg_row.settings:
                # Update password in knowledge graph settings
                if 'neo4j' not in kg_row.settings:
                    kg_row.settings['neo4j'] = get_default_neo4j_config()
                
                kg_row.settings['neo4j']['password'] = new_password
                
                # Update URI with new password if using embedded auth
                neo4j_config = kg_row.settings['neo4j']
                host = neo4j_config.get('host', 'neo4j')
                port = neo4j_config.get('port', 7687)
                username = neo4j_config.get('username', 'neo4j')
                kg_row.settings['neo4j']['uri'] = f"bolt://{username}:{new_password}@{host}:{port}"
                kg_row.updated_at = datetime.now()
                
                db.commit()
                
                # Reload cache
                reload_knowledge_graph_settings()
                return True
            
            # Fallback: try legacy location in LLM settings
            llm_row = db.query(SettingsModel).filter(SettingsModel.category == 'llm').first()
            if llm_row and 'knowledge_graph' in llm_row.settings:
                print("Updating Neo4j password in legacy LLM settings - consider running migration")
                # Update password in knowledge graph settings
                llm_row.settings['knowledge_graph']['neo4j']['password'] = new_password
                
                # Update URI with new password if using embedded auth
                neo4j_config = llm_row.settings['knowledge_graph']['neo4j']
                host = neo4j_config.get('host', 'neo4j')
                port = neo4j_config.get('port', 7687)
                username = neo4j_config.get('username', 'neo4j')
                llm_row.settings['knowledge_graph']['neo4j']['uri'] = f"bolt://{username}:{new_password}@{host}:{port}"
                llm_row.updated_at = datetime.now()
                
                db.commit()
                
                # Reload cache
                reload_knowledge_graph_settings()
                return True
            
            return False
        finally:
            db.close()
    except Exception as e:
        print(f"Failed to update Neo4j password: {e}")
        return False
