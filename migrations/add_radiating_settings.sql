-- Migration: Add Radiating Coverage System Settings
-- Purpose: Initialize radiating settings in the settings table for universal coverage system
-- Date: 2025-01-16

BEGIN;

-- Insert default radiating settings if not exists
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
        },
        "traversal": {
            "strategy": "hybrid",
            "prioritization": "relevance",
            "enable_pruning": true,
            "pruning_threshold": 0.4,
            "enable_cycle_detection": true,
            "max_cycles": 2,
            "enable_path_optimization": true,
            "path_weight_decay": 0.8,
            "enable_dynamic_depth": true,
            "depth_adjustment_factor": 0.9,
            "enable_context_switching": true,
            "context_switch_threshold": 0.5
        },
        "coverage": {
            "enable_gap_detection": true,
            "gap_threshold": 0.6,
            "enable_overlap_detection": true,
            "overlap_threshold": 0.8,
            "enable_completeness_checking": true,
            "completeness_threshold": 0.7,
            "enable_redundancy_elimination": true,
            "redundancy_threshold": 0.9,
            "enable_coverage_metrics": true,
            "metric_calculation_interval": 100
        },
        "synthesis": {
            "enable_result_merging": true,
            "merge_strategy": "weighted",
            "enable_conflict_resolution": true,
            "conflict_strategy": "confidence",
            "enable_result_ranking": true,
            "ranking_algorithm": "combined",
            "enable_result_filtering": true,
            "filter_duplicates": true,
            "filter_low_confidence": true,
            "enable_result_enrichment": true,
            "enrichment_sources": ["knowledge_graph", "vector_store", "llm"]
        },
        "performance": {
            "enable_caching": true,
            "cache_strategy": "lru",
            "cache_size": 1000,
            "batch_size": 10,
            "parallel_processing": true,
            "max_concurrent_queries": 5,
            "timeout_seconds": 30,
            "enable_query_optimization": true,
            "enable_index_optimization": true,
            "enable_memory_optimization": true,
            "memory_limit_mb": 512,
            "enable_progressive_loading": true,
            "progressive_chunk_size": 100
        },
        "model_config": {
            "model": "qwen3:30b-a3b-q4_K_M",
            "temperature": 0.3,
            "max_tokens": 4096,
            "context_length": 32768,
            "repeat_penalty": 1.05,
            "top_p": 0.9,
            "top_k": 40
        },
        "monitoring": {
            "enable_metrics": true,
            "enable_tracing": false,
            "enable_profiling": false,
            "log_level": "INFO",
            "metrics_export_interval": 60,
            "enable_query_logging": true,
            "enable_result_logging": false,
            "enable_performance_logging": true
        }
    }'::jsonb,
    NOW()
)
ON CONFLICT (category) DO UPDATE
SET 
    settings = EXCLUDED.settings,
    updated_at = NOW()
WHERE settings.category = 'radiating';

-- Add comment for documentation
COMMENT ON COLUMN settings.settings IS 'JSON configuration for various system categories including radiating coverage system';

-- Verify the insertion/update
SELECT 
    category,
    jsonb_pretty(settings) as pretty_settings,
    updated_at
FROM settings 
WHERE category = 'radiating';

COMMIT;

-- Verification query (comment out in production)
-- SELECT 'Radiating settings initialized' as status, 
--        CASE WHEN EXISTS (SELECT 1 FROM settings WHERE category = 'radiating') 
--             THEN 'SUCCESS' 
--             ELSE 'FAILED' 
--        END as result;