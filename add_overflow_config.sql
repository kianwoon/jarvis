-- Add overflow configuration to settings table
INSERT INTO settings (category, settings)
VALUES (
    'overflow',
    '{
        "overflow_threshold_tokens": 8000,
        "chunk_size_tokens": 2000,
        "chunk_overlap_tokens": 200,
        "l1_ttl_hours": 24,
        "l2_ttl_days": 7,
        "max_overflow_context_ratio": 0.3,
        "retrieval_top_k": 5,
        "enable_semantic_search": true,
        "enable_keyword_extraction": true,
        "auto_promote_to_l1": true,
        "promotion_threshold_accesses": 3
    }'::jsonb
)
ON CONFLICT (category) DO UPDATE
SET settings = EXCLUDED.settings,
    updated_at = NOW();