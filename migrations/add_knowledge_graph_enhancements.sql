-- Migration: Add Knowledge Graph Enhancement Tables
-- Purpose: Enhance document ingestion with unified Milvus/Neo4j processing and quality metrics
-- Date: $(date +"%Y-%m-%d")

BEGIN;

-- Create knowledge_graph_documents table for enhanced document tracking
CREATE TABLE IF NOT EXISTS knowledge_graph_documents (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) UNIQUE NOT NULL,
    filename VARCHAR(500) NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    file_size_bytes INTEGER,
    file_type VARCHAR(50),
    
    -- Processing configuration
    milvus_collection VARCHAR(100),
    neo4j_graph_id VARCHAR(255),
    processing_mode VARCHAR(50) DEFAULT 'unified',
    
    -- Processing status and metrics
    processing_status VARCHAR(50) DEFAULT 'pending',
    entities_extracted INTEGER DEFAULT 0,
    relationships_extracted INTEGER DEFAULT 0,
    chunks_processed INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 0,
    processing_time_ms INTEGER,
    extraction_confidence FLOAT,
    
    -- Error tracking
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Metadata and provenance
    upload_metadata JSONB,
    processing_config JSONB,
    quality_scores JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for knowledge_graph_documents
CREATE INDEX IF NOT EXISTS idx_kg_doc_document_id ON knowledge_graph_documents(document_id);
CREATE INDEX IF NOT EXISTS idx_kg_doc_file_hash ON knowledge_graph_documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_kg_doc_processing_status ON knowledge_graph_documents(processing_status);

-- Create extraction_quality_metrics table for quality tracking
CREATE TABLE IF NOT EXISTS extraction_quality_metrics (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL REFERENCES knowledge_graph_documents(document_id) ON DELETE CASCADE,
    chunk_id VARCHAR(255),
    
    -- Extraction results
    entities_discovered INTEGER DEFAULT 0,
    relationships_discovered INTEGER DEFAULT 0,
    entities_validated INTEGER DEFAULT 0,
    relationships_validated INTEGER DEFAULT 0,
    
    -- Quality scores
    confidence_scores JSONB,
    validation_scores JSONB,
    
    -- Processing details
    llm_model_used VARCHAR(100),
    processing_method VARCHAR(50),
    processing_time_ms INTEGER,
    
    -- Validation and errors
    validation_errors JSONB,
    extraction_warnings JSONB,
    
    -- Cross-reference tracking
    milvus_chunk_ids JSONB,
    neo4j_entity_ids JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for extraction_quality_metrics
CREATE INDEX IF NOT EXISTS idx_quality_document_id ON extraction_quality_metrics(document_id);
CREATE INDEX IF NOT EXISTS idx_quality_chunk_id ON extraction_quality_metrics(chunk_id);

-- Create graph_schema_evolution table for schema versioning
CREATE TABLE IF NOT EXISTS graph_schema_evolution (
    id SERIAL PRIMARY KEY,
    schema_version VARCHAR(20) UNIQUE NOT NULL,
    
    -- Schema definitions
    entity_types JSONB NOT NULL,
    relationship_types JSONB NOT NULL,
    
    -- Configuration snapshots
    confidence_thresholds JSONB,
    extraction_config JSONB,
    
    -- Change tracking
    change_description TEXT,
    change_type VARCHAR(50),
    changes_summary JSONB,
    
    -- Impact metrics
    documents_affected INTEGER DEFAULT 0,
    entities_reclassified INTEGER DEFAULT 0,
    relationships_reclassified INTEGER DEFAULT 0,
    
    -- Provenance
    created_by VARCHAR(100),
    trigger_event VARCHAR(200),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for graph_schema_evolution
CREATE INDEX IF NOT EXISTS idx_schema_version ON graph_schema_evolution(schema_version);

-- Create document_cross_references table for Milvus/Neo4j mapping
CREATE TABLE IF NOT EXISTS document_cross_references (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL REFERENCES knowledge_graph_documents(document_id) ON DELETE CASCADE,
    
    -- Milvus references
    milvus_collection VARCHAR(100) NOT NULL,
    milvus_chunk_id VARCHAR(255) NOT NULL,
    chunk_text_preview VARCHAR(500),
    
    -- Neo4j references
    neo4j_entity_id VARCHAR(255) NOT NULL,
    entity_name VARCHAR(255) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    
    -- Relationship metadata
    confidence_score FLOAT,
    relationship_type VARCHAR(100),
    context_window JSONB,
    
    -- Quality and validation
    validation_status VARCHAR(50) DEFAULT 'pending',
    manual_review BOOLEAN DEFAULT FALSE,
    review_notes TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure unique mapping between chunks and entities
    UNIQUE(milvus_chunk_id, neo4j_entity_id)
);

-- Create indexes for document_cross_references
CREATE INDEX IF NOT EXISTS idx_cross_ref_document_id ON document_cross_references(document_id);
CREATE INDEX IF NOT EXISTS idx_cross_ref_milvus_chunk ON document_cross_references(milvus_chunk_id);
CREATE INDEX IF NOT EXISTS idx_cross_ref_neo4j_entity ON document_cross_references(neo4j_entity_id);
CREATE INDEX IF NOT EXISTS idx_cross_ref_validation_status ON document_cross_references(validation_status);

-- Create trigger for updating updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update trigger to relevant tables
CREATE TRIGGER update_kg_documents_updated_at 
    BEFORE UPDATE ON knowledge_graph_documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_cross_references_updated_at 
    BEFORE UPDATE ON document_cross_references 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert initial schema version if not exists
INSERT INTO graph_schema_evolution (
    schema_version,
    entity_types,
    relationship_types,
    change_description,
    change_type,
    created_by,
    trigger_event
) VALUES (
    '1.0.0',
    '{"Person": {"description": "Individual people", "examples": ["John Smith", "Dr. Jane Doe"], "confidence_threshold": 0.75}, "Organization": {"description": "Companies and institutions", "examples": ["Apple Inc.", "Stanford University"], "confidence_threshold": 0.75}, "Location": {"description": "Places and geographical entities", "examples": ["San Francisco", "California"], "confidence_threshold": 0.75}, "Event": {"description": "Occurrences and happenings", "examples": ["Conference 2024", "Product Launch"], "confidence_threshold": 0.75}, "Concept": {"description": "Abstract ideas and topics", "examples": ["Machine Learning", "Sustainability"], "confidence_threshold": 0.75}}',
    '{"WORKS_FOR": {"description": "Employment relationship", "inverse": "EMPLOYS", "examples": ["John works for Apple"], "confidence_threshold": 0.7}, "LOCATED_IN": {"description": "Location relationship", "inverse": "CONTAINS", "examples": ["Apple located in Cupertino"], "confidence_threshold": 0.7}, "PART_OF": {"description": "Membership or component relationship", "inverse": "CONTAINS", "examples": ["Department part of Company"], "confidence_threshold": 0.7}, "RELATED_TO": {"description": "General relationship", "examples": ["Project related to AI"], "confidence_threshold": 0.6}, "CAUSES": {"description": "Causal relationship", "examples": ["Rain causes flooding"], "confidence_threshold": 0.7}}',
    'Initial schema with default entity and relationship types',
    'system_initialization',
    'system',
    'database_migration'
) ON CONFLICT (schema_version) DO NOTHING;

-- Add helpful comments
COMMENT ON TABLE knowledge_graph_documents IS 'Enhanced document tracking for unified Milvus/Neo4j processing pipeline';
COMMENT ON TABLE extraction_quality_metrics IS 'Quality metrics and validation tracking for knowledge graph extraction';
COMMENT ON TABLE graph_schema_evolution IS 'Version control and evolution tracking for knowledge graph schema';
COMMENT ON TABLE document_cross_references IS 'Cross-reference mapping between Milvus vector chunks and Neo4j graph entities';

COMMIT;

-- Verification queries (comment out in production)
-- SELECT 'knowledge_graph_documents' as table_name, COUNT(*) as row_count FROM knowledge_graph_documents
-- UNION ALL
-- SELECT 'extraction_quality_metrics' as table_name, COUNT(*) as row_count FROM extraction_quality_metrics
-- UNION ALL
-- SELECT 'graph_schema_evolution' as table_name, COUNT(*) as row_count FROM graph_schema_evolution
-- UNION ALL
-- SELECT 'document_cross_references' as table_name, COUNT(*) as row_count FROM document_cross_references;