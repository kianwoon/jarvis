-- Migration: Add Memory Feature and Universal Chunk Editing
-- Purpose: Add notebook memories and chunk editing capabilities for both documents and memories
-- Date: 2025-01-25

BEGIN;

-- Create notebook_memories table for storing text-based memories
CREATE TABLE IF NOT EXISTS notebook_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    notebook_id UUID NOT NULL,
    memory_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    content TEXT NOT NULL,
    milvus_collection VARCHAR(100),
    chunk_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB,
    
    CONSTRAINT fk_notebook_memories_notebook_id 
        FOREIGN KEY (notebook_id) 
        REFERENCES notebooks(id) 
        ON DELETE CASCADE
);

-- Create chunk_edits audit table for tracking chunk modifications
CREATE TABLE IF NOT EXISTS chunk_edits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id VARCHAR(255) NOT NULL,
    document_id VARCHAR(255) NOT NULL,
    content_type VARCHAR(50) NOT NULL CHECK (content_type IN ('document', 'memory')),
    original_content TEXT NOT NULL,
    edited_content TEXT NOT NULL,
    edited_by VARCHAR(255),
    edited_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    re_embedded BOOLEAN DEFAULT FALSE,
    metadata JSONB
);

-- Add new columns to knowledge_graph_documents for enhanced tracking
ALTER TABLE knowledge_graph_documents 
ADD COLUMN IF NOT EXISTS content_type VARCHAR(50) DEFAULT 'document' CHECK (content_type IN ('document', 'memory'));

ALTER TABLE knowledge_graph_documents 
ADD COLUMN IF NOT EXISTS last_chunk_edit_at TIMESTAMP WITH TIME ZONE;

ALTER TABLE knowledge_graph_documents 
ADD COLUMN IF NOT EXISTS edited_chunks_count INTEGER DEFAULT 0;

-- Create indexes for notebook_memories
CREATE INDEX IF NOT EXISTS idx_notebook_memories_notebook_id ON notebook_memories(notebook_id);
CREATE INDEX IF NOT EXISTS idx_notebook_memories_memory_id ON notebook_memories(memory_id);
CREATE INDEX IF NOT EXISTS idx_notebook_memories_created_at ON notebook_memories(created_at);

-- Create indexes for chunk_edits
CREATE INDEX IF NOT EXISTS idx_chunk_edits_chunk_id ON chunk_edits(chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunk_edits_document_id ON chunk_edits(document_id);
CREATE INDEX IF NOT EXISTS idx_chunk_edits_content_type ON chunk_edits(content_type);
CREATE INDEX IF NOT EXISTS idx_chunk_edits_edited_at ON chunk_edits(edited_at);

-- Create indexes for new knowledge_graph_documents columns
CREATE INDEX IF NOT EXISTS idx_kg_doc_content_type ON knowledge_graph_documents(content_type);
CREATE INDEX IF NOT EXISTS idx_kg_doc_last_chunk_edit_at ON knowledge_graph_documents(last_chunk_edit_at);

-- Create trigger to update updated_at timestamp for notebook_memories
CREATE OR REPLACE FUNCTION update_notebook_memories_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_notebook_memories_updated_at
    BEFORE UPDATE ON notebook_memories
    FOR EACH ROW
    EXECUTE FUNCTION update_notebook_memories_updated_at();

-- Create trigger to update knowledge_graph_documents when chunks are edited
CREATE OR REPLACE FUNCTION update_document_chunk_edit_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update the document's chunk edit statistics
    UPDATE knowledge_graph_documents 
    SET 
        last_chunk_edit_at = NEW.edited_at,
        edited_chunks_count = (
            SELECT COUNT(DISTINCT chunk_id) 
            FROM chunk_edits 
            WHERE document_id = NEW.document_id
        )
    WHERE document_id = NEW.document_id;
    
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_document_chunk_edit_stats
    AFTER INSERT ON chunk_edits
    FOR EACH ROW
    EXECUTE FUNCTION update_document_chunk_edit_stats();

COMMIT;