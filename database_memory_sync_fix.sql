-- Memory ID Synchronization Fix and Prevention
-- This script ensures proper synchronization between notebook_memories and knowledge_graph_documents

-- 1. CREATE FUNCTION TO AUDIT MEMORY SYNCHRONIZATION
CREATE OR REPLACE FUNCTION audit_memory_synchronization()
RETURNS TABLE(
    sync_status VARCHAR(20),
    memory_id VARCHAR(255),
    notebook_id UUID,
    kgd_document_id VARCHAR(255),
    content_type VARCHAR(50),
    filename VARCHAR(500),
    action_needed TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        CASE 
            WHEN nm.memory_id IS NOT NULL AND kgd.document_id IS NOT NULL THEN 'SYNCHRONIZED'::VARCHAR(20)
            WHEN nm.memory_id IS NOT NULL AND kgd.document_id IS NULL THEN 'MISSING_IN_KGD'::VARCHAR(20)
            WHEN nm.memory_id IS NULL AND kgd.document_id IS NOT NULL THEN 'ORPHANED_IN_KGD'::VARCHAR(20)
            ELSE 'UNKNOWN'::VARCHAR(20)
        END as sync_status,
        nm.memory_id,
        nm.notebook_id,
        kgd.document_id as kgd_document_id,
        kgd.content_type,
        kgd.filename,
        CASE 
            WHEN nm.memory_id IS NOT NULL AND kgd.document_id IS NULL THEN 
                'CREATE record in knowledge_graph_documents with document_id=' || nm.memory_id::text
            WHEN nm.memory_id IS NULL AND kgd.document_id IS NOT NULL AND kgd.content_type = 'memory' THEN 
                'DELETE orphaned memory record from knowledge_graph_documents: ' || kgd.document_id::text
            ELSE 'No action needed'
        END as action_needed
    FROM notebook_memories nm 
    FULL OUTER JOIN knowledge_graph_documents kgd 
        ON nm.memory_id = kgd.document_id AND kgd.content_type = 'memory';
END;
$$ LANGUAGE plpgsql;

-- 2. CREATE FUNCTION TO FIX MEMORY SYNCHRONIZATION ISSUES
CREATE OR REPLACE FUNCTION fix_memory_synchronization()
RETURNS TEXT AS $$
DECLARE
    audit_record RECORD;
    fixed_count INTEGER := 0;
    result_text TEXT := '';
BEGIN
    -- Get all synchronization issues
    FOR audit_record IN 
        SELECT * FROM audit_memory_synchronization() 
        WHERE sync_status IN ('MISSING_IN_KGD', 'ORPHANED_IN_KGD')
    LOOP
        IF audit_record.sync_status = 'MISSING_IN_KGD' THEN
            -- Create missing record in knowledge_graph_documents
            INSERT INTO knowledge_graph_documents (
                document_id,
                content_type,
                filename,
                file_hash,
                file_size_bytes,
                processing_status,
                created_at,
                updated_at
            ) VALUES (
                audit_record.memory_id,
                'memory',
                'memory_' || audit_record.memory_id::text,
                MD5('memory_' || audit_record.memory_id::text),
                0, -- Memory records don't have file size
                'completed',
                NOW(),
                NOW()
            );
            
            fixed_count := fixed_count + 1;
            result_text := result_text || 'CREATED: knowledge_graph_documents record for memory_id ' || audit_record.memory_id::text || E'\n';
            
        ELSIF audit_record.sync_status = 'ORPHANED_IN_KGD' AND audit_record.content_type = 'memory' THEN
            -- Delete orphaned memory records (only if content_type = 'memory')
            DELETE FROM knowledge_graph_documents 
            WHERE document_id = audit_record.kgd_document_id 
              AND content_type = 'memory';
            
            fixed_count := fixed_count + 1;
            result_text := result_text || 'DELETED: Orphaned memory record ' || audit_record.kgd_document_id::text || E'\n';
        END IF;
    END LOOP;
    
    RETURN 'Fixed ' || fixed_count || ' synchronization issues:' || E'\n' || result_text;
END;
$$ LANGUAGE plpgsql;

-- 3. CREATE TRIGGER TO MAINTAIN SYNCHRONIZATION
CREATE OR REPLACE FUNCTION sync_memory_to_knowledge_graph()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        -- When a memory is created, ensure it exists in knowledge_graph_documents
        INSERT INTO knowledge_graph_documents (
            document_id,
            content_type,
            filename,
            file_hash,
            file_size_bytes,
            processing_status,
            created_at,
            updated_at
        ) VALUES (
            NEW.memory_id,
            'memory',
            'memory_' || NEW.memory_id::text,
            MD5(COALESCE(NEW.content, '') || NEW.memory_id::text),
            LENGTH(COALESCE(NEW.content, '')),
            'completed',
            NEW.created_at,
            NEW.updated_at
        )
        ON CONFLICT (document_id) DO UPDATE SET
            content_type = EXCLUDED.content_type,
            filename = EXCLUDED.filename,
            updated_at = EXCLUDED.updated_at;
            
        RETURN NEW;
        
    ELSIF TG_OP = 'DELETE' THEN
        -- When a memory is deleted, remove it from knowledge_graph_documents
        DELETE FROM knowledge_graph_documents 
        WHERE document_id = OLD.memory_id AND content_type = 'memory';
        
        RETURN OLD;
    END IF;
    
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- 4. CREATE THE TRIGGER
DROP TRIGGER IF EXISTS trigger_sync_memory_to_knowledge_graph ON notebook_memories;
CREATE TRIGGER trigger_sync_memory_to_knowledge_graph
    AFTER INSERT OR DELETE ON notebook_memories
    FOR EACH ROW EXECUTE FUNCTION sync_memory_to_knowledge_graph();

-- 5. ADD UNIQUE CONSTRAINT TO PREVENT DUPLICATE MEMORY RECORDS
ALTER TABLE knowledge_graph_documents 
ADD CONSTRAINT unique_memory_document_id 
UNIQUE (document_id, content_type);

-- 6. CREATE INDEX FOR BETTER PERFORMANCE ON MEMORY LOOKUPS
CREATE INDEX IF NOT EXISTS idx_knowledge_graph_documents_memory 
ON knowledge_graph_documents (document_id, content_type) 
WHERE content_type = 'memory';

-- 7. IMMEDIATE AUDIT AND FIX
DO $$
DECLARE
    fix_result TEXT;
BEGIN
    -- Run the synchronization fix
    SELECT fix_memory_synchronization() INTO fix_result;
    RAISE NOTICE 'Synchronization Fix Result: %', fix_result;
END;
$$;

-- 8. FINAL AUDIT REPORT
SELECT 
    sync_status,
    COUNT(*) as count,
    STRING_AGG(memory_id::text, ', ') as memory_ids
FROM audit_memory_synchronization()
GROUP BY sync_status
ORDER BY sync_status;