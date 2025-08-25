-- Memory Synchronization Prevention and Monitoring System
-- This script creates robust monitoring and prevention mechanisms

-- 1. CREATE A COMPREHENSIVE MEMORY VALIDATION VIEW
CREATE OR REPLACE VIEW v_memory_health_check AS
SELECT 
    'TOTAL_MEMORIES' as metric,
    COUNT(*)::text as value,
    'Number of memories in notebook_memories table' as description
FROM notebook_memories

UNION ALL

SELECT 
    'TOTAL_MEMORY_DOCS' as metric,
    COUNT(*)::text as value,
    'Number of memory records in knowledge_graph_documents table' as description  
FROM knowledge_graph_documents
WHERE content_type = 'memory'

UNION ALL

SELECT 
    'SYNCHRONIZED_MEMORIES' as metric,
    COUNT(*)::text as value,
    'Number of properly synchronized memory records' as description
FROM notebook_memories nm
INNER JOIN knowledge_graph_documents kgd 
    ON nm.memory_id = kgd.document_id AND kgd.content_type = 'memory'

UNION ALL

SELECT 
    'MISSING_IN_KGD' as metric,
    COUNT(*)::text as value,
    'Memories in notebook_memories but missing in knowledge_graph_documents' as description
FROM notebook_memories nm
LEFT JOIN knowledge_graph_documents kgd 
    ON nm.memory_id = kgd.document_id AND kgd.content_type = 'memory'
WHERE kgd.document_id IS NULL

UNION ALL

SELECT 
    'ORPHANED_IN_KGD' as metric,
    COUNT(*)::text as value,
    'Memory records in knowledge_graph_documents with no corresponding notebook_memories' as description
FROM knowledge_graph_documents kgd
LEFT JOIN notebook_memories nm ON kgd.document_id = nm.memory_id
WHERE kgd.content_type = 'memory' AND nm.memory_id IS NULL;

-- 2. CREATE FUNCTION TO CHECK MEMORY HEALTH
CREATE OR REPLACE FUNCTION check_memory_health()
RETURNS TABLE(
    status VARCHAR(10),
    message TEXT
) AS $$
DECLARE
    missing_count INTEGER;
    orphaned_count INTEGER;
    total_memories INTEGER;
    total_memory_docs INTEGER;
BEGIN
    -- Get counts
    SELECT COUNT(*) INTO total_memories FROM notebook_memories;
    SELECT COUNT(*) INTO total_memory_docs FROM knowledge_graph_documents WHERE content_type = 'memory';
    
    SELECT COUNT(*) INTO missing_count 
    FROM notebook_memories nm
    LEFT JOIN knowledge_graph_documents kgd 
        ON nm.memory_id = kgd.document_id AND kgd.content_type = 'memory'
    WHERE kgd.document_id IS NULL;
    
    SELECT COUNT(*) INTO orphaned_count
    FROM knowledge_graph_documents kgd
    LEFT JOIN notebook_memories nm ON kgd.document_id = nm.memory_id
    WHERE kgd.content_type = 'memory' AND nm.memory_id IS NULL;
    
    -- Return status
    IF missing_count = 0 AND orphaned_count = 0 THEN
        RETURN QUERY SELECT 'HEALTHY'::VARCHAR(10), 
            ('All ' || total_memories || ' memories are properly synchronized')::TEXT;
    ELSE
        RETURN QUERY SELECT 'UNHEALTHY'::VARCHAR(10),
            ('Found ' || missing_count || ' missing and ' || orphaned_count || ' orphaned memory records')::TEXT;
        
        IF missing_count > 0 THEN
            RETURN QUERY SELECT 'WARNING'::VARCHAR(10),
                ('Run fix_memory_synchronization() to create missing knowledge_graph_documents records')::TEXT;
        END IF;
        
        IF orphaned_count > 0 THEN
            RETURN QUERY SELECT 'WARNING'::VARCHAR(10),
                ('Run fix_memory_synchronization() to remove orphaned memory records')::TEXT;
        END IF;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- 3. CREATE FUNCTION TO SAFELY CREATE MEMORY WITH SYNCHRONIZATION
CREATE OR REPLACE FUNCTION create_memory_synchronized(
    p_notebook_id UUID,
    p_memory_id VARCHAR(255),
    p_name VARCHAR(255),
    p_description TEXT,
    p_content TEXT,
    p_milvus_collection VARCHAR(100) DEFAULT NULL
)
RETURNS TABLE(
    success BOOLEAN,
    message TEXT,
    memory_id VARCHAR(255)
) AS $$
DECLARE
    existing_memory_id VARCHAR(255);
BEGIN
    -- Check if memory already exists
    SELECT nm.memory_id INTO existing_memory_id 
    FROM notebook_memories nm 
    WHERE nm.memory_id = p_memory_id;
    
    IF existing_memory_id IS NOT NULL THEN
        RETURN QUERY SELECT FALSE, 'Memory with ID ' || p_memory_id || ' already exists', p_memory_id;
        RETURN;
    END IF;
    
    -- Start transaction
    BEGIN
        -- 1. Insert into notebook_memories
        INSERT INTO notebook_memories (
            notebook_id,
            memory_id,
            name,
            description,
            content,
            milvus_collection,
            created_at,
            updated_at
        ) VALUES (
            p_notebook_id,
            p_memory_id,
            p_name,
            p_description,
            p_content,
            p_milvus_collection,
            NOW(),
            NOW()
        );
        
        -- 2. Insert into knowledge_graph_documents (trigger will handle this, but double-check)
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
            p_memory_id,
            'memory',
            'memory_' || p_memory_id,
            MD5(p_content),
            LENGTH(p_content),
            'completed',
            NOW(),
            NOW()
        )
        ON CONFLICT (document_id, content_type) DO UPDATE SET
            filename = EXCLUDED.filename,
            file_hash = EXCLUDED.file_hash,
            file_size_bytes = EXCLUDED.file_size_bytes,
            updated_at = EXCLUDED.updated_at;
        
        RETURN QUERY SELECT TRUE, 'Memory created and synchronized successfully', p_memory_id;
        
    EXCEPTION WHEN OTHERS THEN
        RETURN QUERY SELECT FALSE, 'Error creating memory: ' || SQLERRM, p_memory_id;
    END;
END;
$$ LANGUAGE plpgsql;

-- 4. CREATE DAILY MONITORING FUNCTION
CREATE OR REPLACE FUNCTION daily_memory_health_report()
RETURNS TEXT AS $$
DECLARE
    health_status RECORD;
    report_text TEXT := '';
    report_date TEXT := TO_CHAR(NOW(), 'YYYY-MM-DD HH24:MI:SS');
BEGIN
    report_text := '=== MEMORY SYNCHRONIZATION HEALTH REPORT ===\n';
    report_text := report_text || 'Generated: ' || report_date || '\n\n';
    
    -- Add health metrics
    FOR health_status IN 
        SELECT metric, value, description FROM v_memory_health_check 
        ORDER BY metric
    LOOP
        report_text := report_text || health_status.metric || ': ' || health_status.value || 
                      ' (' || health_status.description || ')\n';
    END LOOP;
    
    report_text := report_text || '\n=== HEALTH CHECK STATUS ===\n';
    
    -- Add health check results
    FOR health_status IN 
        SELECT status, message FROM check_memory_health()
    LOOP
        report_text := report_text || health_status.status || ': ' || health_status.message || '\n';
    END LOOP;
    
    RETURN report_text;
END;
$$ LANGUAGE plpgsql;

-- 5. SHOW CURRENT HEALTH STATUS
SELECT '=== MEMORY SYNCHRONIZATION STATUS ===' as status_report;
SELECT * FROM v_memory_health_check ORDER BY metric;
SELECT '=== HEALTH CHECK RESULTS ===' as health_check;
SELECT * FROM check_memory_health();