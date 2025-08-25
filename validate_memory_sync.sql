-- Memory Synchronization Validation and Test Script
-- This script validates that the memory synchronization system is working correctly

-- 1. DISPLAY CURRENT STATUS
SELECT '=== INITIAL MEMORY SYNCHRONIZATION STATUS ===' as status_header;
SELECT * FROM v_memory_health_check ORDER BY metric;

-- 2. SHOW DETAILED AUDIT
SELECT '=== DETAILED SYNCHRONIZATION AUDIT ===' as audit_header;
SELECT sync_status, memory_id, notebook_id, kgd_document_id, content_type, filename, action_needed
FROM audit_memory_synchronization()
ORDER BY sync_status, memory_id;

-- 3. HEALTH CHECK
SELECT '=== HEALTH CHECK RESULTS ===' as health_header;
SELECT * FROM check_memory_health();

-- 4. VERIFY TRIGGER EXISTS AND IS ACTIVE
SELECT '=== TRIGGER VALIDATION ===' as trigger_header;
SELECT 
    schemaname,
    tablename,
    triggername,
    triggerdef
FROM pg_catalog.pg_triggers 
WHERE tablename = 'notebook_memories' 
  AND triggername = 'trigger_sync_memory_to_knowledge_graph';

-- 5. VERIFY FUNCTION EXISTS
SELECT '=== FUNCTION VALIDATION ===' as function_header;
SELECT 
    proname as function_name,
    pronargs as argument_count,
    prorettype::regtype as return_type
FROM pg_proc 
WHERE proname IN ('audit_memory_synchronization', 'fix_memory_synchronization', 'create_memory_synchronized', 'check_memory_health')
ORDER BY proname;

-- 6. VERIFY CONSTRAINTS
SELECT '=== CONSTRAINT VALIDATION ===' as constraint_header;
SELECT 
    conname as constraint_name,
    contype as constraint_type,
    pg_get_constraintdef(oid) as constraint_definition
FROM pg_constraint 
WHERE conrelid = 'knowledge_graph_documents'::regclass 
  AND conname LIKE '%memory%';

-- 7. SHOW INDEXES FOR PERFORMANCE
SELECT '=== INDEX VALIDATION ===' as index_header;
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes 
WHERE tablename = 'knowledge_graph_documents' 
  AND indexname LIKE '%memory%';

-- 8. FINAL COMPREHENSIVE HEALTH REPORT
SELECT '=== COMPREHENSIVE HEALTH REPORT ===' as final_header;
SELECT daily_memory_health_report();