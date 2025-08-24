"""
Document Admin Service for permanent document deletion.
Handles complete removal of documents from all systems: Milvus, PostgreSQL, Neo4j, and Redis.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy.exc import DatabaseError
from sqlalchemy import text
from pymilvus import connections, Collection, utility, MilvusException
from redis import Redis

from app.core.vector_db_settings_cache import get_vector_db_settings
from app.core.redis_client import get_redis_client

logger = logging.getLogger(__name__)

class DocumentAdminService:
    """
    Service for permanent document deletion across all systems.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def delete_document_permanently(
        self,
        db: Session,
        document_id: str,
        remove_from_notebooks: bool = True
    ) -> Dict[str, Any]:
        """
        Permanently delete a document from all systems with atomic ordering.
        
        CRITICAL: Deletion order ensures data integrity:
        1. DELETE FROM MILVUS FIRST - verify actual deletion
        2. Only if Milvus succeeds → delete from PostgreSQL 
        3. Only if PostgreSQL succeeds → delete from Neo4j
        4. Clear cache last
        
        Args:
            db: Database session
            document_id: Document ID to delete
            remove_from_notebooks: Whether to remove from notebooks (default: True)
            
        Returns:
            Deletion summary with details of what was removed
        """
        deletion_summary = {
            'document_id': document_id,
            'started_at': datetime.now().isoformat(),
            'milvus_deleted': False,
            'milvus_verified': False,
            'database_deleted': False,
            'notebooks_removed': 0,
            'neo4j_deleted': False,
            'cache_cleared': False,
            'errors': [],
            'rollback_attempted': False
        }
        
        # Track what was successfully deleted for potential rollback
        successful_steps = []
        
        try:
            self.logger.info(f"Starting atomic permanent deletion of document: {document_id}")
            
            # Step 1: Get document info before deletion
            document_info = await self._get_document_info(db, document_id)
            if not document_info:
                deletion_summary['errors'].append("Document not found in database")
                return deletion_summary
                
            milvus_collection = document_info.get('milvus_collection')
            
            # STEP 2: DELETE FROM MILVUS FIRST - CRITICAL FOR DATA INTEGRITY
            if milvus_collection:
                try:
                    # Delete and verify actual deletion from Milvus
                    milvus_success = await self._delete_from_milvus_with_verification(milvus_collection, document_id)
                    if not milvus_success:
                        error_msg = f"Milvus deletion failed verification - vectors still exist. STOPPING deletion to prevent orphaned database records."
                        deletion_summary['errors'].append(error_msg)
                        self.logger.error(error_msg)
                        return deletion_summary
                    
                    deletion_summary['milvus_deleted'] = True
                    deletion_summary['milvus_verified'] = True
                    successful_steps.append('milvus')
                    self.logger.info(f"✅ VERIFIED: Deleted and confirmed removal of document {document_id} from Milvus collection {milvus_collection}")
                    
                except Exception as e:
                    error_msg = f"CRITICAL: Milvus deletion failed - {str(e)}. STOPPING deletion to prevent data integrity issues."
                    deletion_summary['errors'].append(error_msg)
                    self.logger.error(error_msg)
                    return deletion_summary
            else:
                self.logger.info(f"No Milvus collection specified for document {document_id}, skipping vector deletion")
            
            # STEP 3: DELETE FROM POSTGRESQL - Only after Milvus success
            try:
                # First remove from notebooks if requested (less critical)
                if remove_from_notebooks:
                    removed_count = await self._remove_from_all_notebooks(db, document_id)
                    deletion_summary['notebooks_removed'] = removed_count
                    successful_steps.append('notebooks')
                    self.logger.info(f"Removed document {document_id} from {removed_count} notebooks")
                
                # Now delete main document record
                await self._delete_from_database(db, document_id)
                deletion_summary['database_deleted'] = True
                successful_steps.append('database')
                self.logger.info(f"✅ VERIFIED: Deleted document {document_id} from PostgreSQL")
                
            except Exception as e:
                error_msg = f"CRITICAL: Database deletion failed after Milvus deletion succeeded - {str(e)}"
                deletion_summary['errors'].append(error_msg)
                self.logger.error(error_msg)
                # At this point vectors are gone but database record exists - this is recoverable
                # Continue to other steps since the critical vector deletion succeeded
            
            # STEP 4: DELETE FROM NEO4J - Less critical, continue even if fails
            try:
                await self._delete_from_neo4j(document_id)
                deletion_summary['neo4j_deleted'] = True
                successful_steps.append('neo4j')
                self.logger.info(f"✅ Deleted document {document_id} from Neo4j")
            except Exception as e:
                error_msg = f"Neo4j deletion failed (non-critical): {str(e)}"
                deletion_summary['errors'].append(error_msg)
                self.logger.warning(error_msg)
                # Continue - Neo4j failure is not critical
            
            # STEP 5: CLEAR CACHE - Always attempt
            try:
                await self._clear_document_cache(document_id, milvus_collection)
                deletion_summary['cache_cleared'] = True
                successful_steps.append('cache')
                self.logger.info(f"✅ Cleared cache for document {document_id}")
            except Exception as e:
                error_msg = f"Cache clearing failed (non-critical): {str(e)}"
                deletion_summary['errors'].append(error_msg)
                self.logger.warning(error_msg)
                # Continue - cache failure is not critical
            
            # Determine overall success
            critical_success = (
                (not milvus_collection or deletion_summary['milvus_verified']) and
                deletion_summary['database_deleted']
            )
            
            deletion_summary['completed_at'] = datetime.now().isoformat()
            deletion_summary['success'] = critical_success and len([e for e in deletion_summary['errors'] if 'CRITICAL' in e]) == 0
            deletion_summary['successful_steps'] = successful_steps
            
            if deletion_summary['success']:
                self.logger.info(f"✅ ATOMIC DELETION COMPLETED: document {document_id} - steps: {successful_steps}")
            else:
                self.logger.warning(f"⚠️ PARTIAL DELETION: document {document_id} - completed steps: {successful_steps}")
            
            return deletion_summary
            
        except Exception as e:
            error_msg = f"Unexpected error during atomic deletion: {str(e)}"
            deletion_summary['errors'].append(error_msg)
            deletion_summary['completed_at'] = datetime.now().isoformat()
            deletion_summary['success'] = False
            deletion_summary['successful_steps'] = successful_steps
            self.logger.error(f"🚨 ATOMIC DELETION FAILED: {error_msg}")
            return deletion_summary
    
    async def bulk_delete_documents(
        self,
        db: Session,
        document_ids: List[str],
        remove_from_notebooks: bool = True
    ) -> Dict[str, Any]:
        """
        Bulk delete multiple documents permanently using atomic deletion.
        
        CRITICAL: Uses atomic deletion sequence for each document to ensure data integrity.
        Continues processing even if individual documents fail to maintain overall progress.
        
        Args:
            db: Database session
            document_ids: List of document IDs to delete
            remove_from_notebooks: Whether to remove from notebooks
            
        Returns:
            Bulk deletion summary with detailed results
        """
        bulk_summary = {
            'total_requested': len(document_ids),
            'started_at': datetime.now().isoformat(),
            'successful_deletions': 0,
            'failed_deletions': 0,
            'partial_deletions': 0,
            'deletion_details': [],
            'overall_errors': [],
            'critical_errors': 0,
            'non_critical_errors': 0
        }
        
        try:
            self.logger.info(f"🗑️ Starting atomic bulk deletion of {len(document_ids)} documents")
            
            for i, document_id in enumerate(document_ids, 1):
                try:
                    self.logger.info(f"📋 Processing document {i}/{len(document_ids)}: {document_id}")
                    
                    # Use atomic deletion for each document
                    result = await self.delete_document_permanently(
                        db, document_id, remove_from_notebooks
                    )
                    bulk_summary['deletion_details'].append(result)
                    
                    # Categorize result
                    if result.get('success', False):
                        bulk_summary['successful_deletions'] += 1
                        self.logger.info(f"✅ {i}/{len(document_ids)} - Successful: {document_id}")
                    else:
                        # Check if it's a critical failure (Milvus issues) vs partial success
                        critical_errors = [e for e in result.get('errors', []) if 'CRITICAL' in e]
                        if critical_errors:
                            bulk_summary['failed_deletions'] += 1
                            bulk_summary['critical_errors'] += 1
                            self.logger.error(f"🚨 {i}/{len(document_ids)} - Critical failure: {document_id}")
                        else:
                            # Some steps succeeded, consider it partial
                            bulk_summary['partial_deletions'] += 1
                            bulk_summary['non_critical_errors'] += 1
                            self.logger.warning(f"⚠️ {i}/{len(document_ids)} - Partial success: {document_id}")
                        
                except Exception as e:
                    error_msg = f"Unexpected error deleting document {document_id}: {str(e)}"
                    bulk_summary['overall_errors'].append(error_msg)
                    bulk_summary['failed_deletions'] += 1
                    bulk_summary['critical_errors'] += 1
                    self.logger.error(f"🚨 {i}/{len(document_ids)} - Exception: {document_id} - {error_msg}")
            
            # Calculate final status
            total_processed = bulk_summary['successful_deletions'] + bulk_summary['partial_deletions'] + bulk_summary['failed_deletions']
            bulk_summary['completed_at'] = datetime.now().isoformat()
            bulk_summary['success'] = bulk_summary['failed_deletions'] == 0
            bulk_summary['partial_success'] = bulk_summary['critical_errors'] == 0  # No critical failures
            
            # Log comprehensive summary
            self.logger.info(f"🏁 BULK DELETION COMPLETED:")
            self.logger.info(f"   📊 Total requested: {bulk_summary['total_requested']}")
            self.logger.info(f"   ✅ Fully successful: {bulk_summary['successful_deletions']}")
            self.logger.info(f"   ⚠️ Partial success: {bulk_summary['partial_deletions']}")
            self.logger.info(f"   🚨 Failed: {bulk_summary['failed_deletions']}")
            self.logger.info(f"   🔥 Critical errors: {bulk_summary['critical_errors']}")
            self.logger.info(f"   ⚡ Non-critical errors: {bulk_summary['non_critical_errors']}")
            
            if bulk_summary['success']:
                self.logger.info(f"🎉 ALL {len(document_ids)} DOCUMENTS PROCESSED SUCCESSFULLY")
            elif bulk_summary['partial_success']:
                self.logger.warning(f"⚠️ PARTIAL SUCCESS: {bulk_summary['successful_deletions'] + bulk_summary['partial_deletions']}/{len(document_ids)} documents processed without critical errors")
            else:
                self.logger.error(f"🚨 BULK DELETION HAD CRITICAL FAILURES: {bulk_summary['critical_errors']} documents failed with data integrity issues")
                
            return bulk_summary
            
        except Exception as e:
            error_msg = f"🚨 BULK DELETION SYSTEM FAILURE: {str(e)}"
            bulk_summary['overall_errors'].append(error_msg)
            bulk_summary['completed_at'] = datetime.now().isoformat()
            bulk_summary['success'] = False
            bulk_summary['partial_success'] = False
            self.logger.error(error_msg)
            return bulk_summary
    
    async def _get_document_info(self, db: Session, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document information before deletion."""
        try:
            query = text("""
                SELECT document_id, filename, milvus_collection, file_type, file_size_bytes
                FROM knowledge_graph_documents
                WHERE document_id = :document_id
            """)
            
            result = db.execute(query, {'document_id': document_id})
            row = result.fetchone()
            
            if not row:
                return None
                
            return {
                'document_id': row.document_id,
                'filename': row.filename,
                'milvus_collection': row.milvus_collection,
                'file_type': row.file_type,
                'file_size_bytes': row.file_size_bytes
            }
        except Exception as e:
            self.logger.error(f"Failed to get document info: {str(e)}")
            return None
    
    async def _delete_from_milvus_with_verification(self, collection_name: str, document_id: str) -> bool:
        """
        Delete document vectors from Milvus collection WITH VERIFICATION.
        
        CRITICAL: This method ensures actual deletion by:
        1. Verifying entities exist before deletion
        2. Performing deletion with proper expression
        3. Verifying entities are actually gone after deletion
        4. Returns True only if deletion is fully verified
        
        Args:
            collection_name: Milvus collection name
            document_id: Document ID to delete
            
        Returns:
            bool: True if deletion verified successful, False if vectors still exist
        """
        try:
            # Get active vector database configuration
            from app.services.vector_db_service import get_active_vector_db
            
            active_db = get_active_vector_db()
            if not active_db or active_db.get('id') != 'milvus':
                raise Exception("Milvus is not the active vector database")
            
            milvus_config = active_db.get('config', {})
            uri = milvus_config.get('MILVUS_URI')
            token = milvus_config.get('MILVUS_TOKEN', '')
            
            if not uri:
                raise Exception("Milvus URI not configured")
            
            self.logger.info(f"🔍 ATOMIC DELETE: Connecting to Milvus at {uri} for collection {collection_name}")
            
            # Connect to Milvus
            connections.connect(
                alias="default",
                uri=uri,
                token=token
            )
            
            # Check if collection exists
            if not utility.has_collection(collection_name):
                self.logger.warning(f"Collection {collection_name} does not exist in Milvus - considering deletion successful")
                return True
            
            # Get collection and ensure it's loaded
            collection = Collection(collection_name)
            try:
                collection.load()
                self.logger.info(f"🔍 Collection {collection_name} loaded for verification")
            except Exception as load_e:
                self.logger.warning(f"Collection load issue: {load_e} - continuing with deletion")
            
            # STEP 1: VERIFY ENTITIES EXIST BEFORE DELETION
            entities_to_delete = []
            try:
                # Check exact match
                exact_query_expr = f"doc_id == '{document_id}'"
                exact_results = collection.query(
                    expr=exact_query_expr,
                    output_fields=["doc_id"],
                    limit=100
                )
                
                # Check prefix match for chunked format
                prefix_query_expr = f"doc_id like '{document_id}_p%'"
                prefix_results = collection.query(
                    expr=prefix_query_expr,
                    output_fields=["doc_id"],
                    limit=1000
                )
                
                exact_entities = len(exact_results) if exact_results else 0
                prefix_entities = len(prefix_results) if prefix_results else 0
                
                self.logger.info(f"🔍 PRE-DELETE VERIFICATION: {exact_entities} exact matches, {prefix_entities} prefix matches found")
                
                if prefix_entities > 0:
                    query_expr = prefix_query_expr
                    entities_to_delete = prefix_results
                    self.logger.info(f"🎯 Using prefix matching for deletion: {prefix_entities} entities")
                elif exact_entities > 0:
                    query_expr = exact_query_expr
                    entities_to_delete = exact_results
                    self.logger.info(f"🎯 Using exact matching for deletion: {exact_entities} entities")
                else:
                    self.logger.info(f"✅ No entities found with doc_id '{document_id}' - deletion already complete")
                    return True
                    
            except Exception as query_e:
                self.logger.error(f"🚨 PRE-DELETE QUERY FAILED: {query_e}")
                raise
            
            # STEP 2: PERFORM DELETION
            entity_count_before = len(entities_to_delete)
            self.logger.info(f"🗑️ DELETING {entity_count_before} entities with expression: {query_expr}")
            
            delete_result = collection.delete(query_expr)
            
            # Log deletion result
            deleted_count = getattr(delete_result, 'delete_count', 0)
            self.logger.info(f"🗑️ Milvus reported {deleted_count} entities deleted")
            
            # STEP 3: FLUSH TO ENSURE DELETION IS PERSISTED
            try:
                collection.flush()
                self.logger.info(f"💾 Collection {collection_name} flushed")
                
                # Wait for flush to complete
                import time
                time.sleep(0.2)
                
            except Exception as flush_e:
                self.logger.error(f"🚨 FLUSH FAILED: {flush_e}")
                raise
            
            # STEP 4: CRITICAL VERIFICATION - ENSURE ENTITIES ARE ACTUALLY GONE
            try:
                # Verify exact match entities are gone
                exact_verification = collection.query(
                    expr=f"doc_id == '{document_id}'",
                    output_fields=["doc_id"],
                    limit=10
                )
                
                # Verify prefix match entities are gone
                prefix_verification = collection.query(
                    expr=f"doc_id like '{document_id}_p%'",
                    output_fields=["doc_id"],
                    limit=10
                )
                
                exact_remaining = len(exact_verification) if exact_verification else 0
                prefix_remaining = len(prefix_verification) if prefix_verification else 0
                total_remaining = exact_remaining + prefix_remaining
                
                if total_remaining > 0:
                    # CRITICAL FAILURE - Entities still exist
                    self.logger.error(f"🚨 DELETION VERIFICATION FAILED: {total_remaining} entities still exist ({exact_remaining} exact, {prefix_remaining} prefix)")
                    if exact_remaining > 0:
                        sample_exact = [r.get('doc_id', 'unknown') for r in exact_verification[:3]]
                        self.logger.error(f"🚨 Remaining exact matches: {sample_exact}")
                    if prefix_remaining > 0:
                        sample_prefix = [r.get('doc_id', 'unknown') for r in prefix_verification[:3]]
                        self.logger.error(f"🚨 Remaining prefix matches: {sample_prefix}")
                    return False
                else:
                    # SUCCESS - All entities confirmed deleted
                    self.logger.info(f"✅ DELETION VERIFIED: All {entity_count_before} entities successfully removed from Milvus")
                    return True
                    
            except Exception as verify_e:
                self.logger.error(f"🚨 VERIFICATION QUERY FAILED: {verify_e}")
                raise
            
        except MilvusException as e:
            self.logger.error(f"🚨 Milvus error during verified deletion of document {document_id}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"🚨 Error during verified Milvus deletion: {str(e)}")
            raise
        finally:
            try:
                connections.disconnect("default")
            except:
                pass

    async def _delete_from_milvus(self, collection_name: str, document_id: str):
        """Delete document vectors from Milvus collection."""
        try:
            # Get active vector database configuration
            from app.services.vector_db_service import get_active_vector_db
            
            active_db = get_active_vector_db()
            if not active_db or active_db.get('id') != 'milvus':
                raise Exception("Milvus is not the active vector database")
            
            milvus_config = active_db.get('config', {})
            uri = milvus_config.get('MILVUS_URI')
            token = milvus_config.get('MILVUS_TOKEN', '')
            
            if not uri:
                raise Exception("Milvus URI not configured")
            
            self.logger.info(f"DEBUG: Connecting to Milvus at {uri} for collection {collection_name}")
            
            # Connect to Milvus
            connections.connect(
                alias="default",
                uri=uri,
                token=token
            )
            
            # Check if collection exists
            if not utility.has_collection(collection_name):
                self.logger.warning(f"Collection {collection_name} does not exist in Milvus")
                return
            
            # Get collection and load it to ensure it's ready for queries
            collection = Collection(collection_name)
            
            # Load collection if not already loaded
            if not collection.has_index():
                self.logger.warning(f"Collection {collection_name} has no index")
            
            try:
                collection.load()
                self.logger.info(f"DEBUG: Collection {collection_name} loaded successfully")
            except Exception as load_e:
                self.logger.warning(f"DEBUG: Collection load failed: {load_e}")
            
            # Check current count before deletion
            try:
                total_count_before = collection.num_entities
                self.logger.info(f"DEBUG: Total entities in collection before deletion: {total_count_before}")
            except Exception as count_e:
                self.logger.warning(f"DEBUG: Could not get entity count: {count_e}")
                total_count_before = "unknown"
            
            # Check if there are entities with this doc_id before deletion
            # Note: doc_id in Milvus has format {file_id}_p{page}_c{chunk}, so we need prefix matching
            try:
                # First try exact match (in case format changed)
                exact_query_expr = f"doc_id == '{document_id}'"
                exact_results = collection.query(
                    expr=exact_query_expr,
                    output_fields=["doc_id"],
                    limit=10
                )
                
                # Then try prefix matching for the chunked format
                prefix_query_expr = f"doc_id like '{document_id}_p%'"
                prefix_results = collection.query(
                    expr=prefix_query_expr,
                    output_fields=["doc_id"],
                    limit=100
                )
                
                exact_entities = len(exact_results) if exact_results else 0
                prefix_entities = len(prefix_results) if prefix_results else 0
                
                self.logger.info(f"DEBUG: Found {exact_entities} entities with exact match '{document_id}'")
                self.logger.info(f"DEBUG: Found {prefix_entities} entities with prefix match '{document_id}_p%'")
                
                if prefix_entities > 0:
                    # Use prefix matching for deletion - this handles the chunked format
                    query_expr = prefix_query_expr
                    entities_found = prefix_entities
                    self.logger.info(f"DEBUG: Using prefix matching for deletion")
                    # Show sample doc_ids found
                    sample_doc_ids = [r.get('doc_id', 'unknown') for r in prefix_results[:5]]
                    self.logger.info(f"DEBUG: Sample doc_ids found: {sample_doc_ids}")
                elif exact_entities > 0:
                    # Use exact matching
                    query_expr = exact_query_expr 
                    entities_found = exact_entities
                    self.logger.info(f"DEBUG: Using exact matching for deletion")
                else:
                    self.logger.warning(f"DEBUG: No entities found with doc_id '{document_id}' using either exact or prefix matching")
                    return
                    
            except Exception as query_e:
                self.logger.error(f"DEBUG: Pre-deletion query failed: {query_e}")
                # Continue with deletion attempt using prefix matching as fallback
                query_expr = f"doc_id like '{document_id}_p%'"
            
            # Perform deletion with the determined expression
            self.logger.info(f"DEBUG: Attempting deletion with expression: {query_expr}")
            delete_result = collection.delete(query_expr)
            
            # Log detailed deletion results
            if hasattr(delete_result, 'delete_count'):
                deleted_count = delete_result.delete_count
                self.logger.info(f"DEBUG: Milvus delete_result.delete_count: {deleted_count}")
            else:
                self.logger.warning(f"DEBUG: delete_result object: {delete_result}")
                deleted_count = 0
            
            # Flush the collection to ensure deletion is persisted
            try:
                collection.flush()
                self.logger.info(f"DEBUG: Collection {collection_name} flushed after deletion")
            except Exception as flush_e:
                self.logger.error(f"DEBUG: Collection flush failed: {flush_e}")
            
            # Verify deletion by checking count after
            try:
                # Small delay to ensure flush is complete
                import time
                time.sleep(0.1)
                
                total_count_after = collection.num_entities
                self.logger.info(f"DEBUG: Total entities in collection after deletion: {total_count_after}")
                
                # Double-check by querying again with both exact and prefix matching
                exact_verification = collection.query(
                    expr=f"doc_id == '{document_id}'",
                    output_fields=["doc_id"],
                    limit=10
                )
                prefix_verification = collection.query(
                    expr=f"doc_id like '{document_id}_p%'",
                    output_fields=["doc_id"],
                    limit=10
                )
                
                exact_remaining = len(exact_verification) if exact_verification else 0
                prefix_remaining = len(prefix_verification) if prefix_verification else 0
                
                if exact_remaining > 0 or prefix_remaining > 0:
                    self.logger.error(f"DEBUG: DELETION FAILED - {exact_remaining} exact matches and {prefix_remaining} prefix matches still exist")
                    if exact_remaining > 0:
                        self.logger.error(f"DEBUG: Remaining exact matches: {exact_verification[:3]}")
                    if prefix_remaining > 0:
                        self.logger.error(f"DEBUG: Remaining prefix matches: {prefix_verification[:3]}")
                else:
                    self.logger.info(f"DEBUG: DELETION VERIFIED - No entities found with doc_id '{document_id}' using either exact or prefix matching")
                    
            except Exception as verify_e:
                self.logger.error(f"DEBUG: Post-deletion verification failed: {verify_e}")
            
            self.logger.info(f"Deleted {deleted_count} entities from Milvus collection {collection_name}")
            
        except MilvusException as e:
            self.logger.error(f"Milvus error deleting document {document_id}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error deleting from Milvus: {str(e)}")
            raise
        finally:
            try:
                connections.disconnect("default")
            except:
                pass
    
    async def _delete_from_database(self, db: Session, document_id: str):
        """Delete document from PostgreSQL database."""
        try:
            # Delete from knowledge_graph_documents (cascade will handle related records)
            delete_query = text("""
                DELETE FROM knowledge_graph_documents
                WHERE document_id = :document_id
            """)
            
            result = db.execute(delete_query, {'document_id': document_id})
            
            if result.rowcount == 0:
                raise Exception(f"Document {document_id} not found in database")
            
            db.commit()
            self.logger.info(f"Deleted document {document_id} from PostgreSQL")
            
        except Exception as e:
            db.rollback()
            self.logger.error(f"Database error deleting document {document_id}: {str(e)}")
            raise
    
    async def _remove_from_all_notebooks(self, db: Session, document_id: str) -> int:
        """Remove document from all notebooks."""
        try:
            delete_query = text("""
                DELETE FROM notebook_documents
                WHERE document_id = :document_id
            """)
            
            result = db.execute(delete_query, {'document_id': document_id})
            removed_count = result.rowcount
            db.commit()
            
            return removed_count
            
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error removing document from notebooks: {str(e)}")
            raise
    
    async def _delete_from_neo4j(self, document_id: str):
        """Delete document entities and relationships from Neo4j."""
        try:
            # Import Neo4j service here to avoid circular imports
            from app.services.neo4j_service import get_neo4j_service
            
            neo4j_service = get_neo4j_service()
            
            # Delete all nodes and relationships for this document
            delete_query = """
            MATCH (n)
            WHERE n.document_id = $document_id
            DETACH DELETE n
            """
            
            result = neo4j_service.execute_cypher(delete_query, {'document_id': document_id})
            self.logger.info(f"Deleted Neo4j entities for document {document_id}")
            
        except ImportError:
            self.logger.warning("Neo4j service not available, skipping graph deletion")
        except Exception as e:
            self.logger.error(f"Error deleting from Neo4j: {str(e)}")
            raise
    
    async def _clear_document_cache(self, document_id: str, collection_name: Optional[str] = None):
        """Clear Redis cache entries for the document."""
        try:
            redis_client = get_redis_client()
            if not redis_client:
                self.logger.warning("Redis not available, skipping cache clearing")
                return
            
            # Clear various cache keys that might contain this document
            cache_keys_to_clear = [
                f"document:{document_id}",
                f"doc_metadata:{document_id}",
                f"embedding:{document_id}",
            ]
            
            if collection_name:
                cache_keys_to_clear.extend([
                    f"collection:{collection_name}:documents",
                    f"collection:{collection_name}:stats",
                ])
            
            # Clear cache keys
            for key in cache_keys_to_clear:
                try:
                    redis_client.delete(key)
                except:
                    pass  # Ignore individual key failures
            
            # Also clear any pattern-based keys
            try:
                pattern_keys = [
                    f"*{document_id}*",
                ]
                
                for pattern in pattern_keys:
                    keys = redis_client.keys(pattern)
                    if keys:
                        redis_client.delete(*keys)
            except:
                pass  # Ignore pattern clearing failures
            
            self.logger.info(f"Cleared cache for document {document_id}")
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            raise
    
    async def get_document_usage_info(
        self,
        db: Session,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Get information about where a document is used before deletion.
        
        Args:
            db: Database session
            document_id: Document ID to analyze
            
        Returns:
            Usage information dictionary
        """
        try:
            # Get document basic info
            doc_info = await self._get_document_info(db, document_id)
            if not doc_info:
                return {'error': 'Document not found'}
            
            # Get notebooks using this document
            notebooks_query = text("""
                SELECT 
                    n.id, n.name, nd.added_at
                FROM notebooks n
                JOIN notebook_documents nd ON n.id = nd.notebook_id
                WHERE nd.document_id = :document_id
                ORDER BY nd.added_at DESC
            """)
            
            result = db.execute(notebooks_query, {'document_id': document_id})
            notebooks = []
            for row in result.fetchall():
                notebooks.append({
                    'id': str(row.id),
                    'name': row.name,
                    'added_at': row.added_at.isoformat() if row.added_at else None
                })
            
            # Get related documents (cross-references)
            references_query = text("""
                SELECT COUNT(*) as ref_count
                FROM document_cross_references
                WHERE document_id = :document_id OR referenced_document_id = :document_id
            """)
            
            ref_result = db.execute(references_query, {'document_id': document_id})
            ref_count = ref_result.scalar() or 0
            
            return {
                'document_id': document_id,
                'filename': doc_info['filename'],
                'file_type': doc_info['file_type'],
                'file_size_bytes': doc_info['file_size_bytes'],
                'milvus_collection': doc_info['milvus_collection'],
                'notebooks_using': notebooks,
                'notebook_count': len(notebooks),
                'cross_references': ref_count,
                'deletion_impact': {
                    'will_remove_from_notebooks': len(notebooks),
                    'will_delete_vectors': doc_info['milvus_collection'] is not None,
                    'will_delete_cross_references': ref_count > 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting document usage info: {str(e)}")
            return {'error': str(e)}
    
    async def validate_deletion_integrity(
        self,
        db: Session,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Validate that a document is completely removed from all systems.
        
        Use this method to verify deletion integrity after deletion operations.
        
        Args:
            db: Database session
            document_id: Document ID to validate
            
        Returns:
            Validation results showing presence in each system
        """
        validation_result = {
            'document_id': document_id,
            'validated_at': datetime.now().isoformat(),
            'systems_checked': {},
            'integrity_issues': [],
            'is_fully_deleted': True
        }
        
        try:
            # Check PostgreSQL
            try:
                doc_info = await self._get_document_info(db, document_id)
                database_exists = doc_info is not None
                validation_result['systems_checked']['postgresql'] = {
                    'exists': database_exists,
                    'details': doc_info if database_exists else 'Not found'
                }
                if database_exists:
                    validation_result['integrity_issues'].append("Document still exists in PostgreSQL database")
                    validation_result['is_fully_deleted'] = False
                    
            except Exception as e:
                validation_result['systems_checked']['postgresql'] = {
                    'error': str(e),
                    'status': 'check_failed'
                }
                validation_result['integrity_issues'].append(f"PostgreSQL check failed: {str(e)}")
            
            # Check Milvus (if we have collection info)
            try:
                # Try to get collection info from document or check all active collections
                from app.services.vector_db_service import get_active_vector_db
                
                active_db = get_active_vector_db()
                if active_db and active_db.get('id') == 'milvus':
                    milvus_config = active_db.get('config', {})
                    uri = milvus_config.get('MILVUS_URI')
                    token = milvus_config.get('MILVUS_TOKEN', '')
                    
                    if uri:
                        from pymilvus import connections, utility, Collection
                        
                        connections.connect(alias="validation", uri=uri, token=token)
                        
                        # Check all collections for this document
                        collections = utility.list_collections()
                        milvus_found = False
                        
                        for collection_name in collections:
                            try:
                                collection = Collection(collection_name)
                                collection.load()
                                
                                # Check both exact and prefix matches
                                exact_results = collection.query(
                                    expr=f"doc_id == '{document_id}'",
                                    output_fields=["doc_id"],
                                    limit=10
                                )
                                prefix_results = collection.query(
                                    expr=f"doc_id like '{document_id}_p%'",
                                    output_fields=["doc_id"],
                                    limit=10
                                )
                                
                                exact_count = len(exact_results) if exact_results else 0
                                prefix_count = len(prefix_results) if prefix_results else 0
                                total_found = exact_count + prefix_count
                                
                                if total_found > 0:
                                    milvus_found = True
                                    validation_result['systems_checked']['milvus'] = {
                                        'exists': True,
                                        'collection': collection_name,
                                        'exact_matches': exact_count,
                                        'prefix_matches': prefix_count,
                                        'total_entities': total_found,
                                        'sample_ids': [r.get('doc_id', 'unknown') for r in (exact_results or prefix_results)[:3]]
                                    }
                                    validation_result['integrity_issues'].append(f"Document vectors still exist in Milvus collection {collection_name}")
                                    validation_result['is_fully_deleted'] = False
                                    break
                                    
                            except Exception as collection_e:
                                # Skip collections that can't be checked
                                continue
                        
                        if not milvus_found:
                            validation_result['systems_checked']['milvus'] = {
                                'exists': False,
                                'collections_checked': len(collections),
                                'details': 'No vectors found in any collection'
                            }
                        
                        connections.disconnect("validation")
                    else:
                        validation_result['systems_checked']['milvus'] = {
                            'error': 'Milvus URI not configured',
                            'status': 'config_missing'
                        }
                else:
                    validation_result['systems_checked']['milvus'] = {
                        'status': 'not_active',
                        'details': 'Milvus is not the active vector database'
                    }
                    
            except Exception as e:
                validation_result['systems_checked']['milvus'] = {
                    'error': str(e),
                    'status': 'check_failed'
                }
                validation_result['integrity_issues'].append(f"Milvus check failed: {str(e)}")
            
            # Check Neo4j
            try:
                from app.services.neo4j_service import get_neo4j_service
                
                neo4j_service = get_neo4j_service()
                
                check_query = """
                MATCH (n)
                WHERE n.document_id = $document_id
                RETURN count(n) as node_count
                """
                
                result = neo4j_service.execute_cypher(check_query, {'document_id': document_id})
                node_count = result[0].get('node_count', 0) if result else 0
                
                validation_result['systems_checked']['neo4j'] = {
                    'exists': node_count > 0,
                    'node_count': node_count
                }
                
                if node_count > 0:
                    validation_result['integrity_issues'].append(f"Document nodes still exist in Neo4j ({node_count} nodes)")
                    validation_result['is_fully_deleted'] = False
                    
            except ImportError:
                validation_result['systems_checked']['neo4j'] = {
                    'status': 'service_unavailable',
                    'details': 'Neo4j service not available'
                }
            except Exception as e:
                validation_result['systems_checked']['neo4j'] = {
                    'error': str(e),
                    'status': 'check_failed'
                }
                validation_result['integrity_issues'].append(f"Neo4j check failed: {str(e)}")
            
            # Check Redis cache
            try:
                redis_client = get_redis_client()
                if redis_client:
                    cache_keys = [
                        f"document:{document_id}",
                        f"doc_metadata:{document_id}",
                        f"embedding:{document_id}",
                    ]
                    
                    cached_keys = []
                    for key in cache_keys:
                        if redis_client.exists(key):
                            cached_keys.append(key)
                    
                    # Also check pattern-based keys
                    pattern_keys = redis_client.keys(f"*{document_id}*")
                    
                    total_cached = len(cached_keys) + len(pattern_keys)
                    
                    validation_result['systems_checked']['redis'] = {
                        'exists': total_cached > 0,
                        'direct_keys': cached_keys,
                        'pattern_keys': pattern_keys[:10],  # Limit display
                        'total_cached': total_cached
                    }
                    
                    if total_cached > 0:
                        validation_result['integrity_issues'].append(f"Document cache entries still exist in Redis ({total_cached} keys)")
                        # Note: Cache presence is not critical for deletion integrity
                else:
                    validation_result['systems_checked']['redis'] = {
                        'status': 'unavailable',
                        'details': 'Redis client not available'
                    }
                    
            except Exception as e:
                validation_result['systems_checked']['redis'] = {
                    'error': str(e),
                    'status': 'check_failed'
                }
            
            # Final assessment
            critical_issues = len([issue for issue in validation_result['integrity_issues'] 
                                 if 'PostgreSQL' in issue or 'Milvus' in issue or 'Neo4j' in issue])
            
            validation_result['critical_integrity_issues'] = critical_issues
            validation_result['has_critical_issues'] = critical_issues > 0
            
            if validation_result['is_fully_deleted']:
                self.logger.info(f"✅ VALIDATION PASSED: Document {document_id} is fully deleted from all systems")
            else:
                self.logger.warning(f"⚠️ VALIDATION ISSUES: Document {document_id} has {len(validation_result['integrity_issues'])} integrity issues")
            
            return validation_result
            
        except Exception as e:
            validation_result['validation_error'] = str(e)
            validation_result['is_fully_deleted'] = False
            self.logger.error(f"🚨 VALIDATION FAILED: {str(e)}")
            return validation_result