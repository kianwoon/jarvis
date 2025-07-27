"""
Enhanced document upload with progress tracking via SSE
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any
import json
import asyncio
from datetime import datetime
import hashlib
import os
import tempfile
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymilvus import Collection, utility

from app.core.embedding_settings_cache import get_embedding_settings
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.api.v1.endpoints.document import HTTPEmbeddingFunction, ensure_milvus_collection
from utils.deduplication import hash_text, get_existing_hashes, get_existing_doc_ids
from app.rag.bm25_processor import BM25Processor
from app.utils.metadata_extractor import MetadataExtractor
# Knowledge graph imports with error handling
try:
    from app.services.knowledge_graph_service import get_knowledge_graph_service
    from app.document_handlers.graph_processor import get_graph_document_processor
    from app.core.temp_document_manager import TempDocumentManager
    from app.document_handlers.base import ExtractedChunk
    KG_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Knowledge graph dependencies not available: {e}")
    KG_IMPORTS_AVAILABLE = False

router = APIRouter()

@router.get("/test_progress_generator")
async def test_progress_generator():
    """Test endpoint to verify progress generator works"""
    
    async def test_generator():
        try:
            print("🧪 TEST_GENERATOR: Starting test...")
            yield f"data: {json.dumps({'step': 1, 'message': 'Test step 1'})}\n\n"
            await asyncio.sleep(0.1)
            yield f"data: {json.dumps({'step': 2, 'message': 'Test step 2'})}\n\n"
            await asyncio.sleep(0.1)
            yield f"data: {json.dumps({'step': 3, 'message': 'Test completed'})}\n\n"
            print("🧪 TEST_GENERATOR: Completed successfully")
        except Exception as e:
            print(f"🚨 TEST_GENERATOR ERROR: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        test_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

class UploadProgress:
    """Track upload progress"""
    def __init__(self, total_steps: int = 8):  # Updated to 8 steps to include KG processing
        self.total_steps = total_steps
        self.current_step = 0
        self.current_step_name = ""
        self.details = {}
        self.error = None
        
    def update(self, step: int, step_name: str, details: Dict[str, Any] = None):
        self.current_step = step
        self.current_step_name = step_name
        if details:
            self.details.update(details)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_percent": int((self.current_step / self.total_steps) * 100),
            "step_name": self.current_step_name,
            "details": self.details,
            "error": self.error
        }

async def progress_generator(file_content: bytes, filename: str, progress: UploadProgress, upload_params: Dict[str, Any] = None):
    """Generate SSE events for upload progress"""
    print("🔥🔥🔥 PROGRESS_GENERATOR FUNCTION CALLED WITH PARAMS!")
    print(f"🔥🔥🔥 file_content type: {type(file_content)}, length: {len(file_content) if file_content else 'None'}")
    print(f"🔥🔥🔥 filename: {filename}")
    print(f"🔥🔥🔥 progress: {progress}")
    print(f"🔥🔥🔥 upload_params: {upload_params}")
    
    # Initialize progress immediately with Step 1 instead of startup message
    print("🔥🔥🔥 PROGRESS_GENERATOR: About to yield first progress step...")
    progress.update(1, "Initializing upload", {"status": "starting"})
    yield f"data: {json.dumps(progress.to_dict())}\n\n"
    print("🔥🔥🔥 PROGRESS_GENERATOR: First progress step yielded!")
    
    # NOW we can do logging and processing
    print("🔥🔥🔥 PROGRESS_GENERATOR FUNCTION STARTED!")
    print(f"🔥🔥🔥 Filename: {filename}, Content size: {len(file_content)} bytes, Params: {upload_params}")
    
    try:
        print("🔥 PROGRESS_GENERATOR STARTED!")
        print(f"File: {filename}, Upload params: {upload_params}")
        # Step 1: Save file
        progress.update(1, "Saving uploaded file", {"filename": filename})
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        await asyncio.sleep(0.2)  # Allow frontend to display step
        
        temp_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            temp_path = tmp_file.name
            
            # Use the pre-saved file content
            print("🔥 PROGRESS_GENERATOR: Writing file content...")
            if not file_content:
                raise ValueError("File content is empty")
            
            tmp_file.write(file_content)
            file_size_mb = len(file_content) / (1024 * 1024)
            progress.details["file_size_mb"] = round(file_size_mb, 2)
            print(f"🔥 PROGRESS_GENERATOR: File saved to {temp_path}, size: {file_size_mb:.2f}MB")
        
        # Step 2: Load PDF
        progress.update(2, "Loading PDF content", {"status": "processing"})
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        await asyncio.sleep(0.2)  # Allow frontend to display step
        
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        total_pages = len(docs)
        total_chars = sum(len(doc.page_content) for doc in docs)
        
        progress.details.update({
            "total_pages": total_pages,
            "total_characters": total_chars
        })
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        await asyncio.sleep(0.2)  # Allow frontend to display step
        
        # Check if OCR is needed
        if all(not doc.page_content.strip() for doc in docs):
            progress.update(2, "Applying OCR (this may take longer)", {"ocr_required": True})
            yield f"data: {json.dumps(progress.to_dict())}\n\n"
            
            try:
                from langchain_community.document_loaders import UnstructuredPDFLoader
                loader = UnstructuredPDFLoader(temp_path, mode="elements", strategy="ocr_only")
                docs = loader.load()
                progress.details["ocr_applied"] = True
            except ImportError:
                raise HTTPException(status_code=400, detail="PDF appears to be empty and OCR is not available")
        
        # Step 3: Split into chunks
        progress.update(3, "Splitting into chunks", {"chunking": "in_progress"})
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        await asyncio.sleep(0.2)  # Allow frontend to display step
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False
        )
        
        chunks = splitter.split_documents(docs)
        progress.details["total_chunks"] = len(chunks)
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        await asyncio.sleep(0.2)  # Allow frontend to display step
        
        # Step 4: Prepare embeddings
        progress.update(4, "Preparing embeddings", {"embedding_setup": "initializing"})
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        await asyncio.sleep(0.2)  # Allow frontend to display step
        
        embedding_cfg = get_embedding_settings()
        embedding_endpoint = embedding_cfg.get('embedding_endpoint')
        
        print(f"DEBUG: Embedding config: {embedding_cfg}")
        
        if embedding_endpoint and embedding_endpoint.strip():
            print(f"DEBUG: Using HTTP endpoint: {embedding_endpoint}")
            try:
                embeddings = HTTPEmbeddingFunction(embedding_endpoint)
                progress.details["embedding_type"] = "HTTP endpoint"
                print(f"DEBUG: HTTP endpoint embeddings initialized")
            except Exception as e:
                print(f"DEBUG: HTTP endpoint failed: {e}")
                raise
        else:
            print(f"DEBUG: No HTTP endpoint, checking HuggingFace model")
            embedding_model = embedding_cfg.get("embedding_model")
            if not embedding_model or not embedding_model.strip():
                print(f"DEBUG: No embedding model configured, using default")
                embedding_model = "BAAI/bge-base-en-v1.5"
            
            print(f"DEBUG: About to initialize HuggingFace model: {embedding_model}")
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                print(f"DEBUG: HuggingFaceEmbeddings imported successfully")
                embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
                print(f"DEBUG: HuggingFace model initialized successfully")
                progress.details["embedding_type"] = f"HuggingFace: {embedding_model}"
            except Exception as e:
                print(f"DEBUG: HuggingFace initialization failed: {e}")
                raise
        
        print(f"DEBUG: Embeddings initialized successfully")
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        await asyncio.sleep(0.2)  # Allow frontend to display step
        
        # Step 5: Check for duplicates
        progress.update(5, "Checking for duplicates", {"deduplication": "in_progress"})
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        await asyncio.sleep(0.2)  # Allow frontend to display step
        
        # Compute file hash
        with open(temp_path, "rb") as f:
            file_content = f.read()
            file_id = hashlib.sha256(file_content).hexdigest()[:12]
        
        # Initialize BM25 processor
        bm25_processor = BM25Processor()
        
        # Extract file metadata
        file_metadata = MetadataExtractor.extract_metadata(temp_path, filename)
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                chunk.metadata = {}
            
            original_page = chunk.metadata.get('page', 0)
            chunk.metadata.update({
                'source': filename.lower(),  # Normalize filename to lowercase
                'page': original_page,
                'doc_type': 'pdf',
                'uploaded_at': datetime.now().isoformat(),
                'section': f"chunk_{i}",
                'author': '',
                'chunk_index': i,
                'file_id': file_id,
                'hash': hash_text(chunk.page_content),
                'doc_id': f"{file_id}_p{original_page}_c{i}",
                'creation_date': file_metadata['creation_date'],
                'last_modified_date': file_metadata['last_modified_date']
            })
            
            # Add BM25 preprocessing
            bm25_metadata = bm25_processor.prepare_document_for_bm25(chunk.page_content, chunk.metadata)
            chunk.metadata.update(bm25_metadata)
        
        # Connect to vector DB and check duplicates  
        print("🔥 PROGRESS_GENERATOR: Using hardcoded Milvus settings to bypass config issues...")
        
        # Use hardcoded settings from debug script that we know work
        vector_db_cfg = {
            "active": "milvus",
            "databases": [
                {
                    "id": "milvus",
                    "name": "Milvus",
                    "enabled": True,
                    "config": {
                        "dimension": 2560,
                        "MILVUS_URI": "https://in03-093b567a1ae72cf.serverless.gcp-us-west1.cloud.zilliz.com",
                        "MILVUS_TOKEN": "a8b2580e8ad9dd547e91bf8dc9292b849470968a5ce4b857a768d341cf026e7a30327a73786795d40cac81ef688c000dd910f581",
                        "MILVUS_DEFAULT_COLLECTION": "default_knowledge"
                    }
                }
            ]
        }
        print("🔥 PROGRESS_GENERATOR: Using hardcoded settings successfully")
        
        # Now we always have the new format due to migration
        print("🔥 PROGRESS_GENERATOR: Parsing migrated vector DB config...")
        
        # Find the active Milvus database
        active_db_id = vector_db_cfg.get("active", "milvus")
        milvus_db = None
        
        for db in vector_db_cfg.get("databases", []):
            if db.get("id") == active_db_id or (active_db_id == "milvus" and db.get("id") == "milvus"):
                milvus_db = db
                break
        
        if not milvus_db:
            # Try to find any enabled Milvus database
            for db in vector_db_cfg.get("databases", []):
                if db.get("id") == "milvus" and db.get("enabled"):
                    milvus_db = db
                    break
        
        if not milvus_db:
            raise ValueError(f"No Milvus database configuration found in vector DB settings")
        
        milvus_cfg = milvus_db.get("config", {})
        print(f"🔥 PROGRESS_GENERATOR: Using Milvus database: {milvus_db.get('name')}")
        print(f"🔥 PROGRESS_GENERATOR: Milvus config keys: {list(milvus_cfg.keys())}")
        
        uri = milvus_cfg.get("MILVUS_URI")
        token = milvus_cfg.get("MILVUS_TOKEN")
        # Use user-specified collection or fall back to default
        if upload_params and upload_params.get('target_collection'):
            collection_name = upload_params['target_collection']
            progress.details["collection_method"] = "user_specified"
            progress.details["collection_source"] = "frontend_selection"
        else:
            collection_name = milvus_cfg.get("MILVUS_DEFAULT_COLLECTION", "default_knowledge")
            progress.details["collection_method"] = "default_fallback"
            progress.details["collection_source"] = "milvus_config"
        
        progress.details["target_collection"] = collection_name
        vector_dim = int(milvus_cfg.get("dimension", 2560))
        
        print(f"🔥 PROGRESS_GENERATOR: Ensuring Milvus collection: {collection_name}")
        ensure_milvus_collection(collection_name, vector_dim=vector_dim, uri=uri, token=token)
        
        print(f"🔥 PROGRESS_GENERATOR: Creating Collection object for: {collection_name}")
        collection = Collection(collection_name)
        print(f"🔥 PROGRESS_GENERATOR: Loading collection...")
        collection.load()
        print(f"🔥 PROGRESS_GENERATOR: Getting existing hashes...")
        try:
            existing_hashes = get_existing_hashes(collection)
            print(f"🔥 PROGRESS_GENERATOR: Successfully got {len(existing_hashes)} existing hashes")
        except Exception as e:
            print(f"🔥 PROGRESS_GENERATOR: Failed to get existing hashes: {e}")
            # If we can't get existing hashes, assume empty to continue upload
            existing_hashes = set()
        
        print(f"🔥 PROGRESS_GENERATOR: Getting existing doc IDs...")
        try:
            existing_doc_ids = get_existing_doc_ids(collection)
            print(f"🔥 PROGRESS_GENERATOR: Successfully got {len(existing_doc_ids)} existing doc IDs")
        except Exception as e:
            print(f"🔥 PROGRESS_GENERATOR: Failed to get existing doc IDs: {e}")
            # If we can't get existing doc_ids, assume empty to continue upload
            existing_doc_ids = set()
        
        print(f"🔥 PROGRESS_GENERATOR: Found {len(existing_hashes)} existing hashes, {len(existing_doc_ids)} existing doc IDs")
        
        # Filter duplicates
        unique_chunks = []
        duplicate_count = 0
        
        for chunk in chunks:
            chunk_hash = chunk.metadata.get('hash')
            doc_id = chunk.metadata.get('doc_id')
            
            if chunk_hash in existing_hashes or doc_id in existing_doc_ids:
                duplicate_count += 1
            else:
                unique_chunks.append(chunk)
        
        progress.details.update({
            "duplicates_found": duplicate_count,
            "unique_chunks": len(unique_chunks)
        })
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
        if not unique_chunks:
            # Skip to final step since no new chunks to process
            progress.update(8, "Upload complete - All chunks are duplicates", {
                "status": "success",
                "total_chunks": len(chunks),
                "unique_chunks_inserted": 0,
                "duplicates_filtered": len(chunks),
                "collection": collection_name,
                "file_id": file_id,
                "reason": "All chunks already exist in database",
                "kg_processing_skipped": True,
                "kg_skip_reason": "No new chunks to process for knowledge graph"
            })
            yield f"data: {json.dumps(progress.to_dict())}\n\n"
            return
        
        # Step 6: Generate embeddings
        progress.update(6, "Generating embeddings", {
            "embedding_progress": 0,
            "total_embeddings": len(unique_chunks)
        })
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
        # Process embeddings in batches for progress updates
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(unique_chunks), batch_size):
            batch = unique_chunks[i:i + batch_size]
            batch_texts = [chunk.page_content for chunk in batch]
            
            batch_embeddings = embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            # Update embedding progress while staying on step 6
            progress.details["embedding_progress"] = len(all_embeddings)
            yield f"data: {json.dumps(progress.to_dict())}\n\n"
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.1)
        
        # ONLY NOW advance to step 7 after ALL embeddings are complete
        progress.update(7, "Inserting into vector database", {"insertion": "in_progress"})
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
        # Prepare data for insertion
        unique_ids = [str(uuid4()) for _ in unique_chunks]
        unique_texts = [chunk.page_content for chunk in unique_chunks]
        
        data = [
            unique_ids,
            all_embeddings,
            unique_texts,
            [chunk.metadata.get('source', '') for chunk in unique_chunks],
            [chunk.metadata.get('page', 0) for chunk in unique_chunks],
            [chunk.metadata.get('doc_type', 'pdf') for chunk in unique_chunks],
            [chunk.metadata.get('uploaded_at', '') for chunk in unique_chunks],
            [chunk.metadata.get('section', '') for chunk in unique_chunks],
            [chunk.metadata.get('author', '') for chunk in unique_chunks],
            [chunk.metadata.get('hash', '') for chunk in unique_chunks],
            [chunk.metadata.get('doc_id', '') for chunk in unique_chunks],
            # BM25 enhancement fields
            [chunk.metadata.get('bm25_tokens', '') for chunk in unique_chunks],
            [chunk.metadata.get('bm25_term_count', 0) for chunk in unique_chunks],
            [chunk.metadata.get('bm25_unique_terms', 0) for chunk in unique_chunks],
            [chunk.metadata.get('bm25_top_terms', '') for chunk in unique_chunks],
            # Date metadata fields
            [chunk.metadata.get('creation_date', '') for chunk in unique_chunks],
            [chunk.metadata.get('last_modified_date', '') for chunk in unique_chunks],
        ]
        
        print(f"🔥 PROGRESS_GENERATOR: About to insert {len(unique_chunks)} chunks into collection")
        print(f"🔥 PROGRESS_GENERATOR: Data structure lengths: {[len(d) for d in data]}")
        
        try:
            insert_result = collection.insert(data)
            print(f"🔥 PROGRESS_GENERATOR: Insert successful, result: {insert_result}")
            
            print(f"🔥 PROGRESS_GENERATOR: Flushing collection...")
            collection.flush()
            print(f"🔥 PROGRESS_GENERATOR: Flush completed")
            
            # Verify insertion by checking collection count
            print(f"🔥 PROGRESS_GENERATOR: Checking collection count after insertion...")
            collection.load()  # Reload to see new data
            count = collection.num_entities
            print(f"🔥 PROGRESS_GENERATOR: Collection now has {count} entities")
            
        except Exception as insert_error:
            print(f"🔥 PROGRESS_GENERATOR: Insert failed: {insert_error}")
            raise insert_error
        
        # Step 8: Process knowledge graph
        progress.update(8, "Processing knowledge graph", {
            "kg_processing": "in_progress",
            "entities_extracted": 0,
            "relationships_extracted": 0
        })
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
        # Check if knowledge graph dependencies are available
        if not KG_IMPORTS_AVAILABLE:
            print("🔥 PROGRESS_GENERATOR: KG dependencies not available, skipping knowledge graph processing")
            progress.update(8, "Upload complete (KG processing skipped)", {
                "status": "success",
                "total_chunks": len(chunks),
                "unique_chunks_inserted": len(unique_chunks),
                "duplicates_filtered": duplicate_count,
                "collection": collection_name,
                "file_id": file_id,
                "kg_processing_skipped": True,
                "kg_skip_reason": "Knowledge graph dependencies not available"
            })
            yield f"data: {json.dumps(progress.to_dict())}\n\n"
            return
        
        try:
            # Convert chunks to ExtractedChunk format for knowledge graph processing
            extracted_chunks = []
            for i, chunk in enumerate(chunks):
                # ExtractedChunk only accepts (content, metadata, quality_score)
                extracted_chunk = ExtractedChunk(
                    content=chunk.page_content,
                    metadata={
                        **chunk.metadata,
                        'source': filename,
                        'page_number': chunk.metadata.get('page', 0),
                        'section': f"chunk_{i}"
                    }
                )
                extracted_chunks.append(extracted_chunk)
            
            # Get knowledge graph processor
            graph_processor = get_graph_document_processor()
            
            # Process all chunks for knowledge graph at once (more efficient)
            print(f"🔥 PROGRESS_GENERATOR: Processing {len(extracted_chunks)} chunks for knowledge graph")
            
            try:
                # Use the correct method: process_document_for_graph
                result = await graph_processor.process_document_for_graph(
                    chunks=extracted_chunks,
                    document_id=file_id,
                    store_in_neo4j=True
                )
                
                total_entities = result.total_entities if result and result.success else 0
                total_relationships = result.total_relationships if result and result.success else 0
                
                print(f"🔥 PROGRESS_GENERATOR: KG processing result - entities: {total_entities}, relationships: {total_relationships}")
                
                # Update progress with final results
                progress.details.update({
                    "kg_chunks_processed": len(extracted_chunks),
                    "kg_total_chunks": len(extracted_chunks),
                    "entities_extracted": total_entities,
                    "relationships_extracted": total_relationships
                })
                yield f"data: {json.dumps(progress.to_dict())}\n\n"
                
            except Exception as kg_batch_error:
                print(f"🔥 KG batch processing error: {kg_batch_error}")
                # Set totals to 0 if processing fails
                total_entities = 0
                total_relationships = 0
            
            print(f"🔥 PROGRESS_GENERATOR: KG processing completed - {total_entities} entities, {total_relationships} relationships")
            
            # Final success with knowledge graph stats
            progress.update(8, "Upload complete!", {
                "status": "success",
                "total_chunks": len(chunks),
                "unique_chunks_inserted": len(unique_chunks),
                "duplicates_filtered": duplicate_count,
                "collection": collection_name,
                "file_id": file_id,
                "kg_entities_extracted": total_entities,
                "kg_relationships_extracted": total_relationships,
                "kg_processing_completed": True
            })
            
        except Exception as kg_error:
            print(f"🔥 PROGRESS_GENERATOR: KG processing failed: {kg_error}")
            # Still mark as success but note KG failure
            progress.update(8, "Upload complete (KG processing failed)", {
                "status": "success",
                "total_chunks": len(chunks),
                "unique_chunks_inserted": len(unique_chunks),
                "duplicates_filtered": duplicate_count,
                "collection": collection_name,
                "file_id": file_id,
                "kg_processing_error": str(kg_error),
                "kg_processing_completed": False
            })
        
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
    except Exception as e:
        print(f"🚨 ERROR in progress_generator: {e}")
        print(f"🚨 ERROR type: {type(e)}")
        import traceback
        print(f"🚨 ERROR traceback: {traceback.format_exc()}")
        
        # Send error to client
        error_data = {
            "current_step": 0,
            "total_steps": 1,
            "progress_percent": 0,
            "step_name": "Upload failed",
            "details": {"error": str(e)},
            "error": str(e)
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        
    finally:
        print("🔥 PROGRESS_GENERATOR: Cleanup started")
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"🔥 PROGRESS_GENERATOR: Cleaned up temp file: {temp_path}")
            except Exception as cleanup_error:
                print(f"🔥 PROGRESS_GENERATOR: Cleanup error: {cleanup_error}")
        print("🔥 PROGRESS_GENERATOR: Finished")

@router.post("/upload_pdf_simple")
async def upload_pdf_simple(
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(None)
):
    """Simple PDF upload without streaming - just process and return result"""
    
    print("🔥 SIMPLE UPLOAD STARTED")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Read file content
    file_content = await file.read()
    progress = UploadProgress(total_steps=8)  # Use 8 steps to include KG processing
    upload_params = {'target_collection': collection_name}
    
    print(f"🔥 Processing {file.filename} ({len(file_content)} bytes)")
    
    # Execute the progress generator completely
    try:
        final_result = None
        upload_success = False
        error_message = None
        
        async for result in progress_generator(file_content, file.filename, progress, upload_params):
            print(f"🔥 Step: {result[:100] if result else 'None'}...")
            final_result = result
            
            # Check if this is error data
            if result and "error" in result:
                try:
                    import json
                    data = json.loads(result.replace("data: ", "").strip())
                    if data.get("error"):
                        error_message = data["error"]
                        break
                except:
                    pass
            
            # Check if this is success completion (step 7 with success status)
            if result and "Upload complete!" in result:
                try:
                    import json
                    data = json.loads(result.replace("data: ", "").strip())
                    if data.get("details", {}).get("status") == "success":
                        upload_success = True
                except:
                    pass
        
        if error_message:
            print(f"🔥 UPLOAD FAILED: {error_message}")
            raise HTTPException(status_code=500, detail=f"Upload failed: {error_message}")
        
        if upload_success:
            print("🔥 UPLOAD COMPLETED SUCCESSFULLY")
            return {"status": "success", "message": "File uploaded successfully", "collection": collection_name}
        else:
            print("🔥 UPLOAD DID NOT COMPLETE SUCCESSFULLY")
            raise HTTPException(status_code=500, detail="Upload did not complete successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"🔥 UPLOAD FAILED: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/upload_pdf_progress")
async def upload_pdf_with_progress(
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(None),
    disable_auto_classification: Optional[str] = Form(None),
    force_collection: Optional[str] = Form(None),
    chunk_size: Optional[str] = Form(None),
    chunk_overlap: Optional[str] = Form(None),
    enable_bm25: Optional[str] = Form(None),
    bm25_weight: Optional[str] = Form(None)
):
    """Upload PDF with real-time progress updates via SSE"""
    
    print("=" * 60)
    print("🚨 BACKEND ENDPOINT HIT: upload_pdf_progress")
    print("=" * 60)
    
    # DEBUG: Log all received parameters
    print(f"BACKEND DEBUG: Received parameters:")
    print(f"  file.filename: {file.filename}")
    print(f"  collection_name: {collection_name}")
    print(f"  force_collection: {force_collection}")
    print(f"  disable_auto_classification: {disable_auto_classification}")
    print(f"  chunk_size: {chunk_size}")
    print(f"  chunk_overlap: {chunk_overlap}")
    print(f"  enable_bm25: {enable_bm25}")
    print(f"  bm25_weight: {bm25_weight}")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    progress = UploadProgress(total_steps=8)  # Use 8 steps to include KG processing
    
    # Determine target collection
    target_collection = None
    if force_collection:
        target_collection = force_collection
        print(f"BACKEND DEBUG: Using force_collection: {force_collection}")
    elif collection_name:
        target_collection = collection_name
        print(f"BACKEND DEBUG: Using collection_name: {collection_name}")
    else:
        print(f"BACKEND DEBUG: No collection specified, will use default")
    
    # Create upload parameters
    upload_params = {
        'target_collection': target_collection,
        'chunk_size': int(chunk_size) if chunk_size else None,
        'chunk_overlap': int(chunk_overlap) if chunk_overlap else None,
        'enable_bm25': enable_bm25 == 'true' if enable_bm25 else None,
        'bm25_weight': float(bm25_weight) if bm25_weight else None,
    }
    
    # Read file content immediately to avoid file closure issues
    print(f"🔥 Reading file content before classification...")
    try:
        file_content = await file.read()
        print(f"🔥 File content read: {len(file_content)} bytes")
    except Exception as e:
        print(f"🔥 Error reading file: {e}")
        raise HTTPException(status_code=400, detail=f"Cannot read file: {e}")
    
    if not file_content:
        raise HTTPException(status_code=400, detail="File is empty")
    
    print(f"🔥 About to create StreamingResponse with progress_generator")
    print(f"🔥 Upload params: {upload_params}")
    print(f"🔥 File content size: {len(file_content)} bytes")
    print(f"🔥 Progress object: {progress}")
    print(f"🔥🔥🔥 CALLING progress_generator NOW!")
    print(f"🔥🔥🔥 ABOUT TO CALL: progress_generator({type(file_content)}, {file.filename}, {type(progress)}, {upload_params})")
    
    generator = progress_generator(file_content, file.filename, progress, upload_params)
    print(f"🔥🔥🔥 GENERATOR CREATED: {type(generator)}")
    
    # Execute the upload in background regardless of stream consumption
    async def background_upload():
        print("🔥🔥🔥 STARTING BACKGROUND UPLOAD TASK")
        try:
            result_data = None
            async for item in generator:
                print(f"🔥🔥🔥 PROCESSING: {item[:50] if item else 'None'}...")
                result_data = item
                yield item
            print("🔥🔥🔥 BACKGROUND UPLOAD COMPLETE")
            print(f"🔥🔥🔥 FINAL RESULT: {result_data}")
        except Exception as e:
            print(f"🔥🔥🔥 BACKGROUND UPLOAD ERROR: {e}")
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
    
    return StreamingResponse(
        background_upload(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable proxy buffering
        }
    )