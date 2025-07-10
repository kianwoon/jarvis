"""
Temporary document manager for lifecycle management and user controls.
Handles document upload, processing, user preferences, and cleanup coordination.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import os
from pathlib import Path

from app.core.redis_base import RedisCache
from app.services.temp_document_indexer import TempDocumentIndexer
from app.document_handlers.base import DocumentHandler
from app.document_handlers.base import ExtractedChunk
from app.utils.metadata_extractor import MetadataExtractor

logger = logging.getLogger(__name__)

class TempDocumentManager:
    """
    Manages the complete lifecycle of temporary documents including:
    - File upload and validation
    - Document processing and indexing
    - User preference management
    - Cleanup coordination
    """
    
    def __init__(self):
        self.redis_cache = RedisCache(key_prefix="temp_mgr_")
        self.indexer = TempDocumentIndexer()
        self.metadata_extractor = MetadataExtractor()
        
        # Supported file types
        self.supported_extensions = {'.pdf', '.docx', '.xlsx', '.pptx', '.txt'}
        self.max_file_size_mb = 50
    
    async def upload_and_process_document(
        self,
        file_content: bytes,
        filename: str,
        conversation_id: str,
        ttl_hours: int = 2,
        auto_include: bool = True
    ) -> Dict[str, Any]:
        """
        Upload and process a document for temporary indexing.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            conversation_id: Associated conversation ID
            ttl_hours: Time-to-live in hours
            auto_include: Whether to automatically include in chat context
            
        Returns:
            Processing result with temp_doc_id and metadata
        """
        try:
            # Generate unique temp document ID
            temp_doc_id = f"temp_doc_{uuid.uuid4().hex[:12]}"
            
            logger.info(f"Processing temporary document: {filename} (ID: {temp_doc_id})")
            
            # Validate file
            validation_result = await self._validate_file(file_content, filename)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': validation_result['error'],
                    'temp_doc_id': temp_doc_id
                }
            
            # Set processing status
            await self._set_document_status(temp_doc_id, 'processing', {
                'filename': filename,
                'conversation_id': conversation_id,
                'upload_timestamp': datetime.now().isoformat()
            })
            
            # Save temporary file
            temp_file_path = await self._save_temp_file(file_content, filename, temp_doc_id)
            
            try:
                # Extract document content using existing handlers
                document_chunks = await self._extract_document_content(temp_file_path, filename)
                
                if not document_chunks:
                    await self._set_document_status(temp_doc_id, 'error', {'error': 'No content extracted'})
                    return {
                        'success': False,
                        'error': 'No extractable content found in document',
                        'temp_doc_id': temp_doc_id
                    }
                
                # Create temporary index
                index_result = await self.indexer.create_temp_index(
                    document_chunks=document_chunks,
                    temp_doc_id=temp_doc_id,
                    conversation_id=conversation_id,
                    filename=filename,
                    ttl_hours=ttl_hours
                )
                
                # Update document metadata with indexing results
                index_result['is_included'] = auto_include
                index_result['status'] = 'ready'
                
                logger.info(f"Successfully processed temporary document: {filename}")
                
                return {
                    'success': True,
                    'temp_doc_id': temp_doc_id,
                    'metadata': index_result
                }
                
            finally:
                # Cleanup temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            logger.error(f"Failed to process temporary document {filename}: {str(e)}")
            await self._set_document_status(temp_doc_id, 'error', {'error': str(e)})
            return {
                'success': False,
                'error': f'Processing failed: {str(e)}',
                'temp_doc_id': temp_doc_id
            }
    
    async def get_conversation_documents(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get all temporary documents for a conversation with their current status.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            List of document metadata with status information
        """
        try:
            return await self.indexer.get_conversation_temp_docs(conversation_id)
        except Exception as e:
            logger.error(f"Failed to get documents for conversation {conversation_id}: {str(e)}")
            return []
    
    async def update_document_preferences(
        self,
        temp_doc_id: str,
        preferences: Dict[str, Any]
    ) -> bool:
        """
        Update user preferences for a temporary document.
        
        Args:
            temp_doc_id: Temporary document ID
            preferences: User preferences (is_included, ttl_hours, etc.)
            
        Returns:
            Success status
        """
        try:
            # Update inclusion status
            if 'is_included' in preferences:
                success = await self.indexer.update_document_inclusion(
                    temp_doc_id, preferences['is_included']
                )
                if not success:
                    return False
            
            # Update TTL if requested
            if 'ttl_hours' in preferences:
                await self._update_document_ttl(temp_doc_id, preferences['ttl_hours'])
            
            logger.info(f"Updated preferences for document {temp_doc_id}: {preferences}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update preferences for {temp_doc_id}: {str(e)}")
            return False
    
    async def delete_document(self, temp_doc_id: str) -> bool:
        """
        Manually delete a temporary document.
        
        Args:
            temp_doc_id: Temporary document ID
            
        Returns:
            Success status
        """
        try:
            return await self.indexer.delete_temp_document(temp_doc_id)
        except Exception as e:
            logger.error(f"Failed to delete document {temp_doc_id}: {str(e)}")
            return False
    
    async def get_conversation_preferences(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get user preferences for temporary documents in a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            User preferences dictionary
        """
        try:
            preferences = self.redis_cache.get(f"conv_prefs:{conversation_id}")
            if preferences:
                return preferences
            
            # Return defaults
            return {
                'auto_include_new_docs': True,
                'default_ttl_hours': 2,
                'include_temp_docs_in_chat': True,
                'max_documents_per_conversation': 5
            }
        except Exception as e:
            logger.error(f"Failed to get preferences for conversation {conversation_id}: {str(e)}")
            return {}
    
    async def update_conversation_preferences(
        self,
        conversation_id: str,
        preferences: Dict[str, Any]
    ) -> bool:
        """
        Update user preferences for a conversation.
        
        Args:
            conversation_id: Conversation ID
            preferences: User preferences
            
        Returns:
            Success status
        """
        try:
            # Store with 7-day TTL (longer than document TTL)
            self.redis_cache.set(
                f"conv_prefs:{conversation_id}",
                preferences,
                expire=7 * 24 * 3600  # 7 days
            )
            
            logger.info(f"Updated conversation preferences for {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update conversation preferences: {str(e)}")
            return False
    
    async def get_active_documents_for_chat(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get documents that should be included in chat context for a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            List of active document metadata
        """
        try:
            all_docs = await self.get_conversation_documents(conversation_id)
            active_docs = [doc for doc in all_docs if doc.get('is_included', False)]
            return active_docs
        except Exception as e:
            logger.error(f"Failed to get active documents for {conversation_id}: {str(e)}")
            return []
    
    async def query_conversation_documents(
        self,
        conversation_id: str,
        query: str,
        include_all: bool = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query all active temporary documents in a conversation.
        
        Args:
            conversation_id: Conversation ID
            query: Query string
            include_all: Override to include all docs or only active ones
            top_k: Number of results per document
            
        Returns:
            Combined query results from all active documents
        """
        try:
            # Get documents to query
            if include_all:
                docs_to_query = await self.get_conversation_documents(conversation_id)
            else:
                docs_to_query = await self.get_active_documents_for_chat(conversation_id)
            
            if not docs_to_query:
                return []
            
            # Query each document
            all_results = []
            for doc in docs_to_query:
                temp_doc_id = doc['temp_doc_id']
                doc_results = await self.indexer.query_temp_document(temp_doc_id, query, top_k)
                
                if 'sources' in doc_results:
                    # Add document context to each result
                    for source in doc_results['sources']:
                        source['temp_doc_id'] = temp_doc_id
                        source['filename'] = doc['filename']
                        all_results.append(source)
            
            # Sort by relevance score
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            return all_results[:top_k * 2]  # Return top results across all documents
            
        except Exception as e:
            logger.error(f"Failed to query conversation documents: {str(e)}")
            return []
    
    async def cleanup_conversation_documents(self, conversation_id: str) -> int:
        """
        Clean up all temporary documents for a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Number of documents cleaned up
        """
        try:
            docs = await self.get_conversation_documents(conversation_id)
            cleaned_count = 0
            
            for doc in docs:
                if await self.delete_document(doc['temp_doc_id']):
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} documents for conversation {conversation_id}")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup conversation documents: {str(e)}")
            return 0
    
    async def _validate_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Validate uploaded file."""
        try:
            # Check file extension
            file_ext = Path(filename).suffix.lower()
            if file_ext not in self.supported_extensions:
                return {
                    'valid': False,
                    'error': f'Unsupported file type: {file_ext}. Supported: {", ".join(self.supported_extensions)}'
                }
            
            # Check file size
            file_size_mb = len(file_content) / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                return {
                    'valid': False,
                    'error': f'File too large: {file_size_mb:.1f}MB (max: {self.max_file_size_mb}MB)'
                }
            
            # Check if file is empty
            if len(file_content) == 0:
                return {
                    'valid': False,
                    'error': 'File is empty'
                }
            
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation failed: {str(e)}'}
    
    async def _save_temp_file(self, file_content: bytes, filename: str, temp_doc_id: str) -> str:
        """Save file content to temporary location."""
        temp_dir = Path("/tmp/temp_documents")
        temp_dir.mkdir(exist_ok=True)
        
        # Create safe filename
        safe_filename = f"{temp_doc_id}_{Path(filename).name}"
        temp_file_path = temp_dir / safe_filename
        
        with open(temp_file_path, 'wb') as f:
            f.write(file_content)
        
        return str(temp_file_path)
    
    async def _extract_document_content(self, file_path: str, filename: str) -> List[ExtractedChunk]:
        """Extract content using LlamaIndex document loaders."""
        try:
            file_ext = Path(filename).suffix.lower()
            
            if file_ext == '.pdf':
                # Try PyMuPDF first as it handles complex PDFs better
                try:
                    import pymupdf  # PyMuPDF - handles compressed/encrypted PDFs better
                    logger.info(f"[PDF DEBUG] Using PyMuPDF for {filename}")
                    
                    chunks = []
                    doc = pymupdf.open(file_path)
                    total_pages = len(doc)
                    logger.info(f"[PDF DEBUG] PyMuPDF found {total_pages} pages")
                    
                    for page_num, page in enumerate(doc):
                        try:
                            # Get text with better layout preservation
                            text = page.get_text("text")
                            
                            if text and text.strip():
                                # Clean up the text
                                text = text.strip()
                                
                                # Check if page contains mostly readable content
                                printable_chars = sum(1 for c in text if c.isprintable() or c.isspace())
                                total_chars = len(text)
                                
                                if total_chars > 0 and printable_chars / total_chars >= 0.8:
                                    chunk = ExtractedChunk(
                                        content=text,
                                        metadata={
                                            'source': filename,
                                            'page': page_num + 1,
                                            'total_pages': total_pages,
                                            'chunk_index': page_num,
                                            'extraction_method': 'PyMuPDF'
                                        },
                                        quality_score=0.95
                                    )
                                    chunks.append(chunk)
                                    logger.info(f"[PDF DEBUG] Page {page_num}: extracted {len(text)} characters")
                                else:
                                    logger.warning(f"[PDF DEBUG] Page {page_num} has low readable content ratio: {printable_chars/total_chars:.2%}")
                                    
                        except Exception as page_error:
                            logger.warning(f"[PDF DEBUG] Failed to extract page {page_num}: {str(page_error)}")
                    
                    doc.close()
                    
                    if chunks:
                        logger.info(f"[PDF DEBUG] PyMuPDF successfully extracted {len(chunks)} pages")
                        return chunks
                    else:
                        logger.warning(f"[PDF DEBUG] PyMuPDF extracted no readable content")
                        
                except ImportError:
                    logger.warning("PyMuPDF not available, trying LlamaIndex PDFReader")
                except Exception as pymupdf_error:
                    logger.warning(f"PyMuPDF failed: {str(pymupdf_error)}, trying LlamaIndex PDFReader")
                
                # Try LlamaIndex's built-in PDF reader as second option
                try:
                    from llama_index.readers.file import PDFReader
                    
                    logger.info(f"[PDF DEBUG] Using LlamaIndex PDFReader for {filename}")
                    reader = PDFReader()
                    documents = reader.load_data(file=Path(file_path))
                    
                    logger.info(f"[PDF DEBUG] Extracted {len(documents)} document chunks")
                    
                    chunks = []
                    for i, doc in enumerate(documents):
                        logger.info(f"[PDF DEBUG] Chunk {i}: text length {len(doc.text)}")
                        chunk = ExtractedChunk(
                            content=doc.text,
                            metadata={
                                **doc.metadata,
                                'source': filename,
                                'chunk_index': i,
                                'extraction_method': 'LlamaIndex_PDFReader'
                            },
                            quality_score=0.9
                        )
                        chunks.append(chunk)
                    
                    return chunks
                    
                except ImportError:
                    logger.warning("LlamaIndex PDFReader not available, trying SimpleDirectoryReader")
                    
                except Exception as pdf_error:
                    logger.warning(f"PDFReader failed: {str(pdf_error)}, trying SimpleDirectoryReader")
                
                # Fallback to SimpleDirectoryReader
                try:
                    from llama_index.core import SimpleDirectoryReader
                    
                    logger.info(f"[PDF DEBUG] Using LlamaIndex SimpleDirectoryReader for {filename}")
                    # Create a temporary directory with just this file
                    temp_dir = Path(file_path).parent
                    reader = SimpleDirectoryReader(input_files=[file_path])
                    documents = reader.load_data()
                    
                    logger.info(f"[PDF DEBUG] SimpleDirectoryReader extracted {len(documents)} document chunks")
                    
                    chunks = []
                    for i, doc in enumerate(documents):
                        logger.info(f"[PDF DEBUG] Chunk {i}: text length {len(doc.text)}")
                        chunk = ExtractedChunk(
                            content=doc.text,
                            metadata={
                                **doc.metadata,
                                'source': filename,
                                'chunk_index': i,
                                'extraction_method': 'LlamaIndex_SimpleDirectoryReader'
                            },
                            quality_score=0.8
                        )
                        chunks.append(chunk)
                    
                    return chunks
                    
                except Exception as e:
                    logger.error(f"LlamaIndex PDF extraction failed: {str(e)}")
                    
                    # Try pdfplumber as next fallback
                    try:
                        import pdfplumber
                        logger.info(f"[PDF DEBUG] Trying pdfplumber fallback for {filename}")
                        
                        chunks = []
                        with pdfplumber.open(file_path) as pdf:
                            total_pages = len(pdf.pages)
                            logger.info(f"[PDF DEBUG] pdfplumber found {total_pages} pages")
                            
                            for page_num, page in enumerate(pdf.pages):
                                try:
                                    text = page.extract_text()
                                    if text and text.strip():
                                        # Clean up the text
                                        text = text.strip()
                                        # Check readable content ratio
                                        printable_chars = sum(1 for c in text if c.isprintable() or c.isspace())
                                        total_chars = len(text)
                                        
                                        if total_chars > 0 and printable_chars / total_chars >= 0.8:
                                            chunk = ExtractedChunk(
                                                content=text,
                                                metadata={
                                                    'source': filename,
                                                    'page': page_num + 1,
                                                    'total_pages': total_pages,
                                                    'chunk_index': page_num,
                                                    'extraction_method': 'pdfplumber'
                                                },
                                                quality_score=0.85
                                            )
                                            chunks.append(chunk)
                                            logger.info(f"[PDF DEBUG] Page {page_num}: extracted {len(text)} characters")
                                        else:
                                            logger.warning(f"[PDF DEBUG] Page {page_num} has low readable content ratio")
                                except Exception as page_error:
                                    logger.warning(f"[PDF DEBUG] Failed to extract page {page_num}: {str(page_error)}")
                        
                        if chunks:
                            logger.info(f"[PDF DEBUG] pdfplumber successfully extracted {len(chunks)} pages")
                            return chunks
                        else:
                            logger.warning(f"[PDF DEBUG] pdfplumber extracted no readable content")
                            
                    except ImportError:
                        logger.warning("pdfplumber not installed, trying pypdf")
                    except Exception as pdfplumber_error:
                        logger.warning(f"pdfplumber extraction failed: {str(pdfplumber_error)}, trying pypdf")
                    
                    # Try pypdf as final fallback
                    try:
                        import pypdf
                        logger.info(f"[PDF DEBUG] Trying pypdf fallback for {filename}")
                        
                        chunks = []
                        with open(file_path, 'rb') as pdf_file:
                            pdf_reader = pypdf.PdfReader(pdf_file)
                            total_pages = len(pdf_reader.pages)
                            logger.info(f"[PDF DEBUG] pypdf found {total_pages} pages")
                            
                            for page_num, page in enumerate(pdf_reader.pages):
                                try:
                                    text = page.extract_text()
                                    if text and text.strip():
                                        # Clean up the text
                                        text = text.strip()
                                        # Skip pages that appear to be mostly binary data
                                        if len([c for c in text if c.isprintable()]) / len(text) < 0.8:
                                            logger.warning(f"[PDF DEBUG] Page {page_num} appears to contain binary data, skipping")
                                            continue
                                            
                                        chunk = ExtractedChunk(
                                            content=text,
                                            metadata={
                                                'source': filename,
                                                'page': page_num + 1,
                                                'total_pages': total_pages,
                                                'chunk_index': page_num,
                                                'extraction_method': 'pypdf'
                                            },
                                            quality_score=0.7
                                        )
                                        chunks.append(chunk)
                                        logger.info(f"[PDF DEBUG] Page {page_num}: extracted {len(text)} characters")
                                except Exception as page_error:
                                    logger.warning(f"[PDF DEBUG] Failed to extract page {page_num}: {str(page_error)}")
                        
                        if chunks:
                            logger.info(f"[PDF DEBUG] pypdf successfully extracted {len(chunks)} pages")
                            return chunks
                        else:
                            logger.error(f"[PDF DEBUG] pypdf extracted no readable content")
                            
                    except ImportError:
                        logger.error("pypdf not installed, cannot use fallback PDF extraction")
                    except Exception as pypdf_error:
                        logger.error(f"pypdf extraction failed: {str(pypdf_error)}")
                    
                    # If all methods fail, return error chunk
                    return [ExtractedChunk(
                        content=f"Failed to extract PDF content. The PDF may be corrupted, encrypted, or contain only images without text.",
                        metadata={'source': filename, 'extraction_method': 'error'},
                        quality_score=0.1
                    )]
            
            elif file_ext in ['.docx', '.doc']:
                from app.document_handlers.word_handler import WordHandler
                handler = WordHandler()
                return handler.extract(file_path)
            
            elif file_ext in ['.xlsx', '.xls']:
                from app.document_handlers.excel_handler import ExcelHandler
                handler = ExcelHandler()
                return handler.extract(file_path)
            
            elif file_ext in ['.pptx', '.ppt']:
                from app.document_handlers.powerpoint_handler import PowerPointHandler
                handler = PowerPointHandler()
                return handler.extract(file_path)
            
            elif file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    return [ExtractedChunk(
                        content=content,
                        metadata={'source': filename},
                        quality_score=0.9
                    )]
            
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
                
        except Exception as e:
            logger.error(f"Failed to extract content from {filename}: {str(e)}")
            return []
    
    async def _set_document_status(
        self, 
        temp_doc_id: str, 
        status: str, 
        metadata: Dict[str, Any] = None
    ) -> None:
        """Set document processing status."""
        try:
            status_data = {
                'temp_doc_id': temp_doc_id,
                'status': status,
                'timestamp': datetime.now().isoformat(),
                **(metadata or {})
            }
            
            self.redis_cache.set(
                f"status:{temp_doc_id}",
                status_data,
                expire=3600  # 1 hour TTL for status
            )
        except Exception as e:
            logger.error(f"Failed to set status for {temp_doc_id}: {str(e)}")
    
    async def _update_document_ttl(self, temp_doc_id: str, ttl_hours: int) -> None:
        """Update TTL for a temporary document."""
        try:
            ttl_seconds = ttl_hours * 3600
            collection_name = f"temp_llamaindex_{temp_doc_id}"
            
            # Update index TTL
            self.redis_cache.expire(f"llamaindex_temp:{collection_name}", ttl_seconds)
            
            # Update metadata TTL
            self.redis_cache.expire(f"metadata:{temp_doc_id}", ttl_seconds)
            
            logger.info(f"Updated TTL for {temp_doc_id} to {ttl_hours} hours")
            
        except Exception as e:
            logger.error(f"Failed to update TTL for {temp_doc_id}: {str(e)}")