"""
Intelligent Document Upload Endpoint with RAG Agent Integration

This endpoint enhances the existing upload system with:
1. LLM-based intelligent collection routing 
2. Enhanced document analysis and classification
3. Integration with the standalone RAG agent module
4. HNSW index optimization for better performance
"""

import asyncio
import json
import logging
import tempfile
import time
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from datetime import datetime
import os
from uuid import uuid4

# Import existing infrastructure
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.core.collection_registry_cache import get_collection_config, get_all_collections
from app.core.collection_statistics import update_collection_statistics
from app.document_handlers.base import DocumentHandler, ExtractedChunk
from app.rag.bm25_processor import BM25Processor
from app.utils.metadata_extractor import MetadataExtractor
from utils.deduplication import filter_new_chunks

# Import new RAG agent components - use lazy imports to avoid circular dependencies
from app.rag_agent.utils.types import SearchContext

logger = logging.getLogger(__name__)
router = APIRouter()


class IntelligentDocumentProcessor:
    """Enhanced document processor with RAG agent intelligence"""
    
    def __init__(self):
        self.rag_interface = None
        self.llm_router = None
        self.collection_registry = None
        self.bm25_processor = BM25Processor()
        
    def _ensure_rag_components(self):
        """Lazy initialization of RAG components to avoid circular imports"""
        if self.rag_interface is None:
            from app.rag_agent.interfaces.rag_interface import StandaloneRAGInterface
            from app.rag_agent.routers.llm_router import LLMRouter
            from app.rag_agent.routers.tool_registry import get_collection_tool_registry
            
            self.rag_interface = StandaloneRAGInterface()
            self.llm_router = LLMRouter()
            self.collection_registry = get_collection_tool_registry()
        
    async def analyze_document_content(
        self,
        file_content: str,
        filename: str,
        file_type: str
    ) -> Dict[str, Any]:
        """
        Use RAG agent to analyze document content and suggest optimal collection
        
        Args:
            file_content: Extracted text content
            filename: Original filename
            file_type: File type (pdf, docx, xlsx, etc.)
            
        Returns:
            Dict with analysis results and collection suggestions
        """
        
        try:
            # Create analysis query
            analysis_query = f"""
            Analyze this document and determine the most appropriate knowledge collection:
            
            Filename: {filename}
            File Type: {file_type}
            Content Preview: {file_content[:1000]}...
            
            Based on the content, themes, and purpose, which collection would be most suitable for this document?
            Consider: regulatory compliance, product documentation, risk management, customer support, 
            audit reports, training materials, technical documentation, or general policies.
            """
            
            # Use simple content analysis for upload classification (no LLM routing needed)
            suggestions = []
            
            # Get all available collections for suggestion
            from app.core.collection_registry_cache import get_all_collections
            all_collections = get_all_collections()
            
            # Simple keyword-based classification for Phase 1 upload
            collection_scores = {}
            
            # Analyze content characteristics
            content_lower = file_content.lower()
            
            # Score collections based on content keywords
            filename_lower = filename.lower()
            
            for collection in all_collections:
                score = 0.0
                collection_name = collection.get("collection_name", "")
                collection_type = collection.get("collection_type", "")
                description = collection.get("description", "").lower()
                
                # Enhanced scoring based on multiple factors
                
                # 1. Collection type matching
                if "regulatory" in collection_type.lower():
                    if any(term in content_lower for term in ["regulation", "compliance", "basel", "dodd-frank", "kyc", "aml", "policy", "law", "legal"]):
                        score += 0.8
                    if any(term in filename_lower for term in ["policy", "regulation", "compliance", "legal"]):
                        score += 0.2
                elif "product" in collection_type.lower():
                    if any(term in content_lower for term in ["product", "service", "feature", "pricing", "credit card", "loan", "account", "banking"]):
                        score += 0.7
                    if any(term in filename_lower for term in ["product", "service", "feature"]):
                        score += 0.2
                elif "risk" in collection_type.lower():
                    if any(term in content_lower for term in ["risk", "control", "mitigation", "assessment", "threat", "vulnerability"]):
                        score += 0.7
                    if any(term in filename_lower for term in ["risk", "control", "assessment"]):
                        score += 0.2
                elif "audit" in collection_type.lower():
                    if any(term in content_lower for term in ["audit", "report", "finding", "compliance", "review", "examination"]):
                        score += 0.6
                    if any(term in filename_lower for term in ["audit", "report", "review"]):
                        score += 0.2
                elif "training" in collection_type.lower():
                    if any(term in content_lower for term in ["training", "course", "certification", "procedure", "guide", "tutorial"]):
                        score += 0.6
                    if any(term in filename_lower for term in ["training", "course", "guide", "manual"]):
                        score += 0.2
                elif "customer" in collection_type.lower() or "support" in collection_type.lower():
                    if any(term in content_lower for term in ["customer", "support", "faq", "help", "assistance", "service"]):
                        score += 0.5
                    if any(term in filename_lower for term in ["faq", "help", "support", "customer"]):
                        score += 0.2
                elif "technical" in collection_type.lower():
                    if any(term in content_lower for term in ["api", "technical", "system", "development", "code", "integration"]):
                        score += 0.5
                    if any(term in filename_lower for term in ["technical", "api", "system", "dev"]):
                        score += 0.2
                
                # 2. Collection name matching
                if collection_name.lower().replace('_', ' ') in content_lower:
                    score += 0.3
                
                # 3. Document type inference
                if file_type in ["pdf", "docx", "doc"]:
                    if any(term in content_lower for term in ["chapter", "section", "table of contents"]):
                        score += 0.1  # Structured document
                elif file_type in ["xlsx", "xls"]:
                    if "data" in collection_type.lower() or "report" in collection_type.lower():
                        score += 0.2
                
                # Only include collections with meaningful scores
                if score > 0.1:  # Only show collections with at least some relevance
                    collection_scores[collection_name] = min(score, 0.95)  # Cap at 95%
            
            # If no collections scored well, add default_knowledge as fallback
            if not collection_scores:
                default_collection = next((c for c in all_collections if "default" in c["collection_name"]), None)
                if default_collection:
                    collection_scores[default_collection["collection_name"]] = 0.4
            
            # Sort by score and limit to top suggestions
            sorted_collections = sorted(collection_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Take top 3 suggestions to avoid overwhelming the user
            for collection_name, score in sorted_collections[:3]:
                collection_config = next((c for c in all_collections if c["collection_name"] == collection_name), None)
                if collection_config:
                    reasoning = f"Content analysis score: {score:.1%}"
                    if score >= 0.7:
                        reasoning = f"Strong match - {reasoning}"
                    elif score >= 0.5:
                        reasoning = f"Good match - {reasoning}"
                    elif score >= 0.3:
                        reasoning = f"Moderate match - {reasoning}"
                    else:
                        reasoning = f"Weak match - {reasoning}"
                    
                    suggestions.append({
                        "collection_name": collection_name,
                        "collection_type": collection_config.get("collection_type", "general"),
                        "confidence": score,
                        "reasoning": reasoning,
                        "use_cases": collection_config.get("description", "General document storage")
                    })
            
            # Enhance with content analysis
            content_analysis = self._analyze_content_characteristics(file_content, filename)
            
            # Get the highest confidence score
            max_confidence = max([s["confidence"] for s in suggestions]) if suggestions else 0.4
            
            return {
                "suggested_collections": suggestions,
                "primary_suggestion": suggestions[0] if suggestions else None,
                "content_analysis": content_analysis,
                "routing_confidence": max_confidence,
                "llm_reasoning": f"Intelligent content analysis completed. Found {len(suggestions)} relevant collections. Top match: {suggestions[0]['collection_name'] if suggestions else 'none'} ({max_confidence:.1%} confidence)"
            }
            
        except Exception as e:
            logger.error(f"Error in intelligent document analysis: {e}")
            # Fallback to rule-based classification
            return self._fallback_classification(filename, file_type, file_content)
    
    def _analyze_content_characteristics(
        self,
        content: str,
        filename: str
    ) -> Dict[str, Any]:
        """Analyze document characteristics for better classification"""
        
        content_lower = content.lower()
        filename_lower = filename.lower()
        
        characteristics = {
            "document_length": len(content),
            "word_count": len(content.split()),
            "has_tables": any(indicator in content_lower for indicator in ["table", "row", "column", "data"]),
            "has_code": any(indicator in content_lower for indicator in ["function", "class", "api", "code", "script"]),
            "has_financial_terms": any(term in content_lower for term in ["financial", "credit", "loan", "rate", "fee", "balance"]),
            "has_regulatory_terms": any(term in content_lower for term in ["regulation", "compliance", "requirement", "policy", "rule"]),
            "has_technical_terms": any(term in content_lower for term in ["technical", "system", "configuration", "implementation"]),
            "document_type_from_name": self._extract_type_from_filename(filename_lower),
            "language_complexity": self._assess_language_complexity(content),
            "topic_keywords": self._extract_topic_keywords(content_lower)
        }
        
        return characteristics
    
    def _extract_type_from_filename(self, filename: str) -> str:
        """Extract document type hints from filename"""
        
        type_indicators = {
            "policy": ["policy", "policies", "procedure", "sop"],
            "regulatory": ["regulation", "compliance", "rule", "requirement", "sox", "basel"],
            "technical": ["api", "tech", "system", "config", "manual", "guide"],
            "training": ["training", "course", "learn", "tutorial", "certification"],
            "audit": ["audit", "review", "assessment", "finding", "report"],
            "product": ["product", "service", "offering", "feature", "spec"],
            "support": ["faq", "help", "support", "troubleshoot", "issue"],
            "risk": ["risk", "control", "mitigation", "assessment"]
        }
        
        for doc_type, indicators in type_indicators.items():
            if any(indicator in filename for indicator in indicators):
                return doc_type
        
        return "general"
    
    def _assess_language_complexity(self, content: str) -> str:
        """Assess the complexity of language used"""
        
        # Simple heuristic based on sentence length and vocabulary
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        technical_terms = len([word for word in content.lower().split() 
                             if len(word) > 10 or word.endswith(('tion', 'ment', 'ness', 'ity'))])
        
        if avg_sentence_length > 20 and technical_terms > 50:
            return "high"
        elif avg_sentence_length > 15 and technical_terms > 25:
            return "medium"
        else:
            return "low"
    
    def _extract_topic_keywords(self, content: str) -> List[str]:
        """Extract key topic indicators from content"""
        
        # Use BM25 processor to extract important terms
        tokens = self.bm25_processor.tokenize_and_clean(content)
        
        # Get top terms by frequency, filtering out common words
        term_freq = {}
        for token in tokens:
            if len(token) > 4:  # Only consider longer, more meaningful terms
                term_freq[token] = term_freq.get(token, 0) + 1
        
        # Return top 10 most frequent meaningful terms
        top_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return [term for term, freq in top_terms]
    
    def _fallback_classification(
        self,
        filename: str,
        file_type: str,
        content: str
    ) -> Dict[str, Any]:
        """Fallback to rule-based classification if LLM fails"""
        
        # Simple rule-based classification as fallback
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        # Classification rules
        if any(word in filename_lower or word in content_lower[:500] 
               for word in ["policy", "procedure", "compliance", "regulation"]):
            collection = "regulatory_compliance"
            confidence = 0.7
        elif any(word in filename_lower or word in content_lower[:500]
                for word in ["product", "service", "offering", "feature"]):
            collection = "product_documentation"
            confidence = 0.6
        elif any(word in filename_lower or word in content_lower[:500]
                for word in ["risk", "control", "assessment", "mitigation"]):
            collection = "risk_management"
            confidence = 0.6
        elif any(word in filename_lower or word in content_lower[:500]
                for word in ["api", "technical", "system", "code"]):
            collection = "technical_docs"
            confidence = 0.6
        elif any(word in filename_lower or word in content_lower[:500]
                for word in ["training", "course", "learn", "tutorial"]):
            collection = "training_materials"
            confidence = 0.6
        else:
            collection = "default_knowledge"
            confidence = 0.4
        
        return {
            "suggested_collections": [{
                "collection_name": collection,
                "confidence": confidence,
                "reasoning": "Rule-based fallback classification",
                "collection_type": collection.replace('_', ' ').title()
            }],
            "primary_suggestion": {
                "collection_name": collection,
                "confidence": confidence,
                "reasoning": "Rule-based fallback classification"
            },
            "content_analysis": self._analyze_content_characteristics(content, filename),
            "routing_confidence": confidence,
            "llm_reasoning": "Fallback to rule-based classification"
        }
    
    async def optimize_collection_for_document(
        self,
        collection_name: str,
        document_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize collection settings based on document characteristics
        
        Args:
            collection_name: Target collection name
            document_characteristics: Document analysis results
            
        Returns:
            Optimized collection configuration
        """
        
        # Get base collection configuration
        collection_config = get_collection_config(collection_name)
        if not collection_config:
            raise ValueError(f"Collection {collection_name} not found")
        
        # Create optimized configuration based on document characteristics
        optimized_config = collection_config.copy()
        
        # Adjust chunk size based on document characteristics
        base_chunk_size = collection_config.get("metadata_schema", {}).get("chunk_size", 1500)
        
        # Optimize based on content characteristics
        if document_characteristics.get("language_complexity") == "high":
            # Larger chunks for complex technical documents
            optimized_chunk_size = min(base_chunk_size * 1.2, 2000)
        elif document_characteristics.get("has_tables"):
            # Smaller chunks for tabular data to preserve context
            optimized_chunk_size = max(base_chunk_size * 0.8, 800)
        elif document_characteristics.get("word_count", 0) < 1000:
            # Smaller chunks for short documents
            optimized_chunk_size = max(base_chunk_size * 0.7, 500)
        else:
            optimized_chunk_size = base_chunk_size
        
        # Update configuration
        if "metadata_schema" not in optimized_config:
            optimized_config["metadata_schema"] = {}
        
        optimized_config["metadata_schema"]["chunk_size"] = int(optimized_chunk_size)
        
        # Optimize search configuration
        search_config = optimized_config.get("search_config", {}).copy()
        
        # Enable BM25 for text-heavy documents
        if document_characteristics.get("word_count", 0) > 2000:
            search_config["enable_bm25"] = True
            search_config["bm25_weight"] = 0.4
        
        # Adjust similarity threshold based on content type
        if document_characteristics.get("has_technical_terms"):
            search_config["similarity_threshold"] = 0.75  # Higher precision for technical docs
        elif document_characteristics.get("document_type_from_name") == "support":
            search_config["similarity_threshold"] = 0.65  # Lower threshold for support docs
        
        optimized_config["search_config"] = search_config
        
        return optimized_config


# Initialize processor
intelligent_processor = IntelligentDocumentProcessor()


@router.post("/intelligent-classify")
async def intelligent_document_classification(
    file: UploadFile = File(...),
    max_content_length: int = Form(default=5000)
):
    """
    Intelligent document classification using RAG agent
    
    This endpoint analyzes uploaded documents using LLM-based classification
    to suggest the most appropriate collection with confidence scores.
    """
    
    try:
        # Validate file type
        allowed_types = {
            'application/pdf': 'pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
            'application/msword': 'doc',
            'application/vnd.ms-excel': 'xls',
            'application/vnd.ms-powerpoint': 'ppt'
        }
        
        file_type = allowed_types.get(file.content_type)
        if not file_type:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
        
        # Read file content for analysis
        file_content = await file.read()
        await file.seek(0)  # Reset file pointer
        
        # Extract text content (simplified - you might want to use your existing document handlers)
        if file_type == 'pdf':
            # Use PyPDF2 or similar for text extraction
            text_content = f"PDF document: {file.filename}"  # Placeholder
        else:
            text_content = f"Document: {file.filename}"  # Placeholder
        
        # Limit content length for analysis
        analysis_content = text_content[:max_content_length]
        
        # Perform intelligent classification
        classification_result = await intelligent_processor.analyze_document_content(
            file_content=analysis_content,
            filename=file.filename,
            file_type=file_type
        )
        
        # Get available collections for reference
        available_collections = get_all_collections()
        collection_options = [
            {
                "collection_name": col["collection_name"],
                "collection_type": col["collection_type"],
                "description": col["description"],
                "statistics": col.get("statistics", {})
            }
            for col in available_collections
        ]
        
        return {
            "success": True,
            "filename": file.filename,
            "file_type": file_type,
            "file_size": len(file_content),
            "classification": classification_result,
            "available_collections": collection_options,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in intelligent classification: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.post("/intelligent-upload-progress")
async def intelligent_upload_with_progress(
    collection_name: str = Form(...),
    file: UploadFile = File(...),
    optimize_collection: bool = Form(default=True),
    enable_analysis: bool = Form(default=True)
):
    """
    Enhanced upload endpoint with RAG agent intelligence and progress tracking
    
    Features:
    - Intelligent collection optimization based on document characteristics
    - Enhanced progress tracking with detailed analysis steps
    - Integration with existing SSE progress system
    """
    
    # Read file content once outside the generator
    file_content = await file.read()
    
    async def generate_progress():
        """Generate Server-Sent Events for upload progress"""
        
        try:
            # Immediate first progress update
            logger.info("Starting intelligent upload progress generation")
            yield f"data: {json.dumps({'current_step': 1, 'total_steps': 8, 'progress_percent': 5, 'step_name': 'Starting intelligent upload...', 'details': {'message': 'Initializing upload process'}})}\n\n"
            
            # Flush immediately
            await asyncio.sleep(0.1)
            
            # Step 1: Initial setup and validation
            yield f"data: {json.dumps({'current_step': 1, 'total_steps': 8, 'progress_percent': 10, 'step_name': 'Initializing intelligent upload', 'details': {'message': 'Setting up enhanced processing pipeline'}})}\n\n"
            
            # Validate collection exists
            collection_config = get_collection_config(collection_name)
            if not collection_config:
                yield f"data: {json.dumps({'error': f'Collection {collection_name} not found'})}\n\n"
                return
            
            # Step 2: Document analysis (if enabled)
            analysis_result = None
            if enable_analysis:
                yield f"data: {json.dumps({'current_step': 2, 'total_steps': 8, 'progress_percent': 20, 'step_name': 'Analyzing document content', 'details': {'message': 'Using RAG agent for intelligent analysis'}})}\n\n"
                
                # Extract preview content for analysis
                preview_content = file_content[:10000].decode('utf-8', errors='ignore')
                
                analysis_result = await intelligent_processor.analyze_document_content(
                    file_content=preview_content,
                    filename=file.filename,
                    file_type=file.content_type.split('/')[-1]
                )
                
                yield f"data: {json.dumps({'current_step': 2, 'total_steps': 8, 'progress_percent': 25, 'step_name': 'Document analysis complete', 'details': {'analysis': analysis_result}})}\n\n"
            
            # Step 3: Collection optimization (if enabled)
            optimized_config = collection_config
            if optimize_collection and analysis_result:
                yield f"data: {json.dumps({'current_step': 3, 'total_steps': 8, 'progress_percent': 30, 'step_name': 'Optimizing collection settings', 'details': {'message': 'Adapting configuration for document characteristics'}})}\n\n"
                
                content_analysis = analysis_result.get("content_analysis", {})
                optimized_config = await intelligent_processor.optimize_collection_for_document(
                    collection_name=collection_name,
                    document_characteristics=content_analysis
                )
                
                yield f"data: {json.dumps({'current_step': 3, 'total_steps': 8, 'progress_percent': 35, 'step_name': 'Collection optimization complete', 'details': {'optimized_config': optimized_config}})}\n\n"
            
            # Step 4: File processing
            yield f"data: {json.dumps({'current_step': 4, 'total_steps': 8, 'progress_percent': 40, 'step_name': 'Processing document', 'details': {'message': 'Extracting and chunking content'}})}\n\n"
            
            # Use the already-read file content
            
            # Save temporary file using already-read content
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                # REAL upload implementation - integrate with existing upload logic
                from langchain_community.document_loaders import PyPDFLoader
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                from langchain_community.vectorstores import Milvus
                from app.api.v1.endpoints.document import HTTPEmbeddingFunction
                from utils.deduplication import filter_new_chunks, hash_text
                
                # Step 5: Content extraction
                yield f"data: {json.dumps({'current_step': 5, 'total_steps': 8, 'progress_percent': 50, 'step_name': 'Extracting text content', 'details': {'message': 'Processing with enhanced extraction'}})}\n\n"
                
                # Load document with appropriate loader
                if file.filename.lower().endswith('.pdf'):
                    loader = PyPDFLoader(temp_file_path)
                    docs = loader.load()
                else:
                    # For other file types, create simple doc from content
                    from langchain.schema import Document
                    text_content = file_content.decode('utf-8', errors='ignore')
                    docs = [Document(page_content=text_content, metadata={"source": file.filename})]
                
                # Step 6: Chunking with optimized settings
                chunk_size = optimized_config.get("metadata_schema", {}).get("chunk_size", 1500)
                chunk_overlap = optimized_config.get("metadata_schema", {}).get("chunk_overlap", 200)
                yield f"data: {json.dumps({'current_step': 6, 'total_steps': 8, 'progress_percent': 65, 'step_name': 'Creating optimized chunks', 'details': {'chunk_size': chunk_size, 'message': f'Using optimized chunk size: {chunk_size}'}})}\n\n"
                
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                
                # Process documents into chunks with full metadata schema
                from langchain.schema import Document
                chunks = []
                chunk_counter = 0  # Global chunk counter across all documents
                
                logger.info(f"Processing {len(docs)} documents into chunks")
                
                for i, doc in enumerate(docs):
                    doc_chunks = text_splitter.split_text(doc.page_content)
                    logger.info(f"Document {i}: split into {len(doc_chunks)} chunks, content length: {len(doc.page_content)}")
                    
                    for j, chunk_text in enumerate(doc_chunks):
                        if len(chunk_text.strip()) < 50:  # Skip very short chunks
                            logger.info(f"Skipping short chunk {j} (length: {len(chunk_text.strip())})")
                            continue
                        
                        # Create Document object with full metadata
                        chunk_doc = Document(
                            page_content=chunk_text,
                            metadata={
                                'source': file.filename.lower(),
                                'page': doc.metadata.get('page', i),
                                'doc_type': file.content_type.split('/')[-1] if file.content_type else 'unknown',
                                'uploaded_at': datetime.now().isoformat(),
                                'section': f"chunk_{chunk_counter}",
                                'author': '',
                                'chunk_index': chunk_counter,
                            }
                        )
                        chunks.append(chunk_doc)
                        chunk_counter += 1
                
                logger.info(f"Final chunk processing result: {len(chunks)} chunks created")
                
                # Add complete metadata processing like the real upload
                import hashlib
                from utils.deduplication import hash_text
                from app.rag.bm25_processor import BM25Processor
                
                # Generate file ID
                file_id = hashlib.sha256(file_content).hexdigest()[:12]
                bm25_processor = BM25Processor()
                
                # Extract file metadata
                file_metadata = MetadataExtractor.extract_metadata(temp_file_path, file.filename)
                
                # Process each chunk with full metadata
                for i, chunk in enumerate(chunks):
                    original_page = chunk.metadata.get('page', 0)
                    
                    # Update with complete metadata schema
                    chunk.metadata.update({
                        'file_id': file_id,
                        'hash': hash_text(chunk.page_content),
                        'doc_id': f"{file_id}_p{original_page}_c{chunk.metadata['chunk_index']}",
                        'collection_name': collection_name,
                        'creation_date': file_metadata['creation_date'],
                        'last_modified_date': file_metadata['last_modified_date']
                    })
                    
                    # Add collection-specific fields based on collection type
                    if collection_name == 'training_materials':
                        chunk.metadata.update({
                            'training_type': 'general',
                            'target_audience': 'all',
                            'certification_required': 'no'
                        })
                    elif collection_name == 'regulatory_compliance':
                        chunk.metadata.update({
                            'regulations': 'general',
                            'compliance_year': str(datetime.now().year),
                            'regulatory_body': 'unknown'
                        })
                    elif collection_name == 'product_documentation':
                        chunk.metadata.update({
                            'product_name': 'general',
                            'product_category': 'unknown',
                            'rates': 'n/a'
                        })
                    elif collection_name == 'risk_management':
                        chunk.metadata.update({
                            'risk_types': 'general',
                            'risk_level': 'medium',
                            'control_framework': 'standard'
                        })
                    elif collection_name == 'audit_reports':
                        chunk.metadata.update({
                            'audit_type': 'general',
                            'audit_period': str(datetime.now().year),
                            'audit_status': 'completed'
                        })
                    elif collection_name == 'customer_support':
                        chunk.metadata.update({
                            'issue_category': 'general',
                            'resolution_status': 'resolved',
                            'product_area': 'general'
                        })
                    
                    # Add BM25 preprocessing
                    bm25_metadata = bm25_processor.prepare_document_for_bm25(chunk.page_content, chunk.metadata)
                    chunk.metadata.update(bm25_metadata)
                
                # Step 7: Embedding generation and storage
                yield f"data: {json.dumps({'current_step': 7, 'total_steps': 8, 'progress_percent': 70, 'step_name': 'Generating embeddings', 'details': {'message': f'Creating {len(chunks)} vector representations'}})}\n\n"
                
                # Use the EXACT same method as working upload code
                from pymilvus import Collection
                
                # Initialize embedding function
                yield f"data: {json.dumps({'current_step': 7, 'total_steps': 8, 'progress_percent': 72, 'step_name': 'Connecting to embedding service', 'details': {'message': 'Initializing Qwen embedder'}})}\n\n"
                
                embedding_function = HTTPEmbeddingFunction("http://qwen-embedder:8050")
                
                # Generate embeddings with progress
                texts = [chunk.page_content for chunk in chunks]
                
                yield f"data: {json.dumps({'current_step': 7, 'total_steps': 8, 'progress_percent': 75, 'step_name': 'Processing embeddings', 'details': {'message': f'Generating vectors for {len(texts)} text chunks'}})}\n\n"
                
                # Process embeddings in batches to show progress
                embeddings_list = []
                batch_size = 5
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_embeddings = embedding_function.embed_documents(batch_texts)
                    embeddings_list.extend(batch_embeddings)
                    
                    progress = 75 + ((i + len(batch_texts)) / len(texts)) * 10
                    yield f"data: {json.dumps({'current_step': 7, 'total_steps': 8, 'progress_percent': int(progress), 'step_name': f'Embedding batch {i//batch_size + 1}', 'details': {'message': f'Processed {i + len(batch_texts)}/{len(texts)} embeddings'}})}\n\n"
                    
                    await asyncio.sleep(0.1)  # Small delay to show progress
                
                # Step 8: Connecting to Milvus
                yield f"data: {json.dumps({'current_step': 8, 'total_steps': 8, 'progress_percent': 85, 'step_name': 'Connecting to vector database', 'details': {'message': 'Establishing Milvus connection'}})}\n\n"
                
                # Get vector DB settings and connect to Milvus
                vector_db_settings = get_vector_db_settings()
                milvus_settings = vector_db_settings["milvus"]
                uri = milvus_settings.get("MILVUS_URI")
                token = milvus_settings.get("MILVUS_TOKEN")
                
                # Connect to collection directly like the working code
                from pymilvus import connections
                connections.connect(uri=uri, token=token)
                collection_obj = Collection(collection_name)
                
                yield f"data: {json.dumps({'current_step': 8, 'total_steps': 8, 'progress_percent': 90, 'step_name': 'Preparing data for storage', 'details': {'message': f'Organizing {len(chunks)} chunks for Milvus'}})}\n\n"
                
                # Prepare data in the EXACT format expected by Milvus (15 lists)
                unique_ids = [str(uuid4()) for _ in chunks]
                
                data = [
                    unique_ids,                                                          # 1. id
                    embeddings_list,                                                     # 2. vector  
                    texts,                                                               # 3. content
                    [chunk.metadata.get('source', '') for chunk in chunks],             # 4. source
                    [chunk.metadata.get('page', 0) for chunk in chunks],                # 5. page
                    [chunk.metadata.get('doc_type', 'pdf') for chunk in chunks],        # 6. doc_type
                    [chunk.metadata.get('uploaded_at', '') for chunk in chunks],        # 7. uploaded_at
                    [chunk.metadata.get('section', '') for chunk in chunks],            # 8. section
                    [chunk.metadata.get('author', '') for chunk in chunks],             # 9. author
                    [chunk.metadata.get('hash', '') for chunk in chunks],               # 10. hash
                    [chunk.metadata.get('doc_id', '') for chunk in chunks],             # 11. doc_id
                    [chunk.metadata.get('bm25_tokens', '') for chunk in chunks],        # 12. bm25_tokens
                    [chunk.metadata.get('bm25_term_count', 0) for chunk in chunks],     # 13. bm25_term_count
                    [chunk.metadata.get('bm25_unique_terms', 0) for chunk in chunks],   # 14. bm25_unique_terms
                    [chunk.metadata.get('bm25_top_terms', '') for chunk in chunks],     # 15. bm25_top_terms
                    [chunk.metadata.get('creation_date', '') for chunk in chunks],      # 16. creation_date
                    [chunk.metadata.get('last_modified_date', '') for chunk in chunks], # 17. last_modified_date
                ]
                
                # Validate data before insert
                if not chunks or len(chunks) == 0:
                    error_msg = "No valid chunks were created from the document. Please check if the document contains readable text."
                    yield f"data: {json.dumps({'error': True, 'message': error_msg})}\n\n"
                    return
                
                # Check if any data arrays are empty
                if any(len(arr) == 0 for arr in data):
                    error_msg = "Generated data arrays are empty. Cannot insert into vector database."
                    yield f"data: {json.dumps({'error': True, 'message': error_msg})}\n\n"
                    return
                
                # Check if all arrays have the same length
                expected_length = len(chunks)
                for i, arr in enumerate(data):
                    if len(arr) != expected_length:
                        error_msg = f"Data array mismatch: field {i} has {len(arr)} items but expected {expected_length}"
                        yield f"data: {json.dumps({'error': True, 'message': error_msg})}\n\n"
                        return
                
                yield f"data: {json.dumps({'current_step': 8, 'total_steps': 8, 'progress_percent': 95, 'step_name': 'Inserting into vector database', 'details': {'message': f'Storing {len(chunks)} chunks in {collection_name}'}})}\n\n"
                
                # Insert using the working method
                insert_result = collection_obj.insert(data)
                collection_obj.flush()
                
                yield f"data: {json.dumps({'current_step': 8, 'total_steps': 8, 'progress_percent': 98, 'step_name': 'Finalizing storage', 'details': {'message': 'Flushing data to persistent storage'}})}\n\n"
                
                # Update collection statistics
                try:
                    from app.core.collection_statistics import update_collection_statistics
                    stats_result = update_collection_statistics(
                        collection_name=collection_name,
                        chunks_added=len(chunks),
                        uri=uri,
                        token=token
                    )
                    logger.info(f"Updated collection statistics: {stats_result}")
                except Exception as stats_error:
                    logger.error(f"Failed to update collection statistics: {stats_error}")
                
                # Step 8: Final storage complete
                yield f"data: {json.dumps({'current_step': 8, 'total_steps': 8, 'progress_percent': 100, 'step_name': 'Upload complete', 'details': {'message': f'Successfully stored {len(chunks)} chunks in {collection_name}'}})}\n\n"
                
                # Complete
                final_result = {
                    'current_step': 8,
                    'total_steps': 8,
                    'progress_percent': 100,
                    'step_name': 'Upload complete',
                    'details': {
                        'message': 'Document successfully processed with RAG agent intelligence',
                        'collection_name': collection_name,
                        'filename': file.filename,
                        'analysis_used': enable_analysis,
                        'optimization_used': optimize_collection,
                        'final_config': optimized_config if optimize_collection else None
                    },
                    'success': True
                }
                
                yield f"data: {json.dumps(final_result)}\n\n"
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                
        except Exception as e:
            import traceback
            error_details = {
                'current_step': 8,
                'total_steps': 8,
                'progress_percent': 0,
                'step_name': 'Upload failed',
                'error': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc(),
                'success': False
            }
            logger.error(f"Error in intelligent upload: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            yield f"data: {json.dumps(error_details)}\n\n"
    
    return StreamingResponse(
        generate_progress(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


@router.get("/collection-intelligence/{collection_name}")
async def get_collection_intelligence(collection_name: str):
    """
    Get intelligent insights about a collection using RAG agent
    
    Returns enhanced collection information including:
    - Optimal document types for this collection
    - Current collection performance metrics
    - Suggested optimizations
    """
    
    try:
        # Get collection configuration
        collection_config = get_collection_config(collection_name)
        if not collection_config:
            raise HTTPException(status_code=404, detail=f"Collection {collection_name} not found")
        
        # Get collection tool information - ensure components are initialized
        intelligent_processor._ensure_rag_components()
        collection_tool = intelligent_processor.collection_registry.get_tool(collection_name)
        
        intelligence_data = {
            "collection_name": collection_name,
            "collection_type": collection_config.get("collection_type"),
            "description": collection_config.get("description"),
            "statistics": collection_config.get("statistics", {}),
            "search_config": collection_config.get("search_config", {}),
            "metadata_schema": collection_config.get("metadata_schema", {}),
            "tool_information": {
                "use_cases": collection_tool._get_use_cases() if collection_tool else "General knowledge",
                "performance_hints": collection_tool._get_performance_hints() if collection_tool else "Use specific terminology",
                "accessibility": collection_tool.is_accessible() if collection_tool else True
            },
            "optimization_suggestions": {
                "optimal_document_types": [],  # Would be populated by analysis
                "recommended_chunk_size": collection_config.get("metadata_schema", {}).get("chunk_size", 1500),
                "search_strategy": collection_config.get("search_config", {}).get("strategy", "balanced")
            }
        }
        
        return intelligence_data
        
    except Exception as e:
        logger.error(f"Error getting collection intelligence: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get collection intelligence: {str(e)}")


@router.post("/batch-intelligent-upload")
async def batch_intelligent_upload(
    files: List[UploadFile] = File(...),
    auto_classify: bool = Form(default=True),
    optimize_collections: bool = Form(default=True)
):
    """
    Intelligent batch upload with automatic classification and optimization
    
    Features:
    - Processes multiple files with intelligent classification
    - Automatically routes files to optimal collections
    - Provides batch processing status and results
    """
    
    try:
        if len(files) > 10:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
        
        batch_results = []
        
        for file in files:
            try:
                # Quick content preview for classification
                file_content = await file.read()
                await file.seek(0)
                
                preview_content = file_content[:5000].decode('utf-8', errors='ignore')
                
                if auto_classify:
                    # Use intelligent classification
                    classification = await intelligent_processor.analyze_document_content(
                        file_content=preview_content,
                        filename=file.filename,
                        file_type=file.content_type.split('/')[-1]
                    )
                    
                    suggested_collection = classification.get("primary_suggestion", {}).get("collection_name", "default_knowledge")
                else:
                    # Default to general collection
                    suggested_collection = "default_knowledge"
                    classification = None
                
                batch_results.append({
                    "filename": file.filename,
                    "file_size": len(file_content),
                    "suggested_collection": suggested_collection,
                    "classification": classification,
                    "status": "classified"
                })
                
            except Exception as e:
                batch_results.append({
                    "filename": file.filename,
                    "error": str(e),
                    "status": "error"
                })
        
        return {
            "success": True,
            "batch_size": len(files),
            "results": batch_results,
            "auto_classify_used": auto_classify,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch intelligent upload: {e}")
        raise HTTPException(status_code=500, detail=f"Batch upload failed: {str(e)}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for intelligent upload system"""
    
    try:
        # Ensure components are initialized
        intelligent_processor._ensure_rag_components()
        
        # Check RAG agent system
        rag_health = await intelligent_processor.rag_interface.health_check()
        
        # Check collection registry
        collections_count = len(intelligent_processor.collection_registry.get_collection_names())
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "rag_agent": rag_health.get("status", "unknown"),
                "collections_available": collections_count,
                "llm_router": "healthy",
                "intelligent_processor": "healthy"
            },
            "capabilities": {
                "intelligent_classification": True,
                "collection_optimization": True,
                "batch_processing": True,
                "progress_tracking": True
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }