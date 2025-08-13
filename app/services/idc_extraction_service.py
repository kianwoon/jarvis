"""
IDC (Intelligent Document Comparison) Extraction Service

Handles document extraction to structured markdown using LLM intelligence.
Follows Jarvis patterns - NO hardcoding, uses environment variables and settings classes.
"""

import asyncio
import logging
import json
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import os
import requests

from app.core.config import get_settings
from app.core.llm_settings_cache import get_llm_settings
from app.core.idc_settings_cache import (
    get_idc_settings,
    get_extraction_config,
    get_extraction_system_prompt
)
from app.core.db import get_db_session
from app.core.redis_client import get_redis_client
from app.services.settings_prompt_service import get_prompt_service

logger = logging.getLogger(__name__)

class ExtractionMode(Enum):
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    QA_PAIRS = "qa_pairs"
    SECTION = "section"

@dataclass
class ExtractedUnit:
    index: int
    type: str
    content: str
    context_before: str = ""
    context_after: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class GranularExtractionConfig:
    mode: ExtractionMode
    max_unit_size: int = 2000  # tokens per unit
    overlap_size: int = 100    # overlap between units
    preserve_context: bool = True
    quality_threshold: float = 0.9
    extraction_model: str = None  # Will use Ollama settings
    max_context_usage: float = 0.35  # Maximum context usage percentage
    confidence_threshold: float = 0.8  # Minimum confidence threshold

@dataclass
class ExtractionResult:
    extracted_units: List[ExtractedUnit]
    extraction_metadata: Dict[str, Any]
    processing_time_ms: int
    model_used: str
    confidence_score: float
    total_tokens_used: int

class IDCExtractionService:
    """
    Document extraction service following Jarvis patterns:
    - Uses settings classes for configuration
    - Leverages Ollama models via OLLAMA_BASE_URL
    - Redis for caching and progress tracking
    - PostgreSQL for persistent storage
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = get_redis_client()
        self.prompt_service = get_prompt_service()
        
        # Get IDC-specific settings
        self.idc_settings = get_idc_settings()
        self.extraction_config = get_extraction_config(self.idc_settings)
        
        # Get Ollama configuration from settings (NO hardcoding)
        self.ollama_base_url = self.extraction_config.get('model_server')
        if not self.ollama_base_url:
            # Fallback to environment variable
            self.ollama_base_url = os.environ.get("OLLAMA_BASE_URL")
            if not self.ollama_base_url:
                raise ValueError("OLLAMA_BASE_URL must be configured - no hardcoded values allowed")
        
        # Get LLM model configuration from IDC settings
        self.default_model = self.extraction_config.get('model', 'qwen3:30b-a3b-q4_K_M')
        
        # Context limits from settings
        self.max_context_length = self.extraction_config.get('context_length', 8192)
        
        # Get extraction quality settings
        self.max_context_usage = self.extraction_config.get('max_context_usage', 0.35)
        self.confidence_threshold = self.extraction_config.get('confidence_threshold', 0.8)
        self.context_safety_buffer = self.extraction_config.get('safety_buffer', 1000)
        
    def _get_default_extraction_model(self) -> str:
        """Get default extraction model from IDC settings"""
        return self.extraction_config.get('model', 'qwen3:30b-a3b-q4_K_M')
    
    def _count_tokens(self, text: str) -> int:
        """
        Estimate token count for text (conservative estimate)
        Following Jarvis pattern - no hardcoded multipliers
        """
        # Conservative token estimation: 1 token per 3.5 characters on average
        return len(text) // 3
    
    async def _call_ollama(self, prompt: str, model: str = None, max_tokens: int = 4000) -> Dict[str, Any]:
        """
        Call Ollama API using configured base URL
        Following Jarvis pattern - uses environment configuration
        """
        if not model:
            model = self.default_model
            
        url = f"{self.ollama_base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": self.extraction_config.get('temperature', 0.1),
                "top_k": self.extraction_config.get('top_k', 20),
                "top_p": self.extraction_config.get('top_p', 0.9)
            }
        }
        
        try:
            from app.core.timeout_settings_cache import get_timeout_value
            timeout = get_timeout_value("document_processing", "document_processing_timeout", 120)
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            
            return {
                "response": result.get("response", ""),
                "model": result.get("model", model),
                "total_duration": result.get("total_duration", 0),
                "load_duration": result.get("load_duration", 0),
                "prompt_eval_count": result.get("prompt_eval_count", 0),
                "eval_count": result.get("eval_count", 0)
            }
            
        except Exception as e:
            logger.error(f"Ollama API call failed: {str(e)}")
            raise Exception(f"LLM extraction failed: {str(e)}")
    
    async def extract_document_to_markdown(
        self,
        document_content: str,
        document_type: str,
        extraction_model: str = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Extract document content to structured markdown
        Uses Ollama models configured via OLLAMA_BASE_URL
        """
        start_time = time.time()
        
        if not extraction_model:
            extraction_model = self.default_model
        
        # Apply max context usage limit
        max_content_length = int(self.max_context_length * self.max_context_usage)
        if len(document_content) > max_content_length:
            logger.warning(f"Document exceeds max context usage ({self.max_context_usage * 100}%), truncating from {len(document_content)} to {max_content_length} chars")
            document_content = document_content[:max_content_length]
        
        # Get extraction prompt from settings or prompt service
        extraction_system_prompt = get_extraction_system_prompt(self.idc_settings)
        
        # Try to get a more specific prompt from prompt service
        try:
            prompt_template = self.prompt_service.get_prompt(
                'idc_extraction',
                variables={
                    'document_type': document_type,
                    'document_content': document_content
                }
            )
            if prompt_template:
                extraction_prompt = prompt_template
            else:
                # Use system prompt with document content
                extraction_prompt = f"{extraction_system_prompt}\n\nDOCUMENT TYPE: {document_type}\n\nDOCUMENT CONTENT:\n{document_content}\n\nCLEAN MARKDOWN OUTPUT:"
        except Exception as e:
            logger.warning(f"Could not get prompt from service: {e}, using system prompt")
            extraction_prompt = f"{extraction_system_prompt}\n\nDOCUMENT TYPE: {document_type}\n\nDOCUMENT CONTENT:\n{document_content}\n\nCLEAN MARKDOWN OUTPUT:"
        
        try:
            result = await self._call_ollama(
                prompt=extraction_prompt,
                model=extraction_model,
                max_tokens=6000
            )
            
            extracted_markdown = result["response"].strip()
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            extraction_metadata = {
                "model_used": result["model"],
                "processing_time_ms": processing_time_ms,
                "original_length": len(document_content),
                "extracted_length": len(extracted_markdown),
                "tokens_used": result.get("eval_count", 0),
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "total_duration_ns": result.get("total_duration", 0),
                "extraction_timestamp": datetime.utcnow().isoformat()
            }
            
            # Calculate confidence based on extraction quality
            confidence_score = self._calculate_extraction_confidence(
                document_content, 
                extracted_markdown, 
                extraction_metadata
            )
            
            extraction_metadata["confidence_score"] = confidence_score
            extraction_metadata["confidence_threshold"] = self.confidence_threshold
            extraction_metadata["meets_threshold"] = confidence_score >= self.confidence_threshold
            
            if confidence_score < self.confidence_threshold:
                logger.warning(f"Extraction confidence {confidence_score:.2f} below threshold {self.confidence_threshold}")
            
            return extracted_markdown, extraction_metadata
            
        except Exception as e:
            logger.error(f"Document extraction failed: {str(e)}")
            raise
    
    def _calculate_extraction_confidence(
        self, 
        original_content: str, 
        extracted_markdown: str,
        metadata: Dict[str, Any]
    ) -> float:
        """
        Calculate extraction confidence score based on quality metrics
        """
        # Base confidence
        confidence = 0.8
        
        # Length preservation check
        length_ratio = len(extracted_markdown) / len(original_content)
        if 0.5 <= length_ratio <= 1.5:  # Reasonable length preservation
            confidence += 0.1
        else:
            confidence -= 0.2
        
        # Processing time check (reasonable processing indicates quality)
        processing_time = metadata.get("processing_time_ms", 0)
        if 5000 <= processing_time <= 120000:  # 5s to 2min is reasonable
            confidence += 0.05
        
        # Token usage efficiency
        tokens_used = metadata.get("tokens_used", 0)
        if tokens_used > 0:
            if tokens_used < 4000:  # Efficient token usage
                confidence += 0.05
        
        return min(1.0, max(0.1, confidence))
    
    async def extract_by_mode(
        self,
        document_content: str,
        extraction_config: GranularExtractionConfig
    ) -> ExtractionResult:
        """
        Extract document into units based on user-selected granularity
        """
        start_time = time.time()
        
        if extraction_config.mode == ExtractionMode.SENTENCE:
            extracted_units = await self._extract_sentences(document_content, extraction_config)
        elif extraction_config.mode == ExtractionMode.PARAGRAPH:
            extracted_units = await self._extract_paragraphs(document_content, extraction_config)
        elif extraction_config.mode == ExtractionMode.QA_PAIRS:
            extracted_units = await self._extract_qa_pairs(document_content, extraction_config)
        elif extraction_config.mode == ExtractionMode.SECTION:
            extracted_units = await self._extract_sections(document_content, extraction_config)
        else:
            raise ValueError(f"Unsupported extraction mode: {extraction_config.mode}")
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Calculate total tokens used
        total_tokens = sum(self._count_tokens(unit.content) for unit in extracted_units)
        
        # Calculate overall confidence
        confidence_score = self._calculate_unit_extraction_confidence(extracted_units)
        
        extraction_metadata = {
            "extraction_mode": extraction_config.mode.value,
            "total_units_extracted": len(extracted_units),
            "average_unit_size": total_tokens / len(extracted_units) if extracted_units else 0,
            "preserve_context": extraction_config.preserve_context,
            "processing_timestamp": datetime.utcnow().isoformat()
        }
        
        return ExtractionResult(
            extracted_units=extracted_units,
            extraction_metadata=extraction_metadata,
            processing_time_ms=processing_time_ms,
            model_used=extraction_config.extraction_model or self.default_model,
            confidence_score=confidence_score,
            total_tokens_used=total_tokens
        )
    
    async def _extract_sentences(
        self,
        content: str,
        config: GranularExtractionConfig
    ) -> List[ExtractedUnit]:
        """
        Extract individual sentences with context preservation
        """
        # Split into sentences using basic sentence boundaries
        sentences = self._split_into_sentences(content)
        extracted_units = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
                
            # Add context if requested
            context_before = ""
            context_after = ""
            
            if config.preserve_context:
                context_before = " ".join(sentences[max(0, i-2):i])
                context_after = " ".join(sentences[i+1:min(len(sentences), i+3)])
            
            unit = ExtractedUnit(
                index=len(extracted_units),
                type="sentence",
                content=sentence.strip(),
                context_before=context_before,
                context_after=context_after,
                metadata={
                    "sentence_number": i + 1,
                    "total_sentences": len(sentences),
                    "token_count": self._count_tokens(sentence),
                    "character_count": len(sentence)
                }
            )
            extracted_units.append(unit)
        
        return extracted_units
    
    async def _extract_paragraphs(
        self,
        content: str,
        config: GranularExtractionConfig
    ) -> List[ExtractedUnit]:
        """
        Extract paragraphs with intelligent boundary detection
        """
        paragraphs = self._split_into_paragraphs(content)
        extracted_units = []
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) < 50:  # Skip very short paragraphs
                continue
                
            unit = ExtractedUnit(
                index=len(extracted_units),
                type="paragraph",
                content=paragraph.strip(),
                metadata={
                    "paragraph_number": i + 1,
                    "character_count": len(paragraph),
                    "token_count": self._count_tokens(paragraph),
                    "estimated_reading_time": len(paragraph) / 1000  # rough estimate
                }
            )
            extracted_units.append(unit)
        
        return extracted_units
    
    async def _extract_qa_pairs(
        self,
        content: str,
        config: GranularExtractionConfig
    ) -> List[ExtractedUnit]:
        """
        Use LLM to extract question-answer pairs
        """
        extraction_model = config.extraction_model or self.default_model
        
        extraction_prompt = f"""
        Extract all question-answer pairs from this document.
        
        DOCUMENT CONTENT:
        {content}
        
        INSTRUCTIONS:
        1. Identify explicit questions and their answers
        2. Preserve the exact wording
        3. Include question numbers if present
        4. Return as structured JSON array
        
        RESPOND WITH VALID JSON ARRAY ONLY:
        [
            {{"question": "What is...", "answer": "The answer is...", "question_number": "1a"}},
            {{"question": "Calculate...", "answer": "F = ma = ...", "question_number": "2"}}
        ]
        """
        
        try:
            result = await self._call_ollama(
                prompt=extraction_prompt,
                model=extraction_model,
                max_tokens=4000
            )
            
            response_text = result["response"].strip()
            
            # Try to parse JSON response
            try:
                qa_pairs = json.loads(response_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, extract manually
                logger.warning("Failed to parse JSON response, attempting manual extraction")
                qa_pairs = []
            
            extracted_units = []
            for i, qa_pair in enumerate(qa_pairs):
                if not isinstance(qa_pair, dict) or 'question' not in qa_pair or 'answer' not in qa_pair:
                    continue
                    
                unit = ExtractedUnit(
                    index=i,
                    type="qa_pair",
                    content=f"Q: {qa_pair['question']}\nA: {qa_pair['answer']}",
                    metadata={
                        "question": qa_pair["question"],
                        "answer": qa_pair["answer"],
                        "question_number": qa_pair.get("question_number", str(i+1)),
                        "pair_index": i + 1,
                        "token_count": self._count_tokens(f"{qa_pair['question']} {qa_pair['answer']}")
                    }
                )
                extracted_units.append(unit)
            
            return extracted_units
            
        except Exception as e:
            logger.error(f"Q&A extraction failed: {str(e)}")
            raise
    
    async def _extract_sections(
        self,
        content: str,
        config: GranularExtractionConfig
    ) -> List[ExtractedUnit]:
        """
        Extract document sections based on headers and structure
        """
        sections = self._split_into_sections(content)
        extracted_units = []
        
        for i, section in enumerate(sections):
            if len(section.strip()) < 100:  # Skip very short sections
                continue
                
            # Extract section title if available
            lines = section.split('\n')
            section_title = lines[0].strip() if lines else f"Section {i+1}"
            
            unit = ExtractedUnit(
                index=len(extracted_units),
                type="section",
                content=section.strip(),
                metadata={
                    "section_title": section_title,
                    "section_number": i + 1,
                    "character_count": len(section),
                    "token_count": self._count_tokens(section),
                    "line_count": len(lines)
                }
            )
            extracted_units.append(unit)
        
        return extracted_units
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using basic punctuation rules"""
        import re
        
        # Basic sentence splitting pattern
        sentence_pattern = r'[.!?]+\s+'
        sentences = re.split(sentence_pattern, text)
        
        # Clean up sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 5:
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs using double newlines"""
        paragraphs = text.split('\n\n')
        
        # Clean up paragraphs
        clean_paragraphs = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip().replace('\n', ' ')
            if paragraph and len(paragraph) > 20:
                clean_paragraphs.append(paragraph)
        
        return clean_paragraphs
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into sections based on headers and structure"""
        import re
        
        # Look for section headers (lines that start with numbers, caps, or markdown headers)
        section_pattern = r'(\n\s*(?:\d+\.|\w+\.|\#+\s+|\b[A-Z][A-Z\s]+\b)\s*\n)'
        sections = re.split(section_pattern, text)
        
        # Combine headers with their content
        combined_sections = []
        for i in range(0, len(sections) - 1, 2):
            if i + 1 < len(sections):
                section = sections[i] + (sections[i + 1] if i + 1 < len(sections) else "")
                section = section.strip()
                if section and len(section) > 50:
                    combined_sections.append(section)
        
        # If no clear sections found, fall back to paragraph-based chunking
        if not combined_sections:
            combined_sections = self._split_into_paragraphs(text)
        
        return combined_sections
    
    def _calculate_unit_extraction_confidence(self, units: List[ExtractedUnit]) -> float:
        """Calculate confidence score for unit extraction"""
        if not units:
            return 0.0
        
        # Base confidence
        confidence = 0.8
        
        # Check unit size distribution
        token_counts = [unit.metadata.get('token_count', 0) for unit in units]
        avg_tokens = sum(token_counts) / len(token_counts)
        
        # Good average unit size (not too small, not too large)
        if 50 <= avg_tokens <= 500:
            confidence += 0.1
        
        # Check for reasonable number of units
        if 5 <= len(units) <= 1000:
            confidence += 0.05
        
        # Check content quality (basic heuristics)
        non_empty_units = sum(1 for unit in units if len(unit.content.strip()) > 10)
        if non_empty_units == len(units):
            confidence += 0.05
        
        return min(1.0, max(0.1, confidence))
    
    async def cache_extraction_progress(self, session_id: str, progress_data: Dict[str, Any]):
        """Cache extraction progress in Redis for real-time updates"""
        if self.redis_client:
            try:
                cache_key = f"idc_extraction_progress:{session_id}"
                self.redis_client.setex(
                    cache_key, 
                    3600,  # 1 hour TTL
                    json.dumps(progress_data)
                )
            except Exception as e:
                logger.warning(f"Failed to cache extraction progress: {e}")
    
    async def get_extraction_progress(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get extraction progress from Redis cache"""
        if self.redis_client:
            try:
                cache_key = f"idc_extraction_progress:{session_id}"
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Failed to get extraction progress: {e}")
        return None