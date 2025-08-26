"""
IDC (Intelligent Document Comparison) Validation Service

Performs systematic unit-by-unit validation against reference documents.
Follows Jarvis patterns - uses settings classes and conservative context management.
"""

import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import os
import requests
from uuid import uuid4

from app.core.config import get_settings
from app.core.llm_settings_cache import get_llm_settings
from app.core.idc_settings_cache import (
    get_idc_settings,
    get_validation_config,
    get_validation_system_prompt
)
from app.core.db import get_db_session, IDCValidationSession, IDCUnitValidationResult
from app.core.redis_client import get_redis_client
from app.core.timeout_settings_cache import get_redis_cache_ttl
from app.services.idc_extraction_service import ExtractedUnit, ExtractionResult
from app.services.settings_prompt_service import get_prompt_service

logger = logging.getLogger(__name__)

@dataclass
class UnitValidationResult:
    unit: ExtractedUnit
    validation_score: float
    confidence_score: float
    validation_feedback: str
    matched_reference_sections: List[str]
    similarity_scores: Dict[str, float]
    context_tokens_used: int
    context_usage_percentage: float
    processing_time_ms: int
    llm_model_used: str
    reference_used: str  # "complete" or "focused"
    requires_human_review: bool = False
    quality_flags: List[str] = None
    
    def __post_init__(self):
        if self.quality_flags is None:
            self.quality_flags = []

@dataclass
class SystematicValidationResult:
    session_id: str
    total_units: int
    completed_units: List[UnitValidationResult]
    failed_units: List[Dict[str, Any]]
    overall_score: float
    confidence_score: float
    completeness_score: float
    processing_metrics: Dict[str, Any]
    validation_summary: Dict[str, Any]

@dataclass
class ValidationConfig:
    validation_model: str = None
    max_context_usage: float = 0.35  # Conservative 35% limit
    quality_threshold: float = 0.8
    enable_reference_optimization: bool = True
    temperature: float = 0.1  # Conservative for consistency
    max_validation_tokens: int = 2000

class IDCValidationService:
    """
    Systematic validation service following Jarvis patterns:
    - Uses Ollama models via configured OLLAMA_BASE_URL
    - Conservative context management (35% max usage)
    - Redis for progress tracking
    - PostgreSQL for result persistence
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = get_redis_client()
        self.prompt_service = get_prompt_service()
        
        # Get IDC-specific settings
        self.idc_settings = get_idc_settings()
        self.validation_config = get_validation_config(self.idc_settings)
        
        # Get Ollama configuration from settings (NO hardcoding)
        self.ollama_base_url = self.validation_config.get('model_server')
        if not self.ollama_base_url:
            # Fallback to environment variable
            self.ollama_base_url = os.environ.get("OLLAMA_BASE_URL")
            if not self.ollama_base_url:
                raise ValueError("OLLAMA_BASE_URL must be configured - no hardcoded values allowed")
        
        # Get model configuration from IDC settings
        self.default_model = self.validation_config.get('model', 'qwen3:30b-a3b-q4_K_M')
        
        # Context management from settings
        self.max_context_length = self.validation_config.get('context_length', 8192)
        self.max_context_usage = self.validation_config.get('max_context_usage', 0.35)
        self.context_safety_buffer = self.validation_config.get('safety_buffer', 1000)
        
    def _get_default_validation_model(self) -> str:
        """Get default validation model from IDC settings"""
        return self.validation_config.get('model', 'qwen3:30b-a3b-q4_K_M')
    
    def _count_tokens(self, text: str) -> int:
        """Conservative token estimation: 1 token per 3.5 characters"""
        return len(text) // 3
    
    async def _call_ollama(
        self, 
        prompt: str, 
        model: str = None, 
        max_tokens: int = 2000,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Call Ollama API using configured base URL
        """
        if not model:
            model = self.default_model
            
        url = f"{self.ollama_base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_k": 20,
                "top_p": 0.9
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
            raise Exception(f"LLM validation failed: {str(e)}")
    
    async def validate_all_units(
        self,
        extracted_units: List[ExtractedUnit],
        reference_document: str,
        session_id: str,
        validation_config: ValidationConfig
    ) -> SystematicValidationResult:
        """
        Validate every extracted unit systematically
        Core innovation: Each unit gets fresh context with full reference
        """
        start_time = time.time()
        total_units = len(extracted_units)
        completed_units = []
        failed_units = []
        
        # Initialize progress tracking
        await self._init_progress_tracking(session_id, total_units)
        
        logger.info(f"Starting systematic validation for session {session_id}: {total_units} units")
        
        # Process each unit with fresh context
        for i, unit in enumerate(extracted_units):
            try:
                unit_result = await self._validate_single_unit(
                    unit=unit,
                    reference_document=reference_document,
                    unit_position=i + 1,
                    total_units=total_units,
                    session_id=session_id,
                    validation_config=validation_config
                )
                
                completed_units.append(unit_result)
                
                # Update progress
                await self._update_progress(session_id, i + 1, total_units, unit_result)
                
                # Brief pause to prevent API overload
                await asyncio.sleep(0.5)
                
                logger.debug(f"Unit {i+1}/{total_units} validated - Score: {unit_result.validation_score:.2f}")
                
            except Exception as e:
                failed_unit = {
                    "unit_index": i,
                    "unit_content": unit.content[:200] + "..." if len(unit.content) > 200 else unit.content,
                    "error": str(e),
                    "retry_count": 0,
                    "timestamp": datetime.utcnow().isoformat()
                }
                failed_units.append(failed_unit)
                
                logger.error(f"Unit {i+1} failed validation: {str(e)}")
        
        # Calculate processing metrics
        total_processing_time = int((time.time() - start_time) * 1000)
        
        # Aggregate results
        result = self._aggregate_systematic_results(
            completed_units, 
            failed_units, 
            session_id,
            total_processing_time
        )
        
        # Save results to database
        await self._save_validation_results(session_id, result)
        
        logger.info(f"Systematic validation completed for session {session_id}: "
                   f"{len(completed_units)}/{total_units} units succeeded, "
                   f"Overall score: {result.overall_score:.2f}")
        
        return result
    
    async def _validate_single_unit(
        self,
        unit: ExtractedUnit,
        reference_document: str,
        unit_position: int,
        total_units: int,
        session_id: str,
        validation_config: ValidationConfig
    ) -> UnitValidationResult:
        """
        Validate single unit with optimal context management
        Key Innovation: Conservative context usage ensures consistent quality
        """
        unit_start_time = time.time()
        
        # Calculate context usage
        unit_tokens = self._count_tokens(unit.content)
        reference_tokens = self._count_tokens(reference_document)
        prompt_tokens = self._estimate_prompt_tokens(unit_position, total_units)
        total_tokens = unit_tokens + reference_tokens + prompt_tokens + validation_config.max_validation_tokens
        
        context_usage = total_tokens / self.max_context_length
        
        # Key Innovation: Stay within conservative context limits
        reference_to_use = reference_document
        reference_type = "complete"
        
        if context_usage > validation_config.max_context_usage:
            if validation_config.enable_reference_optimization:
                # Create focused reference summary for this unit
                reference_to_use = await self._create_focused_reference(
                    reference_document, 
                    unit.content,
                    unit.type
                )
                reference_type = "focused"
                
                # Recalculate context usage
                focused_ref_tokens = self._count_tokens(reference_to_use)
                total_tokens = unit_tokens + focused_ref_tokens + prompt_tokens + validation_config.max_validation_tokens
                context_usage = total_tokens / self.max_context_length
            else:
                logger.warning(f"Unit {unit_position} exceeds context limit but optimization disabled")
        
        # Create validation prompt
        validation_prompt = self._create_unit_validation_prompt(
            unit=unit,
            reference=reference_to_use,
            position=unit_position,
            total=total_units
        )
        
        # Perform validation
        try:
            validation_result = await self._call_ollama(
                prompt=validation_prompt,
                model=validation_config.validation_model or self.default_model,
                max_tokens=validation_config.max_validation_tokens,
                temperature=validation_config.temperature
            )
            
            # Parse validation response
            parsed_result = self._parse_validation_response(validation_result["response"])
            
            processing_time_ms = int((time.time() - unit_start_time) * 1000)
            
            return UnitValidationResult(
                unit=unit,
                validation_score=parsed_result["score"],
                confidence_score=parsed_result["confidence"],
                validation_feedback=parsed_result["feedback"],
                matched_reference_sections=parsed_result["matched_sections"],
                similarity_scores=parsed_result["similarity_scores"],
                context_tokens_used=total_tokens,
                context_usage_percentage=context_usage,
                processing_time_ms=processing_time_ms,
                llm_model_used=validation_result["model"],
                reference_used=reference_type,
                requires_human_review=parsed_result["score"] < validation_config.quality_threshold,
                quality_flags=self._identify_quality_flags(parsed_result, context_usage)
            )
            
        except Exception as e:
            logger.error(f"Unit {unit_position} validation failed: {str(e)}")
            raise
    
    def _create_unit_validation_prompt(
        self,
        unit: ExtractedUnit,
        reference: str,
        position: int,
        total: int
    ) -> str:
        """Create focused validation prompt for single unit"""
        
        # Get validation prompt from settings or prompt service
        validation_system_prompt = get_validation_system_prompt(self.idc_settings)
        
        # Try to get a more specific prompt from prompt service
        try:
            prompt_template = self.prompt_service.get_prompt(
                'idc_unit_validation',
                variables={
                    'position': position,
                    'total': total,
                    'unit_type': unit.type.upper(),
                    'unit_content': unit.content,
                    'reference': reference
                }
            )
            if prompt_template:
                prompt = prompt_template
            else:
                # Use system prompt with structured format
                prompt = f"""{validation_system_prompt}

UNIT POSITION: {position} of {total}

INPUT UNIT ({unit.type.upper()}):
{unit.content}

REFERENCE DOCUMENT:
{reference}

VALIDATION INSTRUCTIONS:
1. Compare the input unit content against ALL relevant sections of the reference document
2. Identify how well the unit aligns with reference requirements/standards
3. Note any discrepancies, missing information, or areas of concern
4. Provide specific feedback with reference to exact sections
5. Assign numerical scores based on accuracy and completeness

RESPOND WITH STRUCTURED VALIDATION REPORT:

VALIDATION_SCORE: [0.0-1.0] (Overall accuracy and completeness score)
CONFIDENCE_SCORE: [0.0-1.0] (Confidence in this validation assessment)

MATCHED_SECTIONS:
- [List specific reference sections that relate to this unit]

SIMILARITY_ASSESSMENT:
- Content Accuracy: [0.0-1.0]
- Completeness: [0.0-1.0]
- Reference Compliance: [0.0-1.0]"""
        except Exception as e:
            logger.warning(f"Could not get prompt from service: {e}, using default")
            prompt = f"""{validation_system_prompt}

UNIT POSITION: {position} of {total}

INPUT UNIT ({unit.type.upper()}):
{unit.content}

REFERENCE DOCUMENT:
{reference}

Provide detailed validation assessment.

DETAILED_FEEDBACK:
[Provide specific, actionable feedback explaining the validation score. Include:
- What aspects align well with the reference
- What discrepancies or gaps exist
- Specific recommendations for improvement
- Citations to relevant reference sections]

REQUIRES_REVIEW: [YES/NO] (Whether this unit needs human review)
"""
        return prompt
    
    def _parse_validation_response(self, response: str) -> Dict[str, Any]:
        """
        Parse structured validation response from LLM
        """
        try:
            # Extract structured information from response
            lines = response.split('\n')
            result = {
                "score": 0.8,  # Default fallback
                "confidence": 0.7,
                "feedback": response,
                "matched_sections": [],
                "similarity_scores": {
                    "content_accuracy": 0.8,
                    "completeness": 0.8,
                    "reference_compliance": 0.8
                }
            }
            
            for line in lines:
                line = line.strip()
                if line.startswith("VALIDATION_SCORE:"):
                    try:
                        score_text = line.split(":", 1)[1].strip()
                        # Extract numeric value from text
                        import re
                        score_match = re.search(r'(\d+\.?\d*)', score_text)
                        if score_match:
                            result["score"] = min(1.0, max(0.0, float(score_match.group(1))))
                    except:
                        pass
                        
                elif line.startswith("CONFIDENCE_SCORE:"):
                    try:
                        conf_text = line.split(":", 1)[1].strip()
                        import re
                        conf_match = re.search(r'(\d+\.?\d*)', conf_text)
                        if conf_match:
                            result["confidence"] = min(1.0, max(0.0, float(conf_match.group(1))))
                    except:
                        pass
                        
                elif line.startswith("- Content Accuracy:"):
                    try:
                        acc_text = line.split(":", 1)[1].strip()
                        import re
                        acc_match = re.search(r'(\d+\.?\d*)', acc_text)
                        if acc_match:
                            result["similarity_scores"]["content_accuracy"] = float(acc_match.group(1))
                    except:
                        pass
                        
                elif line.startswith("- Completeness:"):
                    try:
                        comp_text = line.split(":", 1)[1].strip()
                        import re
                        comp_match = re.search(r'(\d+\.?\d*)', comp_text)
                        if comp_match:
                            result["similarity_scores"]["completeness"] = float(comp_match.group(1))
                    except:
                        pass
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to parse validation response: {e}")
            # Return fallback result
            return {
                "score": 0.7,
                "confidence": 0.6,
                "feedback": response,
                "matched_sections": [],
                "similarity_scores": {"content_accuracy": 0.7, "completeness": 0.7, "reference_compliance": 0.7}
            }
    
    async def _create_focused_reference(
        self,
        full_reference: str,
        unit_content: str,
        unit_type: str
    ) -> str:
        """
        Create focused reference summary relevant to current unit
        """
        focus_prompt = (
            f"Create a focused summary of this reference document that's most relevant to validating the following {unit_type}:\n\n"
            f"UNIT TO VALIDATE:\n{unit_content}\n\n"
            f"FULL REFERENCE DOCUMENT:\n{full_reference}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Extract sections most relevant to the unit\n"
            f"2. Preserve exact requirements and criteria\n"
            f"3. Include related context for comprehensive validation\n"
            f"4. Keep summary under 4000 tokens\n"
            f"5. Maintain reference accuracy - do not paraphrase requirements\n\n"
            f"Return focused reference that enables thorough validation:"
        )
        
        try:
            result = await self._call_ollama(
                prompt=focus_prompt,
                max_tokens=4000
            )
            
            focused_reference = result["response"].strip()
            
            # Ensure minimum reference content
            if len(focused_reference) < 100:
                logger.warning("Focused reference too short, using original")
                return full_reference[:4000]
            
            return focused_reference
            
        except Exception as e:
            logger.error(f"Failed to create focused reference: {e}")
            # Fallback: truncate original reference
            return full_reference[:4000]
    
    def _estimate_prompt_tokens(self, position: int, total: int) -> int:
        """Estimate tokens needed for validation prompt structure"""
        # Conservative estimate for prompt template
        return 800
    
    def _identify_quality_flags(self, parsed_result: Dict[str, Any], context_usage: float) -> List[str]:
        """Identify quality flags for validation result"""
        flags = []
        
        if parsed_result["confidence"] < 0.7:
            flags.append("low_confidence")
            
        if context_usage > 0.8:
            flags.append("high_context_usage")
            
        if parsed_result["score"] < 0.6:
            flags.append("low_validation_score")
            
        if len(parsed_result["matched_sections"]) == 0:
            flags.append("no_reference_matches")
        
        return flags
    
    def _aggregate_systematic_results(
        self,
        completed_units: List[UnitValidationResult],
        failed_units: List[Dict[str, Any]],
        session_id: str,
        total_processing_time: int
    ) -> SystematicValidationResult:
        """Aggregate individual unit results into systematic validation result"""
        
        if not completed_units:
            return SystematicValidationResult(
                session_id=session_id,
                total_units=len(failed_units),
                completed_units=[],
                failed_units=failed_units,
                overall_score=0.0,
                confidence_score=0.0,
                completeness_score=0.0,
                processing_metrics={},
                validation_summary={}
            )
        
        # Calculate aggregate scores
        validation_scores = [unit.validation_score for unit in completed_units]
        confidence_scores = [unit.confidence_score for unit in completed_units]
        
        overall_score = sum(validation_scores) / len(validation_scores)
        confidence_score = sum(confidence_scores) / len(confidence_scores)
        
        # Calculate completeness (% of units successfully processed)
        total_attempted = len(completed_units) + len(failed_units)
        completeness_score = len(completed_units) / total_attempted if total_attempted > 0 else 0.0
        
        # Processing metrics
        processing_times = [unit.processing_time_ms for unit in completed_units]
        context_usages = [unit.context_usage_percentage for unit in completed_units]
        
        processing_metrics = {
            "total_processing_time_ms": total_processing_time,
            "average_unit_processing_time_ms": sum(processing_times) / len(processing_times),
            "average_context_usage": sum(context_usages) / len(context_usages),
            "max_context_usage": max(context_usages) if context_usages else 0.0,
            "units_requiring_review": sum(1 for unit in completed_units if unit.requires_human_review),
            "reference_optimization_usage": sum(1 for unit in completed_units if unit.reference_used == "focused")
        }
        
        # Validation summary
        validation_summary = {
            "high_quality_units": sum(1 for unit in completed_units if unit.validation_score >= 0.8),
            "medium_quality_units": sum(1 for unit in completed_units if 0.6 <= unit.validation_score < 0.8),
            "low_quality_units": sum(1 for unit in completed_units if unit.validation_score < 0.6),
            "average_validation_score": overall_score,
            "score_distribution": {
                "min_score": min(validation_scores),
                "max_score": max(validation_scores),
                "std_deviation": self._calculate_std_dev(validation_scores)
            }
        }
        
        return SystematicValidationResult(
            session_id=session_id,
            total_units=total_attempted,
            completed_units=completed_units,
            failed_units=failed_units,
            overall_score=overall_score,
            confidence_score=confidence_score,
            completeness_score=completeness_score,
            processing_metrics=processing_metrics,
            validation_summary=validation_summary
        )
    
    def _calculate_std_dev(self, scores: List[float]) -> float:
        """Calculate standard deviation of scores"""
        if len(scores) <= 1:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        return variance ** 0.5
    
    async def _init_progress_tracking(self, session_id: str, total_units: int):
        """Initialize progress tracking in Redis"""
        if self.redis_client:
            try:
                progress_data = {
                    "session_id": session_id,
                    "total_units": total_units,
                    "completed_units": 0,
                    "current_unit_index": 0,
                    "progress_percentage": 0.0,
                    "status": "initializing",
                    "started_at": datetime.utcnow().isoformat()
                }
                
                cache_key = f"idc_validation_progress:{session_id}"
                # Use configurable TTL from timeout settings
                validation_cache_ttl = get_redis_cache_ttl("validation_cache_ttl", 7200)
                self.redis_client.setex(cache_key, validation_cache_ttl, json.dumps(progress_data))
                
            except Exception as e:
                logger.warning(f"Failed to initialize progress tracking: {e}")
    
    async def _update_progress(
        self, 
        session_id: str, 
        completed: int, 
        total: int, 
        latest_result: UnitValidationResult
    ):
        """Update validation progress in Redis"""
        if self.redis_client:
            try:
                progress_data = {
                    "session_id": session_id,
                    "total_units": total,
                    "completed_units": completed,
                    "current_unit_index": completed,
                    "progress_percentage": (completed / total) * 100,
                    "status": "processing",
                    "latest_unit_score": latest_result.validation_score,
                    "latest_processing_time": latest_result.processing_time_ms,
                    "updated_at": datetime.utcnow().isoformat()
                }
                
                cache_key = f"idc_validation_progress:{session_id}"
                self.redis_client.setex(cache_key, 7200, json.dumps(progress_data))
                
            except Exception as e:
                logger.warning(f"Failed to update progress tracking: {e}")
    
    async def get_validation_progress(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get validation progress from Redis"""
        if self.redis_client:
            try:
                cache_key = f"idc_validation_progress:{session_id}"
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Failed to get validation progress: {e}")
        return None
    
    async def _save_validation_results(self, session_id: str, result: SystematicValidationResult):
        """Save validation results to database"""
        try:
            with get_db_session() as db:
                # Update validation session
                session = db.query(IDCValidationSession).filter(
                    IDCValidationSession.session_id == session_id
                ).first()
                
                if session:
                    session.units_processed = len(result.completed_units)
                    session.units_failed = len(result.failed_units)
                    session.overall_score = result.overall_score
                    session.confidence_score = result.confidence_score
                    session.completeness_score = result.completeness_score
                    session.validation_results = asdict(result)
                    session.failed_units = result.failed_units
                    session.status = "completed"
                    session.processing_end_time = datetime.utcnow()
                    session.total_processing_time_ms = result.processing_metrics.get("total_processing_time_ms")
                    session.average_context_usage = result.processing_metrics.get("average_context_usage")
                    session.max_context_usage_recorded = result.processing_metrics.get("max_context_usage")
                    
                    # Save individual unit results
                    for unit_result in result.completed_units:
                        db_unit_result = IDCUnitValidationResult(
                            session_id=session_id,
                            unit_index=unit_result.unit.index,
                            unit_type=unit_result.unit.type,
                            unit_content=unit_result.unit.content,
                            unit_metadata=unit_result.unit.metadata,
                            validation_score=unit_result.validation_score,
                            confidence_score=unit_result.confidence_score,
                            validation_feedback=unit_result.validation_feedback,
                            matched_reference_sections=unit_result.matched_reference_sections,
                            similarity_scores=unit_result.similarity_scores,
                            context_tokens_used=unit_result.context_tokens_used,
                            context_usage_percentage=unit_result.context_usage_percentage,
                            processing_time_ms=unit_result.processing_time_ms,
                            llm_model_used=unit_result.llm_model_used,
                            requires_human_review=unit_result.requires_human_review,
                            quality_flags=unit_result.quality_flags
                        )
                        db.add(db_unit_result)
                    
                    db.commit()
                    logger.info(f"Validation results saved for session {session_id}")
                    
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")
            raise