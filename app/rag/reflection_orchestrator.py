"""
Reflection Orchestrator for Self-Reflective RAG

This module coordinates the entire self-reflection pipeline,
managing the flow between evaluation, refinement, and improvement.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from app.rag.response_quality_evaluator import (
    ResponseQualityEvaluator, ResponseEvaluation
)
from app.rag.retrieval_quality_monitor import (
    RetrievalQualityMonitor, RetrievalAssessment
)
from app.rag.iterative_refinement_engine import (
    IterativeRefinementEngine, RefinementResult
)

logger = logging.getLogger(__name__)


class ReflectionMode(Enum):
    """Modes of reflection"""
    FAST = "fast"  # Single evaluation and refinement
    BALANCED = "balanced"  # Multiple iterations with quality checks
    THOROUGH = "thorough"  # Exhaustive refinement until convergence


@dataclass
class ReflectionContext:
    """Context for the reflection process"""
    query: str
    initial_response: str
    retrieved_documents: List[Dict]
    conversation_history: Optional[List[Dict]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    
@dataclass
class ReflectionMetrics:
    """Metrics collected during reflection"""
    total_time_ms: float
    evaluation_time_ms: float
    refinement_time_ms: float
    iterations_performed: int
    quality_improvement: float
    strategies_attempted: List[str]
    final_confidence: float


@dataclass
class ReflectionResult:
    """Complete result of the reflection process"""
    final_response: str
    quality_score: float
    improvements_made: List[str]
    reflection_metrics: ReflectionMetrics
    evaluation_results: ResponseEvaluation
    retrieval_assessment: RetrievalAssessment
    refinement_details: Optional[RefinementResult]
    success: bool
    metadata: Dict[str, Any]


class ReflectionOrchestrator:
    """
    Orchestrates the self-reflection process for RAG,
    coordinating evaluation, monitoring, and refinement.
    """
    
    def __init__(
        self,
        response_evaluator: Optional[ResponseQualityEvaluator] = None,
        retrieval_monitor: Optional[RetrievalQualityMonitor] = None,
        refinement_engine: Optional[IterativeRefinementEngine] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the orchestrator
        
        Args:
            response_evaluator: Evaluator for response quality
            retrieval_monitor: Monitor for retrieval quality
            refinement_engine: Engine for iterative refinement
            config: Configuration dictionary
        """
        self.response_evaluator = response_evaluator or ResponseQualityEvaluator()
        self.retrieval_monitor = retrieval_monitor or RetrievalQualityMonitor()
        self.refinement_engine = refinement_engine
        self.config = config or self._get_default_config()
        
        # Callbacks for extensibility
        self.pre_evaluation_hooks: List[Callable] = []
        self.post_refinement_hooks: List[Callable] = []
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "reflection_modes": {
                ReflectionMode.FAST: {
                    "max_iterations": 1,
                    "quality_threshold": 0.7,
                    "enable_parallel": False
                },
                ReflectionMode.BALANCED: {
                    "max_iterations": 2,
                    "quality_threshold": 0.8,
                    "enable_parallel": True
                },
                ReflectionMode.THOROUGH: {
                    "max_iterations": 3,
                    "quality_threshold": 0.9,
                    "enable_parallel": True
                }
            },
            "default_mode": ReflectionMode.BALANCED,
            "enable_caching": True,
            "timeout_seconds": 30,
            "min_quality_for_refinement": 0.5,
            "log_metrics": True
        }
    
    async def reflect_and_improve(
        self,
        context: ReflectionContext,
        mode: Optional[ReflectionMode] = None,
        force_refinement: bool = False
    ) -> ReflectionResult:
        """
        Main entry point for self-reflection process
        
        Args:
            context: Reflection context with query, response, and documents
            mode: Reflection mode (fast/balanced/thorough)
            force_refinement: Force refinement even if quality is acceptable
            
        Returns:
            ReflectionResult with improved response and metrics
        """
        start_time = datetime.now()
        mode = mode or self.config["default_mode"]
        mode_config = self.config["reflection_modes"][mode]
        
        try:
            # Run pre-evaluation hooks
            await self._run_hooks(self.pre_evaluation_hooks, context)
            
            # Step 1: Evaluate initial response quality
            eval_start = datetime.now()
            evaluation = await self._evaluate_response(context)
            eval_time = (datetime.now() - eval_start).total_seconds() * 1000
            
            # Step 2: Assess retrieval quality
            retrieval_assessment = self._assess_retrieval(
                context.query, context.retrieved_documents
            )
            
            # Step 3: Determine if refinement is needed
            needs_refinement = self._should_refine(
                evaluation, retrieval_assessment, mode_config, force_refinement
            )
            
            # Step 4: Perform refinement if needed
            refinement_result = None
            refinement_time = 0
            
            if needs_refinement and self.refinement_engine:
                refine_start = datetime.now()
                refinement_result = await self._perform_refinement(
                    context, evaluation, retrieval_assessment, mode_config
                )
                refinement_time = (datetime.now() - refine_start).total_seconds() * 1000
                
                # Re-evaluate after refinement
                if refinement_result and refinement_result.final_response != context.initial_response:
                    context_copy = ReflectionContext(
                        query=context.query,
                        initial_response=refinement_result.final_response,
                        retrieved_documents=context.retrieved_documents,
                        conversation_history=context.conversation_history,
                        metadata=context.metadata
                    )
                    evaluation = await self._evaluate_response(context_copy)
            
            # Run post-refinement hooks
            await self._run_hooks(self.post_refinement_hooks, context, refinement_result)
            
            # Calculate metrics
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            metrics = ReflectionMetrics(
                total_time_ms=total_time,
                evaluation_time_ms=eval_time,
                refinement_time_ms=refinement_time,
                iterations_performed=refinement_result.total_iterations if refinement_result else 0,
                quality_improvement=refinement_result.quality_improvement if refinement_result else 0,
                strategies_attempted=[s.value for s in refinement_result.strategies_used] if refinement_result else [],
                final_confidence=evaluation.confidence_level
            )
            
            # Log metrics if enabled
            if self.config["log_metrics"]:
                self._log_metrics(metrics)
            
            # Determine final response
            final_response = (
                refinement_result.final_response if refinement_result
                else context.initial_response
            )
            
            # Compile improvements made
            improvements = self._compile_improvements(
                evaluation, retrieval_assessment, refinement_result
            )
            
            return ReflectionResult(
                final_response=final_response,
                quality_score=evaluation.overall_score,
                improvements_made=improvements,
                reflection_metrics=metrics,
                evaluation_results=evaluation,
                retrieval_assessment=retrieval_assessment,
                refinement_details=refinement_result,
                success=evaluation.overall_score >= mode_config["quality_threshold"],
                metadata={
                    "mode": mode.value,
                    "refinement_performed": needs_refinement,
                    "convergence_achieved": refinement_result.converged if refinement_result else True
                }
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Reflection timeout after {self.config['timeout_seconds']}s")
            return self._create_timeout_result(context, start_time)
        except Exception as e:
            logger.error(f"Reflection error: {str(e)}", exc_info=True)
            return self._create_error_result(context, start_time, str(e))
    
    async def _evaluate_response(self, context: ReflectionContext) -> ResponseEvaluation:
        """Evaluate response quality"""
        return await self.response_evaluator.evaluate_response(
            query=context.query,
            response=context.initial_response,
            context_documents=context.retrieved_documents,
            conversation_history=context.conversation_history
        )
    
    def _assess_retrieval(
        self, query: str, documents: List[Dict]
    ) -> RetrievalAssessment:
        """Assess retrieval quality"""
        return self.retrieval_monitor.assess_retrieval_quality(
            query=query,
            retrieved_documents=documents
        )
    
    def _should_refine(
        self,
        evaluation: ResponseEvaluation,
        retrieval: RetrievalAssessment,
        mode_config: Dict,
        force: bool
    ) -> bool:
        """Determine if refinement is needed"""
        if force:
            return True
        
        # Check if quality is below threshold
        if evaluation.overall_score < mode_config["quality_threshold"]:
            return True
        
        # Check if minimum quality for any refinement
        if evaluation.overall_score < self.config["min_quality_for_refinement"]:
            return False
        
        # Check specific issues
        if evaluation.needs_refinement or retrieval.needs_reretrieval:
            return True
        
        return False
    
    async def _perform_refinement(
        self,
        context: ReflectionContext,
        evaluation: ResponseEvaluation,
        retrieval: RetrievalAssessment,
        mode_config: Dict
    ) -> RefinementResult:
        """Perform iterative refinement"""
        # Configure refinement engine
        self.refinement_engine.config["max_iterations"] = mode_config["max_iterations"]
        self.refinement_engine.config["quality_threshold"] = mode_config["quality_threshold"]
        
        # Perform refinement
        if mode_config.get("enable_parallel", False):
            # Use parallel refinement for faster results
            result = await self.refinement_engine.parallel_refinement(
                query=context.query,
                response=context.initial_response,
                evaluation=evaluation,
                retrieval=retrieval
            )
        else:
            # Use sequential refinement
            result = await self.refinement_engine.refine_iteratively(
                initial_query=context.query,
                initial_response=context.initial_response,
                initial_evaluation=evaluation,
                initial_retrieval=retrieval,
                conversation_history=context.conversation_history
            )
        
        return result
    
    def _compile_improvements(
        self,
        evaluation: ResponseEvaluation,
        retrieval: RetrievalAssessment,
        refinement: Optional[RefinementResult]
    ) -> List[str]:
        """Compile list of improvements made"""
        improvements = []
        
        # From evaluation
        if evaluation.overall_score > 0.8:
            improvements.append(f"Achieved high quality score: {evaluation.overall_score:.2f}")
        
        # From retrieval
        if retrieval.overall_quality > 0.8:
            improvements.append(f"High retrieval quality: {retrieval.overall_quality:.2f}")
        
        # From refinement
        if refinement:
            if refinement.quality_improvement > 0:
                improvements.append(
                    f"Improved quality by {refinement.quality_improvement:.2f}"
                )
            
            for step in refinement.refinement_steps:
                improvements.extend(step.improvements)
        
        return list(set(improvements))  # Remove duplicates
    
    def _log_metrics(self, metrics: ReflectionMetrics):
        """Log reflection metrics"""
        logger.info(
            f"Reflection completed: "
            f"time={metrics.total_time_ms:.0f}ms, "
            f"iterations={metrics.iterations_performed}, "
            f"improvement={metrics.quality_improvement:.2f}, "
            f"confidence={metrics.final_confidence:.2f}"
        )
    
    async def _run_hooks(
        self, hooks: List[Callable], *args, **kwargs
    ):
        """Run registered hooks"""
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(*args, **kwargs)
                else:
                    hook(*args, **kwargs)
            except Exception as e:
                logger.error(f"Hook error: {str(e)}")
    
    def _create_timeout_result(
        self, context: ReflectionContext, start_time: datetime
    ) -> ReflectionResult:
        """Create result for timeout scenario"""
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ReflectionResult(
            final_response=context.initial_response,
            quality_score=0.0,
            improvements_made=["Reflection timed out"],
            reflection_metrics=ReflectionMetrics(
                total_time_ms=total_time,
                evaluation_time_ms=0,
                refinement_time_ms=0,
                iterations_performed=0,
                quality_improvement=0,
                strategies_attempted=[],
                final_confidence=0
            ),
            evaluation_results=None,
            retrieval_assessment=None,
            refinement_details=None,
            success=False,
            metadata={"error": "timeout"}
        )
    
    def _create_error_result(
        self, context: ReflectionContext, start_time: datetime, error: str
    ) -> ReflectionResult:
        """Create result for error scenario"""
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ReflectionResult(
            final_response=context.initial_response,
            quality_score=0.0,
            improvements_made=[f"Reflection error: {error}"],
            reflection_metrics=ReflectionMetrics(
                total_time_ms=total_time,
                evaluation_time_ms=0,
                refinement_time_ms=0,
                iterations_performed=0,
                quality_improvement=0,
                strategies_attempted=[],
                final_confidence=0
            ),
            evaluation_results=None,
            retrieval_assessment=None,
            refinement_details=None,
            success=False,
            metadata={"error": error}
        )
    
    def register_pre_evaluation_hook(self, hook: Callable):
        """Register a pre-evaluation hook"""
        self.pre_evaluation_hooks.append(hook)
    
    def register_post_refinement_hook(self, hook: Callable):
        """Register a post-refinement hook"""
        self.post_refinement_hooks.append(hook)
    
    async def batch_reflect(
        self, contexts: List[ReflectionContext], mode: Optional[ReflectionMode] = None
    ) -> List[ReflectionResult]:
        """
        Perform reflection on multiple contexts in parallel
        
        Args:
            contexts: List of reflection contexts
            mode: Reflection mode to use
            
        Returns:
            List of reflection results
        """
        tasks = [
            self.reflect_and_improve(context, mode)
            for context in contexts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch reflection error for context {i}: {str(result)}")
                processed_results.append(
                    self._create_error_result(
                        contexts[i], datetime.now(), str(result)
                    )
                )
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_reflection_stats(self, results: List[ReflectionResult]) -> Dict[str, Any]:
        """
        Calculate statistics from multiple reflection results
        
        Args:
            results: List of reflection results
            
        Returns:
            Dictionary of statistics
        """
        if not results:
            return {}
        
        successful = [r for r in results if r.success]
        
        return {
            "total_reflections": len(results),
            "successful_reflections": len(successful),
            "success_rate": len(successful) / len(results),
            "average_quality": sum(r.quality_score for r in results) / len(results),
            "average_improvement": sum(
                r.reflection_metrics.quality_improvement for r in results
            ) / len(results),
            "average_time_ms": sum(
                r.reflection_metrics.total_time_ms for r in results
            ) / len(results),
            "total_iterations": sum(
                r.reflection_metrics.iterations_performed for r in results
            ),
            "strategies_used": list(set(
                strategy
                for r in results
                for strategy in r.reflection_metrics.strategies_attempted
            ))
        }