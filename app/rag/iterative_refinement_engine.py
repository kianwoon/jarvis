"""
Iterative Refinement Engine for Self-Reflective RAG

This module handles query reformulation, multi-round retrieval,
and iterative answer improvement based on reflection feedback.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from app.rag.response_quality_evaluator import ResponseEvaluation, QualityDimension
from app.rag.retrieval_quality_monitor import RetrievalAssessment

logger = logging.getLogger(__name__)


class RefinementStrategy(Enum):
    """Strategies for refinement"""
    QUERY_EXPANSION = "query_expansion"
    QUERY_DECOMPOSITION = "query_decomposition"
    CONTEXT_ENRICHMENT = "context_enrichment"
    ANSWER_AUGMENTATION = "answer_augmentation"
    FOCUSED_RETRIEVAL = "focused_retrieval"
    ALTERNATIVE_PHRASING = "alternative_phrasing"


@dataclass
class RefinementStep:
    """Single step in the refinement process"""
    iteration: int
    strategy: RefinementStrategy
    original_query: str
    refined_query: str
    retrieved_documents: List[Dict]
    generated_response: str
    quality_score: float
    improvements: List[str]
    metadata: Dict[str, Any]


@dataclass
class RefinementResult:
    """Final result of the refinement process"""
    final_response: str
    total_iterations: int
    refinement_steps: List[RefinementStep]
    quality_improvement: float
    strategies_used: List[RefinementStrategy]
    final_quality_score: float
    converged: bool
    metadata: Dict[str, Any]


class IterativeRefinementEngine:
    """
    Handles iterative refinement of queries and responses
    based on quality feedback and reflection results.
    """
    
    def __init__(self, retriever, generator, llm_client=None, config: Optional[Dict] = None):
        """
        Initialize the refinement engine
        
        Args:
            retriever: Document retrieval interface
            generator: Response generation interface
            llm_client: LLM client for refinement tasks
            config: Configuration dictionary
        """
        self.retriever = retriever
        self.generator = generator
        self.llm_client = llm_client
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "max_iterations": 3,
            "quality_threshold": 0.85,
            "min_improvement_threshold": 0.05,
            "strategy_selection": "adaptive",  # or "sequential", "parallel"
            "enable_caching": True,
            "refinement_prompts": {
                "query_expansion": "Expand this query with related terms: {query}",
                "query_decomposition": "Break down this complex query into simpler parts: {query}",
                "context_enrichment": "What additional context would help answer: {query}",
                "answer_augmentation": "What details are missing from this answer: {response}",
                "focused_retrieval": "Create a more focused search query for: {missing_aspect}",
                "alternative_phrasing": "Rephrase this query differently: {query}"
            }
        }
    
    async def refine_iteratively(
        self,
        initial_query: str,
        initial_response: str,
        initial_evaluation: ResponseEvaluation,
        initial_retrieval: RetrievalAssessment,
        conversation_history: Optional[List[Dict]] = None
    ) -> RefinementResult:
        """
        Iteratively refine the query and response
        
        Args:
            initial_query: Original user query
            initial_response: Initial generated response
            initial_evaluation: Quality evaluation of initial response
            initial_retrieval: Assessment of initial retrieval
            conversation_history: Previous conversation context
            
        Returns:
            RefinementResult with improved response and process details
        """
        refinement_steps = []
        current_query = initial_query
        current_response = initial_response
        current_quality = initial_evaluation.overall_score
        
        # Track best response
        best_response = initial_response
        best_quality = current_quality
        
        for iteration in range(self.config["max_iterations"]):
            # Select refinement strategy
            strategy = self._select_refinement_strategy(
                initial_evaluation, initial_retrieval, iteration
            )
            
            # Apply refinement strategy
            refined_query, strategy_metadata = await self._apply_strategy(
                strategy, current_query, current_response,
                initial_evaluation, initial_retrieval
            )
            
            # Retrieve with refined query
            retrieved_docs = await self._retrieve_documents(refined_query)
            
            # Generate new response
            new_response = await self._generate_response(
                refined_query, retrieved_docs, conversation_history
            )
            
            # Evaluate new response
            new_evaluation = await self._evaluate_response(
                initial_query, new_response, retrieved_docs
            )
            
            # Calculate improvement
            improvement = new_evaluation.overall_score - current_quality
            
            # Record refinement step
            step = RefinementStep(
                iteration=iteration + 1,
                strategy=strategy,
                original_query=current_query,
                refined_query=refined_query,
                retrieved_documents=retrieved_docs,
                generated_response=new_response,
                quality_score=new_evaluation.overall_score,
                improvements=self._identify_improvements(
                    initial_evaluation, new_evaluation
                ),
                metadata={
                    "strategy_metadata": strategy_metadata,
                    "improvement": improvement
                }
            )
            refinement_steps.append(step)
            
            # Update best response if improved
            if new_evaluation.overall_score > best_quality:
                best_response = new_response
                best_quality = new_evaluation.overall_score
            
            # Check convergence criteria
            if self._has_converged(
                new_evaluation.overall_score, improvement, iteration
            ):
                break
            
            # Update for next iteration
            current_query = refined_query
            current_response = new_response
            current_quality = new_evaluation.overall_score
            initial_evaluation = new_evaluation
        
        # Calculate overall improvement
        quality_improvement = best_quality - initial_evaluation.overall_score
        
        return RefinementResult(
            final_response=best_response,
            total_iterations=len(refinement_steps),
            refinement_steps=refinement_steps,
            quality_improvement=quality_improvement,
            strategies_used=list(set(step.strategy for step in refinement_steps)),
            final_quality_score=best_quality,
            converged=best_quality >= self.config["quality_threshold"],
            metadata={
                "initial_quality": initial_evaluation.overall_score,
                "average_improvement_per_iteration": quality_improvement / max(len(refinement_steps), 1)
            }
        )
    
    def _select_refinement_strategy(
        self,
        evaluation: ResponseEvaluation,
        retrieval: RetrievalAssessment,
        iteration: int
    ) -> RefinementStrategy:
        """Select appropriate refinement strategy based on issues"""
        
        # Analyze main issues
        weakest_dimension = min(
            evaluation.dimension_scores.items(),
            key=lambda x: x[1].score
        )[0]
        
        # Map issues to strategies
        if retrieval.needs_reretrieval:
            if evaluation.missing_aspects:
                return RefinementStrategy.FOCUSED_RETRIEVAL
            else:
                return RefinementStrategy.QUERY_EXPANSION
        
        if weakest_dimension == QualityDimension.COMPLETENESS:
            if len(evaluation.missing_aspects) > 2:
                return RefinementStrategy.QUERY_DECOMPOSITION
            else:
                return RefinementStrategy.ANSWER_AUGMENTATION
        
        if weakest_dimension == QualityDimension.RELEVANCE:
            return RefinementStrategy.ALTERNATIVE_PHRASING
        
        if weakest_dimension == QualityDimension.SPECIFICITY:
            return RefinementStrategy.CONTEXT_ENRICHMENT
        
        # Default strategy based on iteration
        strategies = [
            RefinementStrategy.QUERY_EXPANSION,
            RefinementStrategy.CONTEXT_ENRICHMENT,
            RefinementStrategy.ANSWER_AUGMENTATION
        ]
        return strategies[iteration % len(strategies)]
    
    async def _apply_strategy(
        self,
        strategy: RefinementStrategy,
        current_query: str,
        current_response: str,
        evaluation: ResponseEvaluation,
        retrieval: RetrievalAssessment
    ) -> Tuple[str, Dict]:
        """Apply a specific refinement strategy"""
        
        if strategy == RefinementStrategy.QUERY_EXPANSION:
            return await self._expand_query(current_query, retrieval.missing_information)
        
        elif strategy == RefinementStrategy.QUERY_DECOMPOSITION:
            return await self._decompose_query(current_query, evaluation.missing_aspects)
        
        elif strategy == RefinementStrategy.CONTEXT_ENRICHMENT:
            return await self._enrich_context(current_query, current_response)
        
        elif strategy == RefinementStrategy.ANSWER_AUGMENTATION:
            return await self._augment_answer_query(
                current_query, current_response, evaluation.missing_aspects
            )
        
        elif strategy == RefinementStrategy.FOCUSED_RETRIEVAL:
            return await self._create_focused_query(
                current_query, evaluation.missing_aspects[0] if evaluation.missing_aspects else ""
            )
        
        elif strategy == RefinementStrategy.ALTERNATIVE_PHRASING:
            return await self._rephrase_query(current_query)
        
        else:
            # Fallback to original query
            return current_query, {"strategy": "fallback"}
    
    async def _expand_query(
        self, query: str, missing_info: List[str]
    ) -> Tuple[str, Dict]:
        """Expand query with related terms and missing information"""
        # Add missing information to query
        if missing_info:
            expanded = f"{query} including {', '.join(missing_info[:2])}"
        else:
            # Use LLM to suggest expansions
            if self.llm_client:
                prompt = self.config["refinement_prompts"]["query_expansion"].format(query=query)
                expansion = await self._llm_complete(prompt)
                expanded = f"{query} {expansion}"
            else:
                expanded = query
        
        return expanded, {
            "original_length": len(query.split()),
            "expanded_length": len(expanded.split()),
            "added_terms": missing_info[:2] if missing_info else []
        }
    
    async def _decompose_query(
        self, query: str, missing_aspects: List[str]
    ) -> Tuple[str, Dict]:
        """Break down complex query into focused parts"""
        # Focus on most important missing aspect
        if missing_aspects:
            # Create focused query for first missing aspect
            focused = f"{missing_aspects[0]} in context of {query}"
        else:
            # Use LLM to decompose
            if self.llm_client:
                prompt = self.config["refinement_prompts"]["query_decomposition"].format(query=query)
                decomposed = await self._llm_complete(prompt)
                # Take first part
                parts = decomposed.split('\n')
                focused = parts[0] if parts else query
            else:
                # Simple decomposition
                words = query.split()
                focused = ' '.join(words[:len(words)//2])
        
        return focused, {
            "original_query": query,
            "focused_on": missing_aspects[0] if missing_aspects else "first_half"
        }
    
    async def _enrich_context(
        self, query: str, response: str
    ) -> Tuple[str, Dict]:
        """Add context to improve retrieval"""
        # Extract key entities from response
        entities = self._extract_key_entities(response)
        
        if entities:
            enriched = f"{query} specifically about {', '.join(entities[:2])}"
        else:
            # Add general context
            enriched = f"detailed information about {query}"
        
        return enriched, {
            "entities_added": entities[:2] if entities else [],
            "strategy": "entity_enrichment"
        }
    
    async def _augment_answer_query(
        self, query: str, response: str, missing_aspects: List[str]
    ) -> Tuple[str, Dict]:
        """Create query to augment existing answer"""
        if missing_aspects:
            augmented = f"{query} focusing on {missing_aspects[0]}"
        else:
            # Ask for more details
            augmented = f"more specific details about {query}"
        
        return augmented, {
            "targeting": missing_aspects[0] if missing_aspects else "additional_details"
        }
    
    async def _create_focused_query(
        self, query: str, missing_aspect: str
    ) -> Tuple[str, Dict]:
        """Create highly focused query for specific aspect"""
        if missing_aspect:
            focused = f"{missing_aspect} {query}"
        else:
            # Focus on core terms
            core_terms = self._extract_core_terms(query)
            focused = ' '.join(core_terms)
        
        return focused, {
            "focus": missing_aspect or "core_terms"
        }
    
    async def _rephrase_query(self, query: str) -> Tuple[str, Dict]:
        """Rephrase query for better relevance"""
        if self.llm_client:
            prompt = self.config["refinement_prompts"]["alternative_phrasing"].format(query=query)
            rephrased = await self._llm_complete(prompt)
        else:
            # Simple rephrasing - reorder words
            words = query.split()
            if len(words) > 3:
                # Move last important word to front
                rephrased = f"{words[-1]} {' '.join(words[:-1])}"
            else:
                rephrased = query
        
        return rephrased, {
            "original": query,
            "strategy": "reordering"
        }
    
    async def _retrieve_documents(self, query: str) -> List[Dict]:
        """Retrieve documents with refined query"""
        # This is a placeholder - integrate with actual retriever
        if hasattr(self.retriever, 'retrieve'):
            return await self.retriever.retrieve(query)
        else:
            # Fallback for testing
            return [
                {"id": "1", "content": f"Document for {query}", "score": 0.8}
            ]
    
    async def _generate_response(
        self, query: str, documents: List[Dict], history: Optional[List[Dict]]
    ) -> str:
        """Generate response with retrieved documents"""
        # This is a placeholder - integrate with actual generator
        if hasattr(self.generator, 'generate'):
            return await self.generator.generate(query, documents, history)
        else:
            # Fallback for testing
            return f"Generated response for {query} using {len(documents)} documents"
    
    async def _evaluate_response(
        self, original_query: str, response: str, documents: List[Dict]
    ) -> ResponseEvaluation:
        """Evaluate response quality"""
        # This is a placeholder - integrate with ResponseQualityEvaluator
        from app.rag.response_quality_evaluator import ResponseQualityEvaluator
        evaluator = ResponseQualityEvaluator()
        return await evaluator.evaluate_response(
            original_query, response, documents
        )
    
    async def _llm_complete(self, prompt: str) -> str:
        """Complete prompt using LLM"""
        if self.llm_client:
            # Actual LLM call would go here
            return f"LLM response to: {prompt}"
        else:
            return "refinement suggestion"
    
    def _has_converged(
        self, current_quality: float, improvement: float, iteration: int
    ) -> bool:
        """Check if refinement has converged"""
        # Quality threshold reached
        if current_quality >= self.config["quality_threshold"]:
            return True
        
        # No significant improvement
        if abs(improvement) < self.config["min_improvement_threshold"]:
            return True
        
        # Max iterations reached
        if iteration >= self.config["max_iterations"] - 1:
            return True
        
        return False
    
    def _identify_improvements(
        self, old_eval: ResponseEvaluation, new_eval: ResponseEvaluation
    ) -> List[str]:
        """Identify specific improvements between evaluations"""
        improvements = []
        
        for dimension in QualityDimension:
            if dimension in old_eval.dimension_scores and dimension in new_eval.dimension_scores:
                old_score = old_eval.dimension_scores[dimension].score
                new_score = new_eval.dimension_scores[dimension].score
                
                if new_score > old_score + 0.05:
                    improvements.append(
                        f"Improved {dimension.value}: {old_score:.2f} → {new_score:.2f}"
                    )
        
        # Check missing aspects
        old_missing = set(old_eval.missing_aspects)
        new_missing = set(new_eval.missing_aspects)
        addressed = old_missing - new_missing
        
        if addressed:
            improvements.append(f"Addressed aspects: {', '.join(addressed)}")
        
        return improvements
    
    def _extract_key_entities(self, text: str) -> List[str]:
        """Extract key entities from text (simplified)"""
        # In production, use NER
        # For now, extract capitalized words
        words = text.split()
        entities = [
            word for word in words
            if word[0].isupper() and len(word) > 2
        ]
        return list(set(entities))[:5]
    
    def _extract_core_terms(self, query: str) -> List[str]:
        """Extract core terms from query"""
        stop_words = {'the', 'is', 'are', 'what', 'how', 'when', 'where', 'why'}
        words = query.lower().split()
        core = [w for w in words if w not in stop_words and len(w) > 2]
        return core
    
    async def parallel_refinement(
        self,
        query: str,
        response: str,
        evaluation: ResponseEvaluation,
        retrieval: RetrievalAssessment,
        strategies: Optional[List[RefinementStrategy]] = None
    ) -> RefinementResult:
        """
        Apply multiple refinement strategies in parallel
        and select the best result
        """
        if strategies is None:
            strategies = [
                RefinementStrategy.QUERY_EXPANSION,
                RefinementStrategy.CONTEXT_ENRICHMENT,
                RefinementStrategy.ALTERNATIVE_PHRASING
            ]
        
        # Run strategies in parallel
        tasks = []
        for strategy in strategies:
            task = self._apply_single_refinement(
                strategy, query, response, evaluation, retrieval
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        
        # Select best result
        best_result = max(results, key=lambda r: r.final_quality_score)
        
        return best_result
    
    async def _apply_single_refinement(
        self,
        strategy: RefinementStrategy,
        query: str,
        response: str,
        evaluation: ResponseEvaluation,
        retrieval: RetrievalAssessment
    ) -> RefinementResult:
        """Apply a single refinement strategy"""
        # Apply strategy
        refined_query, metadata = await self._apply_strategy(
            strategy, query, response, evaluation, retrieval
        )
        
        # Retrieve and generate
        documents = await self._retrieve_documents(refined_query)
        new_response = await self._generate_response(refined_query, documents, None)
        
        # Evaluate
        new_evaluation = await self._evaluate_response(query, new_response, documents)
        
        # Create result
        step = RefinementStep(
            iteration=1,
            strategy=strategy,
            original_query=query,
            refined_query=refined_query,
            retrieved_documents=documents,
            generated_response=new_response,
            quality_score=new_evaluation.overall_score,
            improvements=[],
            metadata=metadata
        )
        
        return RefinementResult(
            final_response=new_response,
            total_iterations=1,
            refinement_steps=[step],
            quality_improvement=new_evaluation.overall_score - evaluation.overall_score,
            strategies_used=[strategy],
            final_quality_score=new_evaluation.overall_score,
            converged=new_evaluation.overall_score >= self.config["quality_threshold"],
            metadata={"strategy": strategy.value}
        )