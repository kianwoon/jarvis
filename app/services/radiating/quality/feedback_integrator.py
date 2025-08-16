"""
Feedback Integrator

Collects user feedback on results, learns from corrections, adjusts scoring weights,
and stores feedback for training.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import asyncio
from collections import defaultdict
import numpy as np

from app.core.redis_client import get_redis_client
from app.core.db import get_db_session
from app.core.radiating_settings_cache import get_radiating_settings

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of user feedback"""
    RELEVANCE = "relevance"          # Entity/relationship relevance
    ACCURACY = "accuracy"            # Extraction accuracy
    COMPLETENESS = "completeness"    # Missing entities/relationships
    USEFULNESS = "usefulness"        # Overall result usefulness
    CORRECTION = "correction"        # Specific correction provided


class FeedbackSentiment(Enum):
    """Feedback sentiment"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


@dataclass
class UserFeedback:
    """User feedback data"""
    id: str
    query_id: str
    feedback_type: FeedbackType
    sentiment: FeedbackSentiment
    score: float  # 0-1 scale
    entity_id: Optional[str] = None
    relationship_id: Optional[str] = None
    comment: Optional[str] = None
    correction: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    processed: bool = False


@dataclass
class LearningOutcome:
    """Result of learning from feedback"""
    feedback_id: str
    learning_type: str
    adjustments_made: Dict[str, Any]
    confidence: float
    impact_score: float  # How much this learning impacts future results
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WeightAdjustment:
    """Weight adjustment based on feedback"""
    component: str
    original_weight: float
    new_weight: float
    adjustment_reason: str
    feedback_count: int
    confidence: float


class FeedbackIntegrator:
    """
    Integrates user feedback to improve radiating system performance
    through learning and weight adjustments.
    """
    
    # Learning configuration
    LEARNING_CONFIG = {
        'min_feedback_for_adjustment': 5,
        'weight_adjustment_rate': 0.1,
        'confidence_threshold': 0.7,
        'feedback_decay_days': 30,
        'max_weight_change': 0.3
    }
    
    # Component weights that can be adjusted
    ADJUSTABLE_WEIGHTS = {
        'relevance_score': 0.3,
        'semantic_similarity': 0.25,
        'structural_importance': 0.2,
        'confidence_score': 0.15,
        'temporal_relevance': 0.1
    }
    
    # Feedback impact weights
    FEEDBACK_IMPACT = {
        FeedbackType.RELEVANCE: 0.35,
        FeedbackType.ACCURACY: 0.3,
        FeedbackType.COMPLETENESS: 0.2,
        FeedbackType.USEFULNESS: 0.15
    }
    
    def __init__(self):
        """Initialize FeedbackIntegrator"""
        self.redis_client = get_redis_client()
        
        # Feedback storage
        self.feedback_cache: Dict[str, List[UserFeedback]] = defaultdict(list)
        self.learning_history: List[LearningOutcome] = []
        
        # Current weights (loaded from settings or defaults)
        self.current_weights = self.ADJUSTABLE_WEIGHTS.copy()
        self._load_weights()
        
        # Statistics
        self.stats = {
            'total_feedback': 0,
            'positive_feedback': 0,
            'negative_feedback': 0,
            'corrections_applied': 0,
            'weights_adjusted': 0,
            'learning_outcomes': 0
        }
        
        # Start background tasks
        asyncio.create_task(self._feedback_processor())
        asyncio.create_task(self._weight_optimizer())
    
    def _load_weights(self):
        """Load current weights from settings"""
        try:
            settings = get_radiating_settings()
            if 'scoring_weights' in settings:
                self.current_weights.update(settings['scoring_weights'])
                logger.info("Loaded custom scoring weights from settings")
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
    
    async def collect_feedback(
        self,
        query_id: str,
        feedback_type: FeedbackType,
        score: float,
        entity_id: Optional[str] = None,
        relationship_id: Optional[str] = None,
        comment: Optional[str] = None,
        correction: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> UserFeedback:
        """
        Collect user feedback on results
        
        Args:
            query_id: ID of the query/result being evaluated
            feedback_type: Type of feedback
            score: Feedback score (0-1, where 1 is best)
            entity_id: Specific entity being evaluated
            relationship_id: Specific relationship being evaluated
            comment: User comment
            correction: Specific correction provided
            user_id: User providing feedback
            
        Returns:
            UserFeedback object
        """
        self.stats['total_feedback'] += 1
        
        # Determine sentiment
        if score >= 0.7:
            sentiment = FeedbackSentiment.POSITIVE
            self.stats['positive_feedback'] += 1
        elif score >= 0.4:
            sentiment = FeedbackSentiment.NEUTRAL
        else:
            sentiment = FeedbackSentiment.NEGATIVE
            self.stats['negative_feedback'] += 1
        
        # Create feedback object
        feedback = UserFeedback(
            id=f"feedback-{datetime.now().timestamp()}",
            query_id=query_id,
            feedback_type=feedback_type,
            sentiment=sentiment,
            score=score,
            entity_id=entity_id,
            relationship_id=relationship_id,
            comment=comment,
            correction=correction,
            user_id=user_id
        )
        
        # Store feedback
        await self._store_feedback(feedback)
        
        # Add to cache for processing
        self.feedback_cache[query_id].append(feedback)
        
        # Process correction immediately if provided
        if correction:
            await self._apply_correction(feedback)
        
        return feedback
    
    async def _store_feedback(self, feedback: UserFeedback):
        """Store feedback in database and cache"""
        try:
            # Store in Redis for quick access
            feedback_key = f"radiating:feedback:{feedback.id}"
            await self.redis_client.setex(
                feedback_key,
                86400 * 30,  # 30 days
                json.dumps(asdict(feedback), default=str)
            )
            
            # Add to feedback list
            await self.redis_client.lpush(
                f"radiating:feedback:list:{feedback.query_id}",
                feedback.id
            )
            
            # Store in database for persistence
            async with get_db_session() as session:
                query = """
                INSERT INTO radiating_feedback 
                (id, query_id, feedback_type, sentiment, score, entity_id, 
                 relationship_id, comment, correction, user_id, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                await session.execute(
                    query,
                    (
                        feedback.id,
                        feedback.query_id,
                        feedback.feedback_type.value,
                        feedback.sentiment.value,
                        feedback.score,
                        feedback.entity_id,
                        feedback.relationship_id,
                        feedback.comment,
                        json.dumps(feedback.correction) if feedback.correction else None,
                        feedback.user_id,
                        feedback.timestamp
                    )
                )
                await session.commit()
        
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
    
    async def _apply_correction(self, feedback: UserFeedback):
        """Apply user correction immediately"""
        if not feedback.correction:
            return
        
        try:
            correction = feedback.correction
            
            # Handle entity corrections
            if feedback.entity_id and 'entity' in correction:
                await self._correct_entity(
                    feedback.entity_id,
                    correction['entity']
                )
            
            # Handle relationship corrections
            if feedback.relationship_id and 'relationship' in correction:
                await self._correct_relationship(
                    feedback.relationship_id,
                    correction['relationship']
                )
            
            self.stats['corrections_applied'] += 1
            
            # Create learning outcome
            outcome = LearningOutcome(
                feedback_id=feedback.id,
                learning_type="correction",
                adjustments_made=correction,
                confidence=0.9,  # High confidence for direct corrections
                impact_score=0.8
            )
            
            self.learning_history.append(outcome)
            self.stats['learning_outcomes'] += 1
            
        except Exception as e:
            logger.error(f"Error applying correction: {e}")
    
    async def _correct_entity(self, entity_id: str, corrections: Dict[str, Any]):
        """Apply corrections to an entity"""
        # Store corrections for future reference
        correction_key = f"radiating:corrections:entity:{entity_id}"
        await self.redis_client.setex(
            correction_key,
            86400 * 7,  # 7 days
            json.dumps(corrections)
        )
        
        logger.info(f"Applied corrections to entity {entity_id}: {corrections}")
    
    async def _correct_relationship(self, rel_id: str, corrections: Dict[str, Any]):
        """Apply corrections to a relationship"""
        # Store corrections for future reference
        correction_key = f"radiating:corrections:relationship:{rel_id}"
        await self.redis_client.setex(
            correction_key,
            86400 * 7,  # 7 days
            json.dumps(corrections)
        )
        
        logger.info(f"Applied corrections to relationship {rel_id}: {corrections}")
    
    async def learn_from_feedback(
        self,
        query_id: str,
        min_feedback: int = 3
    ) -> List[LearningOutcome]:
        """
        Learn from accumulated feedback for a query
        
        Args:
            query_id: Query to learn from
            min_feedback: Minimum feedback required for learning
            
        Returns:
            List of learning outcomes
        """
        feedback_list = self.feedback_cache.get(query_id, [])
        
        if len(feedback_list) < min_feedback:
            return []
        
        outcomes = []
        
        # Analyze feedback patterns
        patterns = self._analyze_feedback_patterns(feedback_list)
        
        # Learn from relevance feedback
        if patterns['relevance']['count'] >= min_feedback:
            outcome = await self._learn_relevance_preferences(
                patterns['relevance'],
                feedback_list
            )
            if outcome:
                outcomes.append(outcome)
        
        # Learn from accuracy feedback
        if patterns['accuracy']['count'] >= min_feedback:
            outcome = await self._learn_accuracy_improvements(
                patterns['accuracy'],
                feedback_list
            )
            if outcome:
                outcomes.append(outcome)
        
        # Learn from completeness feedback
        if patterns['completeness']['negative_count'] > 0:
            outcome = await self._learn_missing_patterns(
                patterns['completeness'],
                feedback_list
            )
            if outcome:
                outcomes.append(outcome)
        
        # Update learning history
        self.learning_history.extend(outcomes)
        
        return outcomes
    
    def _analyze_feedback_patterns(
        self,
        feedback_list: List[UserFeedback]
    ) -> Dict[str, Any]:
        """Analyze patterns in feedback"""
        patterns = {
            'relevance': {
                'count': 0,
                'avg_score': 0,
                'positive_count': 0,
                'negative_count': 0
            },
            'accuracy': {
                'count': 0,
                'avg_score': 0,
                'corrections': []
            },
            'completeness': {
                'count': 0,
                'avg_score': 0,
                'missing_entities': [],
                'negative_count': 0
            },
            'usefulness': {
                'count': 0,
                'avg_score': 0
            }
        }
        
        for feedback in feedback_list:
            pattern = patterns.get(feedback.feedback_type.value, {})
            pattern['count'] = pattern.get('count', 0) + 1
            
            # Update average score
            current_avg = pattern.get('avg_score', 0)
            count = pattern['count']
            pattern['avg_score'] = (
                (current_avg * (count - 1) + feedback.score) / count
            )
            
            # Track sentiment
            if feedback.sentiment == FeedbackSentiment.POSITIVE:
                pattern['positive_count'] = pattern.get('positive_count', 0) + 1
            elif feedback.sentiment == FeedbackSentiment.NEGATIVE:
                pattern['negative_count'] = pattern.get('negative_count', 0) + 1
            
            # Collect corrections
            if feedback.correction:
                if feedback.feedback_type == FeedbackType.ACCURACY:
                    pattern.setdefault('corrections', []).append(feedback.correction)
                elif feedback.feedback_type == FeedbackType.COMPLETENESS:
                    pattern.setdefault('missing_entities', []).extend(
                        feedback.correction.get('missing', [])
                    )
        
        return patterns
    
    async def _learn_relevance_preferences(
        self,
        pattern: Dict[str, Any],
        feedback_list: List[UserFeedback]
    ) -> Optional[LearningOutcome]:
        """Learn relevance preferences from feedback"""
        if pattern['avg_score'] < 0.5:
            # Users finding results irrelevant
            adjustments = {
                'relevance_threshold': 'increase',
                'depth_limit': 'decrease'
            }
            
            return LearningOutcome(
                feedback_id=f"batch-{datetime.now().timestamp()}",
                learning_type="relevance_preferences",
                adjustments_made=adjustments,
                confidence=pattern['count'] / 10,  # More feedback = higher confidence
                impact_score=0.7
            )
        
        return None
    
    async def _learn_accuracy_improvements(
        self,
        pattern: Dict[str, Any],
        feedback_list: List[UserFeedback]
    ) -> Optional[LearningOutcome]:
        """Learn accuracy improvements from corrections"""
        if pattern['corrections']:
            # Analyze common correction patterns
            correction_types = defaultdict(int)
            
            for correction in pattern['corrections']:
                for key in correction.keys():
                    correction_types[key] += 1
            
            # Find most common correction type
            if correction_types:
                most_common = max(correction_types, key=correction_types.get)
                
                adjustments = {
                    'focus_area': most_common,
                    'validation_strength': 'increase'
                }
                
                return LearningOutcome(
                    feedback_id=f"batch-{datetime.now().timestamp()}",
                    learning_type="accuracy_improvements",
                    adjustments_made=adjustments,
                    confidence=len(pattern['corrections']) / 10,
                    impact_score=0.8
                )
        
        return None
    
    async def _learn_missing_patterns(
        self,
        pattern: Dict[str, Any],
        feedback_list: List[UserFeedback]
    ) -> Optional[LearningOutcome]:
        """Learn about commonly missing entities/relationships"""
        if pattern.get('missing_entities'):
            # Identify patterns in missing entities
            missing_types = defaultdict(int)
            
            for entity in pattern['missing_entities']:
                if isinstance(entity, dict) and 'type' in entity:
                    missing_types[entity['type']] += 1
            
            if missing_types:
                adjustments = {
                    'expand_entity_types': list(missing_types.keys()),
                    'increase_depth_for_types': dict(missing_types)
                }
                
                return LearningOutcome(
                    feedback_id=f"batch-{datetime.now().timestamp()}",
                    learning_type="completeness_improvements",
                    adjustments_made=adjustments,
                    confidence=pattern['negative_count'] / max(1, pattern['count']),
                    impact_score=0.6
                )
        
        return None
    
    async def adjust_scoring_weights(
        self,
        feedback_summary: Dict[str, float]
    ) -> List[WeightAdjustment]:
        """
        Adjust scoring weights based on feedback
        
        Args:
            feedback_summary: Summary of feedback scores by component
            
        Returns:
            List of weight adjustments made
        """
        adjustments = []
        
        for component, current_weight in self.current_weights.items():
            if component not in feedback_summary:
                continue
            
            feedback_score = feedback_summary[component]
            
            # Determine adjustment direction
            if feedback_score < 0.4:
                # Component performing poorly, reduce weight
                adjustment = -self.LEARNING_CONFIG['weight_adjustment_rate']
                reason = f"Poor feedback score: {feedback_score:.2f}"
            elif feedback_score > 0.7:
                # Component performing well, increase weight
                adjustment = self.LEARNING_CONFIG['weight_adjustment_rate']
                reason = f"Good feedback score: {feedback_score:.2f}"
            else:
                # No adjustment needed
                continue
            
            # Apply adjustment with limits
            new_weight = current_weight + adjustment
            new_weight = max(0.05, min(0.5, new_weight))  # Keep weights between 0.05 and 0.5
            
            # Check maximum change
            if abs(new_weight - current_weight) > self.LEARNING_CONFIG['max_weight_change']:
                new_weight = current_weight + (
                    self.LEARNING_CONFIG['max_weight_change'] * 
                    (1 if adjustment > 0 else -1)
                )
            
            # Create adjustment record
            weight_adjustment = WeightAdjustment(
                component=component,
                original_weight=current_weight,
                new_weight=new_weight,
                adjustment_reason=reason,
                feedback_count=len(self.feedback_cache),
                confidence=min(1.0, len(self.feedback_cache) / 10)
            )
            
            adjustments.append(weight_adjustment)
            
            # Update weight
            self.current_weights[component] = new_weight
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.current_weights.values())
        if total_weight > 0:
            for component in self.current_weights:
                self.current_weights[component] /= total_weight
        
        # Save updated weights
        await self._save_weights()
        
        self.stats['weights_adjusted'] += len(adjustments)
        
        return adjustments
    
    async def _save_weights(self):
        """Save updated weights to settings"""
        try:
            # Store in Redis
            await self.redis_client.setex(
                "radiating:scoring_weights",
                86400,  # 24 hours
                json.dumps(self.current_weights)
            )
            
            # Update in database
            async with get_db_session() as session:
                query = """
                UPDATE settings 
                SET settings = jsonb_set(settings, '{scoring_weights}', %s::jsonb)
                WHERE category = 'radiating'
                """
                
                await session.execute(
                    query,
                    (json.dumps(self.current_weights),)
                )
                await session.commit()
            
            logger.info(f"Updated scoring weights: {self.current_weights}")
        
        except Exception as e:
            logger.error(f"Error saving weights: {e}")
    
    async def _feedback_processor(self):
        """Background task to process feedback"""
        while True:
            try:
                await asyncio.sleep(60)  # Process every minute
                
                # Process feedback for queries with enough data
                for query_id, feedback_list in self.feedback_cache.items():
                    if len(feedback_list) >= self.LEARNING_CONFIG['min_feedback_for_adjustment']:
                        # Learn from feedback
                        outcomes = await self.learn_from_feedback(query_id)
                        
                        if outcomes:
                            logger.info(
                                f"Learned from {len(feedback_list)} feedback items "
                                f"for query {query_id}: {len(outcomes)} outcomes"
                            )
                        
                        # Mark feedback as processed
                        for feedback in feedback_list:
                            feedback.processed = True
                
                # Clean old processed feedback
                self._clean_old_feedback()
                
            except Exception as e:
                logger.error(f"Error in feedback processor: {e}")
    
    async def _weight_optimizer(self):
        """Background task to optimize weights"""
        while True:
            try:
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
                # Calculate feedback summary
                feedback_summary = await self._calculate_feedback_summary()
                
                if feedback_summary:
                    # Adjust weights based on feedback
                    adjustments = await self.adjust_scoring_weights(feedback_summary)
                    
                    if adjustments:
                        logger.info(f"Adjusted {len(adjustments)} scoring weights")
                
            except Exception as e:
                logger.error(f"Error in weight optimizer: {e}")
    
    async def _calculate_feedback_summary(self) -> Dict[str, float]:
        """Calculate summary of feedback by component"""
        summary = {}
        
        # Aggregate feedback scores by component
        component_scores = defaultdict(list)
        
        for feedback_list in self.feedback_cache.values():
            for feedback in feedback_list:
                if feedback.feedback_type == FeedbackType.RELEVANCE:
                    component_scores['relevance_score'].append(feedback.score)
                elif feedback.feedback_type == FeedbackType.ACCURACY:
                    component_scores['confidence_score'].append(feedback.score)
                elif feedback.feedback_type == FeedbackType.USEFULNESS:
                    # Usefulness affects all components equally
                    for component in self.current_weights:
                        component_scores[component].append(feedback.score)
        
        # Calculate averages
        for component, scores in component_scores.items():
            if scores:
                summary[component] = sum(scores) / len(scores)
        
        return summary
    
    def _clean_old_feedback(self):
        """Remove old processed feedback from cache"""
        cutoff = datetime.now() - timedelta(
            days=self.LEARNING_CONFIG['feedback_decay_days']
        )
        
        for query_id in list(self.feedback_cache.keys()):
            # Keep only recent or unprocessed feedback
            self.feedback_cache[query_id] = [
                f for f in self.feedback_cache[query_id]
                if not f.processed or f.timestamp > cutoff
            ]
            
            # Remove empty entries
            if not self.feedback_cache[query_id]:
                del self.feedback_cache[query_id]
    
    async def get_feedback_summary(self, query_id: str) -> Dict[str, Any]:
        """Get summary of feedback for a query"""
        feedback_list = self.feedback_cache.get(query_id, [])
        
        if not feedback_list:
            # Try loading from storage
            feedback_ids = await self.redis_client.lrange(
                f"radiating:feedback:list:{query_id}",
                0,
                -1
            )
            
            for feedback_id in feedback_ids:
                feedback_data = await self.redis_client.get(
                    f"radiating:feedback:{feedback_id}"
                )
                if feedback_data:
                    # Convert back to UserFeedback
                    data = json.loads(feedback_data)
                    feedback = UserFeedback(
                        id=data['id'],
                        query_id=data['query_id'],
                        feedback_type=FeedbackType(data['feedback_type']),
                        sentiment=FeedbackSentiment(data['sentiment']),
                        score=data['score'],
                        entity_id=data.get('entity_id'),
                        relationship_id=data.get('relationship_id'),
                        comment=data.get('comment'),
                        correction=data.get('correction'),
                        user_id=data.get('user_id'),
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        processed=data.get('processed', False)
                    )
                    feedback_list.append(feedback)
        
        if not feedback_list:
            return {
                'query_id': query_id,
                'feedback_count': 0,
                'average_score': 0,
                'sentiment_distribution': {},
                'type_distribution': {}
            }
        
        # Calculate summary statistics
        scores = [f.score for f in feedback_list]
        
        sentiment_dist = defaultdict(int)
        type_dist = defaultdict(int)
        
        for feedback in feedback_list:
            sentiment_dist[feedback.sentiment.value] += 1
            type_dist[feedback.feedback_type.value] += 1
        
        return {
            'query_id': query_id,
            'feedback_count': len(feedback_list),
            'average_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'sentiment_distribution': dict(sentiment_dist),
            'type_distribution': dict(type_dist),
            'corrections_provided': sum(
                1 for f in feedback_list if f.correction
            ),
            'latest_feedback': feedback_list[-1].timestamp.isoformat()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get feedback integration statistics"""
        return {
            'total_feedback': self.stats['total_feedback'],
            'positive_feedback': self.stats['positive_feedback'],
            'negative_feedback': self.stats['negative_feedback'],
            'positive_rate': (
                self.stats['positive_feedback'] / self.stats['total_feedback'] * 100
                if self.stats['total_feedback'] > 0 else 0
            ),
            'corrections_applied': self.stats['corrections_applied'],
            'weights_adjusted': self.stats['weights_adjusted'],
            'learning_outcomes': self.stats['learning_outcomes'],
            'current_weights': self.current_weights,
            'active_queries': len(self.feedback_cache),
            'total_cached_feedback': sum(
                len(feedback_list)
                for feedback_list in self.feedback_cache.values()
            )
        }