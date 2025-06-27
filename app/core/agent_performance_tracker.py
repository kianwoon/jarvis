"""
Agent Performance Tracker

Tracks agent performance metrics to improve selection decisions over time.
Stores success rates, response quality, and execution times.
"""

import json
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

from app.core.redis_base import RedisCache

logger = logging.getLogger(__name__)

@dataclass
class AgentPerformanceMetric:
    """Individual performance metric for an agent"""
    timestamp: float
    success: bool
    response_length: int
    execution_time: float
    user_feedback: Optional[int] = None  # 1-5 rating
    question_complexity: Optional[str] = None
    question_domain: Optional[str] = None
    collaboration_mode: Optional[str] = None

@dataclass
class AgentPerformanceStats:
    """Aggregated performance statistics for an agent"""
    agent_name: str
    total_executions: int
    success_rate: float
    avg_response_length: float
    avg_execution_time: float
    avg_user_rating: float
    domain_success_rates: Dict[str, float]
    complexity_success_rates: Dict[str, float]
    last_updated: float

class AgentPerformanceTracker:
    """Tracks and analyzes agent performance over time"""
    
    def __init__(self, cache_ttl: int = 3600):  # 1 hour cache
        self.cache = RedisCache(key_prefix="agent_performance:")
        self.cache_ttl = cache_ttl
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=100))  # Keep last 100 metrics per agent
        
    def record_agent_execution(self, agent_name: str, success: bool, 
                             response_length: int, execution_time: float,
                             question_complexity: str = None, 
                             question_domain: str = None,
                             collaboration_mode: str = None,
                             user_feedback: int = None):
        """Record an agent execution result"""
        
        metric = AgentPerformanceMetric(
            timestamp=time.time(),
            success=success,
            response_length=response_length,
            execution_time=execution_time,
            user_feedback=user_feedback,
            question_complexity=question_complexity,
            question_domain=question_domain,
            collaboration_mode=collaboration_mode
        )
        
        # Add to buffer
        self.metrics_buffer[agent_name].append(metric)
        
        # Store in persistent cache
        self._store_metric(agent_name, metric)
        
        # Update aggregated stats
        self._update_agent_stats(agent_name)
        
        logger.info(f"Recorded performance metric for {agent_name}: success={success}, "
                   f"time={execution_time:.2f}s, length={response_length}")
    
    def _store_metric(self, agent_name: str, metric: AgentPerformanceMetric):
        """Store individual metric in Redis"""
        try:
            metrics_key = f"metrics:{agent_name}"
            existing_metrics = self.cache.get(metrics_key) or []
            
            # Add new metric
            existing_metrics.append(asdict(metric))
            
            # Keep only last 100 metrics
            if len(existing_metrics) > 100:
                existing_metrics = existing_metrics[-100:]
            
            self.cache.set(metrics_key, existing_metrics, expire=self.cache_ttl * 24)  # 24 hours for metrics
            
        except Exception as e:
            logger.warning(f"Failed to store metric for {agent_name}: {e}")
    
    def _update_agent_stats(self, agent_name: str):
        """Update aggregated statistics for an agent"""
        try:
            # Get recent metrics
            metrics_key = f"metrics:{agent_name}"
            metrics_data = self.cache.get(metrics_key) or []
            
            if not metrics_data:
                return
            
            metrics = [AgentPerformanceMetric(**m) for m in metrics_data]
            
            # Calculate aggregated stats
            total_executions = len(metrics)
            successes = sum(1 for m in metrics if m.success)
            success_rate = successes / total_executions if total_executions > 0 else 0.0
            
            avg_response_length = sum(m.response_length for m in metrics) / total_executions
            avg_execution_time = sum(m.execution_time for m in metrics) / total_executions
            
            # Calculate average user rating (excluding None values)
            ratings = [m.user_feedback for m in metrics if m.user_feedback is not None]
            avg_user_rating = sum(ratings) / len(ratings) if ratings else 0.0
            
            # Calculate domain-specific success rates
            domain_stats = defaultdict(lambda: {'total': 0, 'success': 0})
            for m in metrics:
                if m.question_domain:
                    domain_stats[m.question_domain]['total'] += 1
                    if m.success:
                        domain_stats[m.question_domain]['success'] += 1
            
            domain_success_rates = {
                domain: stats['success'] / stats['total'] if stats['total'] > 0 else 0.0
                for domain, stats in domain_stats.items()
            }
            
            # Calculate complexity-specific success rates
            complexity_stats = defaultdict(lambda: {'total': 0, 'success': 0})
            for m in metrics:
                if m.question_complexity:
                    complexity_stats[m.question_complexity]['total'] += 1
                    if m.success:
                        complexity_stats[m.question_complexity]['success'] += 1
            
            complexity_success_rates = {
                complexity: stats['success'] / stats['total'] if stats['total'] > 0 else 0.0
                for complexity, stats in complexity_stats.items()
            }
            
            # Create aggregated stats
            stats = AgentPerformanceStats(
                agent_name=agent_name,
                total_executions=total_executions,
                success_rate=success_rate,
                avg_response_length=avg_response_length,
                avg_execution_time=avg_execution_time,
                avg_user_rating=avg_user_rating,
                domain_success_rates=domain_success_rates,
                complexity_success_rates=complexity_success_rates,
                last_updated=time.time()
            )
            
            # Store aggregated stats
            stats_key = f"stats:{agent_name}"
            self.cache.set(stats_key, asdict(stats), expire=self.cache_ttl)
            
        except Exception as e:
            logger.warning(f"Failed to update stats for {agent_name}: {e}")
    
    def get_agent_stats(self, agent_name: str) -> Optional[AgentPerformanceStats]:
        """Get performance statistics for an agent"""
        try:
            stats_key = f"stats:{agent_name}"
            stats_data = self.cache.get(stats_key)
            
            if stats_data:
                return AgentPerformanceStats(**stats_data)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get stats for {agent_name}: {e}")
            return None
    
    def get_all_agent_stats(self) -> Dict[str, AgentPerformanceStats]:
        """Get performance statistics for all tracked agents"""
        all_stats = {}
        
        try:
            # This would require scanning Redis keys, which is expensive
            # For now, return stats for agents in buffer
            for agent_name in self.metrics_buffer.keys():
                stats = self.get_agent_stats(agent_name)
                if stats:
                    all_stats[agent_name] = stats
                    
        except Exception as e:
            logger.warning(f"Failed to get all agent stats: {e}")
        
        return all_stats
    
    def get_performance_score(self, agent_name: str, question_domain: str = None,
                            question_complexity: str = None) -> float:
        """Get a performance-based score for agent selection (0.0 to 1.0)"""
        
        stats = self.get_agent_stats(agent_name)
        if not stats:
            return 0.5  # Neutral score for new agents
        
        # Base success rate
        score = stats.success_rate
        
        # Adjust for domain-specific performance
        if question_domain and question_domain in stats.domain_success_rates:
            domain_success_rate = stats.domain_success_rates[question_domain]
            # Weight domain-specific performance more heavily
            score = score * 0.3 + domain_success_rate * 0.7
        
        # Adjust for complexity-specific performance
        if question_complexity and question_complexity in stats.complexity_success_rates:
            complexity_success_rate = stats.complexity_success_rates[question_complexity]
            # Incorporate complexity performance
            score = score * 0.7 + complexity_success_rate * 0.3
        
        # Boost score based on user feedback
        if stats.avg_user_rating > 0:
            # Convert 1-5 rating to 0-1 multiplier (3 is neutral)
            rating_multiplier = 0.8 + (stats.avg_user_rating - 3) * 0.1
            score *= max(0.5, min(1.5, rating_multiplier))
        
        # Penalize very slow agents
        if stats.avg_execution_time > 60:  # More than 1 minute
            score *= 0.9
        elif stats.avg_execution_time > 120:  # More than 2 minutes
            score *= 0.8
        
        # Boost agents with sufficient data
        if stats.total_executions >= 10:
            score *= 1.05  # Small boost for experienced agents
        elif stats.total_executions >= 5:
            score *= 1.02
        
        return max(0.0, min(1.0, score))
    
    def get_top_performers(self, domain: str = None, complexity: str = None, 
                          limit: int = 5) -> List[str]:
        """Get top performing agents for specific criteria"""
        
        all_stats = self.get_all_agent_stats()
        agent_scores = []
        
        for agent_name, stats in all_stats.items():
            if stats.total_executions >= 3:  # Minimum executions for ranking
                score = self.get_performance_score(agent_name, domain, complexity)
                agent_scores.append((agent_name, score))
        
        # Sort by score and return top performers
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        return [agent_name for agent_name, _ in agent_scores[:limit]]
    
    def clear_agent_metrics(self, agent_name: str):
        """Clear all metrics for an agent (useful for testing)"""
        try:
            metrics_key = f"metrics:{agent_name}"
            stats_key = f"stats:{agent_name}"
            
            self.cache.delete(metrics_key)
            self.cache.delete(stats_key)
            
            if agent_name in self.metrics_buffer:
                del self.metrics_buffer[agent_name]
                
            logger.info(f"Cleared metrics for agent: {agent_name}")
            
        except Exception as e:
            logger.warning(f"Failed to clear metrics for {agent_name}: {e}")
    
    def export_agent_metrics(self, agent_name: str) -> Dict:
        """Export all metrics for an agent (for analysis)"""
        try:
            metrics_key = f"metrics:{agent_name}"
            stats_key = f"stats:{agent_name}"
            
            metrics_data = self.cache.get(metrics_key) or []
            stats_data = self.cache.get(stats_key)
            
            return {
                "agent_name": agent_name,
                "metrics": metrics_data,
                "aggregated_stats": stats_data,
                "export_timestamp": time.time()
            }
            
        except Exception as e:
            logger.warning(f"Failed to export metrics for {agent_name}: {e}")
            return {}

# Global instance
performance_tracker = AgentPerformanceTracker()