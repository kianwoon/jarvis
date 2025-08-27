"""
Operation Performance Tracker

Tracks performance metrics for expensive operations with focus on duplicate
detection and prevention. Integrates with existing timeout settings and
caching infrastructure to provide comprehensive performance monitoring.

Key Operations Tracked:
- Intent Analysis (AI-powered query understanding)  
- Task Planning (AI task execution planning)
- Query Embedding (Vector embedding generation)
- Batch Extraction (Large-scale data extraction)
- Verification (AI verification processes)
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from functools import wraps
import inspect

from app.core.redis_client import get_redis_client
from app.core.timeout_settings_cache import get_timeout_value
from app.services.duplicate_execution_monitor import (
    DuplicateExecutionMonitor, DuplicateOperationType, get_duplicate_execution_monitor
)

logger = logging.getLogger(__name__)


@dataclass
class OperationContext:
    """Context information for operation tracking"""
    operation_type: DuplicateOperationType
    request_id: str
    conversation_id: Optional[str] = None
    notebook_id: Optional[str] = None
    query: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None
    cache_key: Optional[str] = None
    timeout_config: Optional[Dict[str, Any]] = None


@dataclass
class OperationResult:
    """Result of tracked operation execution"""
    success: bool
    execution_time: float
    cache_hit: bool = False
    timeout_occurred: bool = False
    circuit_breaker_activated: bool = False
    error_message: Optional[str] = None
    result_data: Optional[Any] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class OperationPerformanceTracker:
    """
    Performance tracker for expensive operations with duplicate detection.
    Provides decorators and context managers for seamless integration.
    """
    
    def __init__(self):
        """Initialize OperationPerformanceTracker"""
        self.redis_client = get_redis_client()
        # Initialize duplicate monitor reference but don't call it immediately
        self._duplicate_monitor = None
        
        # Operation timing cache (for short-term duplicate detection)
        self.operation_timings: Dict[str, List[Tuple[datetime, float]]] = {}
        self.timing_lock = asyncio.Lock()
        
        # Performance thresholds per operation type
        self.performance_thresholds = {
            DuplicateOperationType.INTENT_ANALYSIS: {
                'target_time': 2.0,     # 2 seconds target
                'warning_time': 5.0,    # 5 seconds warning
                'error_time': 10.0      # 10 seconds error
            },
            DuplicateOperationType.TASK_PLANNING: {
                'target_time': 3.0,     # 3 seconds target
                'warning_time': 8.0,    # 8 seconds warning  
                'error_time': 15.0      # 15 seconds error
            },
            DuplicateOperationType.QUERY_EMBEDDING: {
                'target_time': 1.0,     # 1 second target
                'warning_time': 3.0,    # 3 seconds warning
                'error_time': 8.0       # 8 seconds error
            },
            DuplicateOperationType.BATCH_EXTRACTION: {
                'target_time': 10.0,    # 10 seconds target
                'warning_time': 30.0,   # 30 seconds warning
                'error_time': 60.0      # 60 seconds error
            },
            DuplicateOperationType.VERIFICATION: {
                'target_time': 5.0,     # 5 seconds target
                'warning_time': 12.0,   # 12 seconds warning
                'error_time': 25.0      # 25 seconds error
            }
        }
        
        logger.info("OperationPerformanceTracker initialized")
    
    @property
    def duplicate_monitor(self):
        """Get duplicate monitor instance, creating if needed"""
        if self._duplicate_monitor is None:
            self._duplicate_monitor = get_duplicate_execution_monitor()
        return self._duplicate_monitor
    
    @asynccontextmanager
    async def track_operation(self, context: OperationContext):
        """
        Context manager for tracking operation performance
        
        Usage:
            async with tracker.track_operation(context) as result:
                # Perform operation
                result.result_data = await some_expensive_operation()
        """
        # Start tracking
        operation_id = self.duplicate_monitor.track_operation_start(
            context.operation_type,
            context.request_id,
            context.conversation_id,
            context.notebook_id,
            {
                'query': context.query,
                'cache_key': context.cache_key,
                'user_context': context.user_context
            }
        )
        
        start_time = time.time()
        result = OperationResult(success=False, execution_time=0.0)
        
        # Check for timeout configuration
        timeout_value = None
        if context.operation_type == DuplicateOperationType.INTENT_ANALYSIS:
            timeout_value = get_timeout_value('query_intent_analysis', 30000) / 1000  # Convert ms to s
        elif context.operation_type == DuplicateOperationType.TASK_PLANNING:
            timeout_value = get_timeout_value('ai_task_planning', 45000) / 1000
        elif context.operation_type == DuplicateOperationType.BATCH_EXTRACTION:
            timeout_value = get_timeout_value('batch_extraction', 120000) / 1000
        
        try:
            # Check cache first
            if context.cache_key:
                cached_result = await self._check_cache(context.cache_key)
                if cached_result:
                    result.cache_hit = True
                    result.result_data = cached_result
                    result.success = True
                    result.execution_time = time.time() - start_time
                    logger.debug(f"[CACHE_HIT] {context.operation_type.value} for {context.request_id}")
                    yield result
                    return
            
            # Yield for actual operation execution
            yield result
            
            # If we get here, operation was executed
            result.execution_time = time.time() - start_time
            
            # Check performance against thresholds
            thresholds = self.performance_thresholds.get(context.operation_type, {})
            if result.execution_time > thresholds.get('error_time', float('inf')):
                logger.error(
                    f"[SLOW_OPERATION] {context.operation_type.value} took {result.execution_time:.2f}s "
                    f"(threshold: {thresholds['error_time']}s)"
                )
            elif result.execution_time > thresholds.get('warning_time', float('inf')):
                logger.warning(
                    f"[SLOW_OPERATION] {context.operation_type.value} took {result.execution_time:.2f}s "
                    f"(threshold: {thresholds['warning_time']}s)"
                )
            
            # Store result in cache if applicable
            if context.cache_key and result.success and result.result_data:
                await self._store_cache(context.cache_key, result.result_data)
            
            # Record timing for duplicate detection
            await self._record_operation_timing(context, result.execution_time)
            
        except asyncio.TimeoutError:
            result.timeout_occurred = True
            result.execution_time = time.time() - start_time
            logger.warning(f"[TIMEOUT] {context.operation_type.value} timed out after {result.execution_time:.2f}s")
            
        except Exception as e:
            result.error_message = str(e)
            result.execution_time = time.time() - start_time
            logger.error(f"[ERROR] {context.operation_type.value} failed: {e}")
            
        finally:
            # End tracking
            self.duplicate_monitor.track_operation_end(
                operation_id,
                result.success,
                result.execution_time,
                result.cache_hit,
                result.timeout_occurred,
                result.circuit_breaker_activated
            )
    
    def track_function(
        self,
        operation_type: DuplicateOperationType,
        cache_key_generator: Optional[Callable] = None
    ):
        """
        Decorator for tracking function performance
        
        Args:
            operation_type: Type of operation being tracked
            cache_key_generator: Optional function to generate cache key from args
        
        Usage:
            @tracker.track_function(DuplicateOperationType.INTENT_ANALYSIS)
            async def analyze_intent(query: str, request_id: str):
                # Function implementation
                pass
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract context from function arguments
                context = await self._extract_function_context(
                    func, args, kwargs, operation_type, cache_key_generator
                )
                
                async with self.track_operation(context) as result:
                    if not result.cache_hit:
                        # Execute the actual function
                        if asyncio.iscoroutinefunction(func):
                            result.result_data = await func(*args, **kwargs)
                        else:
                            result.result_data = func(*args, **kwargs)
                        result.success = True
                
                return result.result_data
            
            return wrapper
        return decorator
    
    async def _extract_function_context(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        operation_type: DuplicateOperationType,
        cache_key_generator: Optional[Callable]
    ) -> OperationContext:
        """Extract operation context from function call"""
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Extract common parameters
        request_id = bound_args.arguments.get('request_id') or kwargs.get('request_id') or 'unknown'
        conversation_id = bound_args.arguments.get('conversation_id') or kwargs.get('conversation_id')
        notebook_id = bound_args.arguments.get('notebook_id') or kwargs.get('notebook_id')
        query = bound_args.arguments.get('query') or kwargs.get('query')
        
        # Generate cache key if generator provided
        cache_key = None
        if cache_key_generator:
            try:
                cache_key = cache_key_generator(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Cache key generation failed: {e}")
        
        return OperationContext(
            operation_type=operation_type,
            request_id=str(request_id),
            conversation_id=conversation_id,
            notebook_id=notebook_id,
            query=query,
            cache_key=cache_key,
            user_context=bound_args.arguments
        )
    
    async def _check_cache(self, cache_key: str) -> Optional[Any]:
        """Check if result exists in cache"""
        try:
            cached_data = await self.redis_client.get(f"operation_cache:{cache_key}")
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.debug(f"Cache check failed: {e}")
        return None
    
    async def _store_cache(self, cache_key: str, result_data: Any):
        """Store result in cache"""
        try:
            await self.redis_client.setex(
                f"operation_cache:{cache_key}",
                3600,  # 1 hour TTL
                json.dumps(result_data, default=str)
            )
        except Exception as e:
            logger.debug(f"Cache store failed: {e}")
    
    async def _record_operation_timing(self, context: OperationContext, execution_time: float):
        """Record operation timing for duplicate analysis"""
        try:
            async with self.timing_lock:
                operation_key = f"{context.operation_type.value}:{context.request_id}"
                
                if operation_key not in self.operation_timings:
                    self.operation_timings[operation_key] = []
                
                self.operation_timings[operation_key].append((datetime.now(), execution_time))
                
                # Keep only recent timings (last hour)
                cutoff_time = datetime.now() - timedelta(hours=1)
                self.operation_timings[operation_key] = [
                    (timestamp, timing) for timestamp, timing in self.operation_timings[operation_key]
                    if timestamp > cutoff_time
                ]
                
                # Clean empty entries
                if not self.operation_timings[operation_key]:
                    del self.operation_timings[operation_key]
        
        except Exception as e:
            logger.error(f"Error recording operation timing: {e}")
    
    async def get_duplicate_prevention_report(
        self,
        hours: int = 24,
        operation_type: Optional[DuplicateOperationType] = None
    ) -> Dict[str, Any]:
        """
        Generate detailed duplicate prevention report
        
        Args:
            hours: Hours of data to analyze
            operation_type: Optional filter by operation type
            
        Returns:
            Comprehensive duplicate prevention report
        """
        # Get duplicate report from monitor
        duplicate_report = await self.duplicate_monitor.get_duplicate_report(operation_type, hours)
        
        # Add performance tracking insights
        performance_insights = await self._generate_performance_insights(hours, operation_type)
        
        # Combine into comprehensive report
        return {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_period_hours': hours,
                'operation_filter': operation_type.value if operation_type else 'all',
                'tracker_version': '1.0.0'
            },
            'duplicate_analysis': duplicate_report,
            'performance_insights': performance_insights,
            'prevention_recommendations': await self._generate_prevention_recommendations(duplicate_report),
            'system_health': self.duplicate_monitor.get_operation_health_status(),
            'implementation_status': await self._check_implementation_status()
        }
    
    async def _generate_performance_insights(
        self,
        hours: int,
        operation_type: Optional[DuplicateOperationType]
    ) -> Dict[str, Any]:
        """Generate performance insights from tracked operations"""
        insights = {
            'timing_analysis': {},
            'efficiency_metrics': {},
            'optimization_opportunities': [],
            'cache_performance': {},
            'timeout_analysis': {}
        }
        
        try:
            # Analyze operation timings
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            async with self.timing_lock:
                for operation_key, timings in self.operation_timings.items():
                    op_type_str, request_id = operation_key.split(':', 1)
                    
                    # Filter by operation type if specified
                    if operation_type and op_type_str != operation_type.value:
                        continue
                    
                    # Filter by time range
                    recent_timings = [
                        (timestamp, timing) for timestamp, timing in timings
                        if timestamp > cutoff_time
                    ]
                    
                    if not recent_timings:
                        continue
                    
                    # Calculate timing statistics
                    execution_times = [timing for _, timing in recent_timings]
                    
                    insights['timing_analysis'][op_type_str] = {
                        'operation_count': len(execution_times),
                        'average_time': sum(execution_times) / len(execution_times),
                        'min_time': min(execution_times),
                        'max_time': max(execution_times),
                        'std_dev': self._calculate_std_dev(execution_times),
                        'trend': self._calculate_timing_trend(recent_timings)
                    }
            
            # Analyze efficiency metrics
            for op_type, metrics in self.duplicate_monitor.operation_metrics.items():
                if operation_type and operation_type != op_type:
                    continue
                
                if metrics.total_executions > 0:
                    insights['efficiency_metrics'][op_type.value] = {
                        'duplicate_rate': metrics.duplicate_rate,
                        'unique_operations_ratio': metrics.unique_executions / metrics.total_executions,
                        'cache_hit_rate': metrics.cache_hit_rate,
                        'timeout_rate': metrics.timeout_occurrences / metrics.total_executions,
                        'circuit_breaker_rate': metrics.circuit_breaker_activations / metrics.total_executions,
                        'efficiency_score': self._calculate_operation_efficiency(metrics)
                    }
            
            # Identify optimization opportunities
            for op_type, timing_data in insights['timing_analysis'].items():
                thresholds = self.performance_thresholds.get(
                    DuplicateOperationType(op_type), {}
                )
                
                if timing_data['average_time'] > thresholds.get('warning_time', float('inf')):
                    insights['optimization_opportunities'].append({
                        'operation': op_type,
                        'issue': 'slow_performance',
                        'current_time': timing_data['average_time'],
                        'target_time': thresholds.get('target_time'),
                        'improvement_potential': timing_data['average_time'] - thresholds.get('target_time', 0)
                    })
                
                if op_type in insights['efficiency_metrics']:
                    efficiency = insights['efficiency_metrics'][op_type]
                    
                    if efficiency['duplicate_rate'] > 0.1:  # >10% duplicates
                        insights['optimization_opportunities'].append({
                            'operation': op_type,
                            'issue': 'high_duplicate_rate',
                            'duplicate_rate': efficiency['duplicate_rate'],
                            'potential_savings': f"{efficiency['duplicate_rate']:.1%} of execution time"
                        })
                    
                    if efficiency['cache_hit_rate'] < 0.5:  # <50% cache hits
                        insights['optimization_opportunities'].append({
                            'operation': op_type,
                            'issue': 'poor_cache_performance',
                            'cache_hit_rate': efficiency['cache_hit_rate'],
                            'potential_improvement': 'Implement better caching strategy'
                        })
        
        except Exception as e:
            logger.error(f"Error generating performance insights: {e}")
            insights['error'] = str(e)
        
        return insights
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation of values"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _calculate_timing_trend(self, timings: List[Tuple[datetime, float]]) -> str:
        """Calculate timing trend (improving/degrading/stable)"""
        if len(timings) < 5:
            return 'insufficient_data'
        
        # Split into first half and second half
        mid_point = len(timings) // 2
        first_half = [timing for _, timing in timings[:mid_point]]
        second_half = [timing for _, timing in timings[mid_point:]]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        change_percent = ((second_avg - first_avg) / first_avg) * 100 if first_avg > 0 else 0
        
        if change_percent > 10:
            return 'degrading'
        elif change_percent < -10:
            return 'improving'
        else:
            return 'stable'
    
    def _calculate_operation_efficiency(self, metrics: Any) -> float:
        """Calculate efficiency score for an operation (0-100)"""
        efficiency = 100.0
        
        # Deduct for duplicates (max 40 points)
        efficiency -= metrics.duplicate_rate * 40
        
        # Deduct for timeouts (max 30 points)  
        if metrics.total_executions > 0:
            timeout_rate = metrics.timeout_occurrences / metrics.total_executions
            efficiency -= timeout_rate * 30
        
        # Deduct for circuit breaker activations (max 20 points)
        if metrics.total_executions > 0:
            cb_rate = metrics.circuit_breaker_activations / metrics.total_executions
            efficiency -= cb_rate * 20
        
        # Bonus for high cache hit rate (max 10 points bonus)
        if metrics.cache_hit_rate > 0.8:
            efficiency += (metrics.cache_hit_rate - 0.8) * 50  # Bonus for >80% cache hits
        
        return max(0, min(100, efficiency))
    
    async def _generate_prevention_recommendations(self, duplicate_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific recommendations for preventing duplicates"""
        recommendations = []
        
        try:
            # Analyze operation analysis from duplicate report
            if 'operation_analysis' in duplicate_report:
                for op_type, analysis in duplicate_report['operation_analysis'].items():
                    metrics = analysis.get('metrics', {})
                    
                    if metrics.get('duplicate_rate', 0) > 0.1:  # >10% duplicate rate
                        recommendations.append({
                            'priority': 'high',
                            'operation': op_type,
                            'issue': 'duplicate_execution',
                            'current_rate': metrics['duplicate_rate'],
                            'recommendations': [
                                f"Implement stronger operation fingerprinting for {op_type}",
                                f"Add pre-execution duplicate check with Redis semaphore",
                                f"Extend cache TTL for {op_type} results",
                                f"Add request-level deduplication before {op_type} execution"
                            ],
                            'estimated_impact': f"Reduce {metrics['duplicate_rate']:.1%} of {op_type} executions"
                        })
                    
                    if analysis.get('avg_gap_between_duplicates', 0) < 10:  # Very rapid duplicates
                        recommendations.append({
                            'priority': 'critical',
                            'operation': op_type,
                            'issue': 'rapid_duplicates',
                            'average_gap': analysis['avg_gap_between_duplicates'],
                            'recommendations': [
                                f"Implement immediate duplicate detection for {op_type}",
                                f"Add operation throttling with minimum gap enforcement",
                                f"Review caller code for retry loops",
                                f"Implement operation state locking during execution"
                            ],
                            'estimated_impact': f"Prevent rapid-fire duplicates within {analysis['avg_gap_between_duplicates']:.1f}s"
                        })
            
            # Add general system recommendations
            system_efficiency = duplicate_report.get('system_health', {}).get('efficiency_score', 100)
            if system_efficiency < 80:
                recommendations.append({
                    'priority': 'high',
                    'operation': 'system_wide',
                    'issue': 'low_system_efficiency', 
                    'current_efficiency': system_efficiency,
                    'recommendations': [
                        "Implement global operation deduplication service",
                        "Add system-wide circuit breaker for expensive operations",
                        "Review and optimize all caching strategies",
                        "Implement request-level execution state tracking",
                        "Add monitoring alerts for efficiency degradation"
                    ],
                    'estimated_impact': f"Improve system efficiency from {system_efficiency:.1f}% to >90%"
                })
        
        except Exception as e:
            logger.error(f"Error generating prevention recommendations: {e}")
        
        return sorted(recommendations, key=lambda x: {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[x['priority']])
    
    async def _check_implementation_status(self) -> Dict[str, Any]:
        """Check implementation status of duplicate prevention measures"""
        status = {
            'execution_state_tracker': False,
            'operation_caching': False, 
            'circuit_breakers': False,
            'timeout_configuration': False,
            'monitoring_integration': False
        }
        
        try:
            # Check if request_execution_state_tracker is in use
            from app.services.request_execution_state_tracker import check_operation_completed
            status['execution_state_tracker'] = True
            
            # Check cache infrastructure
            cache_test = await self.redis_client.ping()
            status['operation_caching'] = bool(cache_test)
            
            # Check timeout configuration
            timeout_test = get_timeout_value('query_intent_analysis')
            status['timeout_configuration'] = timeout_test is not None
            
            # This service provides monitoring integration
            status['monitoring_integration'] = True
            
            # Check for circuit breaker patterns
            from app.services.radiating.circuit_breaker import CircuitBreaker
            status['circuit_breakers'] = True
            
        except Exception as e:
            logger.debug(f"Implementation status check encountered: {e}")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'components': status,
            'overall_readiness': sum(status.values()) / len(status),
            'missing_components': [k for k, v in status.items() if not v],
            'recommendations': [
                f"Implement {component}" for component, implemented in status.items()
                if not implemented
            ]
        }
    
    async def create_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Create comprehensive dashboard data for monitoring UI"""
        try:
            # Get real-time metrics
            real_time_metrics = await self.duplicate_monitor.get_real_time_metrics()
            
            # Get health status
            health_status = self.duplicate_monitor.get_operation_health_status()
            
            # Get recent performance data
            performance_data = await self._get_recent_performance_data()
            
            # Create dashboard sections
            dashboard = {
                'metadata': {
                    'last_updated': datetime.now().isoformat(),
                    'monitoring_uptime': (
                        datetime.now() - self.duplicate_monitor.monitor_stats['monitoring_start']
                    ).total_seconds(),
                    'data_completeness': self._assess_data_completeness()
                },
                'summary_widgets': {
                    'total_operations': self.duplicate_monitor.monitor_stats['total_operations_tracked'],
                    'duplicates_prevented': self.duplicate_monitor.monitor_stats['total_duplicates_detected'],
                    'system_efficiency': self.duplicate_monitor.monitor_stats['system_efficiency_score'],
                    'active_patterns': len(self.duplicate_monitor.detected_patterns),
                    'health_status': health_status.get('overall_health', 0)
                },
                'real_time_metrics': real_time_metrics,
                'performance_trends': performance_data,
                'operation_health': health_status,
                'alerts_and_recommendations': {
                    'active_alerts': real_time_metrics.get('performance_alerts', []),
                    'top_recommendations': real_time_metrics.get('recommendations', [])[:5]
                }
            }
            
            return dashboard
        
        except Exception as e:
            logger.error(f"Error creating dashboard data: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _get_recent_performance_data(self, hours: int = 6) -> Dict[str, Any]:
        """Get recent performance trend data for charts"""
        try:
            # Get historical metrics from Redis
            history_data = await self.redis_client.lrange(
                f"{self.duplicate_monitor.REDIS_PREFIXES['metrics']}:history",
                0, hours * 60  # Get hourly data points
            )
            
            # Parse and process historical data
            performance_trends = {
                'duplicate_rates': [],
                'execution_times': [],
                'cache_hit_rates': [],
                'system_efficiency': []
            }
            
            for data_point in history_data[-hours * 12:]:  # Last N hours, every 5 minutes
                try:
                    point = json.loads(data_point)
                    timestamp = point['timestamp']
                    
                    # Extract key metrics for trends
                    for op_type, op_metrics in point.get('operation_metrics', {}).items():
                        performance_trends['duplicate_rates'].append({
                            'timestamp': timestamp,
                            'operation': op_type,
                            'value': op_metrics.get('duplicate_rate', 0)
                        })
                        
                        performance_trends['execution_times'].append({
                            'timestamp': timestamp,
                            'operation': op_type,
                            'value': op_metrics.get('average_execution_time', 0)
                        })
                        
                        performance_trends['cache_hit_rates'].append({
                            'timestamp': timestamp,
                            'operation': op_type,
                            'value': op_metrics.get('cache_hit_rate', 0)
                        })
                    
                    # System efficiency
                    performance_trends['system_efficiency'].append({
                        'timestamp': timestamp,
                        'value': point.get('system_stats', {}).get('system_efficiency_score', 100)
                    })
                
                except Exception as e:
                    logger.debug(f"Error parsing performance data point: {e}")
            
            return performance_trends
        
        except Exception as e:
            logger.error(f"Error getting performance trends: {e}")
            return {}
    
    def _assess_data_completeness(self) -> float:
        """Assess completeness of monitoring data (0-1 scale)"""
        completeness = 0.0
        total_checks = 0
        
        # Check if we have operation metrics for all operation types
        for op_type in DuplicateOperationType:
            total_checks += 1
            if self.duplicate_monitor.operation_metrics[op_type].total_executions > 0:
                completeness += 1
        
        # Check if we have recent data
        total_checks += 1
        if len(self.duplicate_monitor.duplicate_events) > 0:
            completeness += 1
        
        # Check if we have patterns
        total_checks += 1  
        if len(self.duplicate_monitor.detected_patterns) > 0:
            completeness += 1
        
        return completeness / total_checks if total_checks > 0 else 0.0


# Global instance
operation_performance_tracker = OperationPerformanceTracker()


def get_operation_performance_tracker() -> OperationPerformanceTracker:
    """Get the global operation performance tracker instance"""
    return operation_performance_tracker


# Convenience decorator for quick integration
def track_duplicate_execution(
    operation_type: DuplicateOperationType,
    cache_key_generator: Optional[Callable] = None
):
    """
    Convenience decorator for tracking operation duplicates
    
    Usage:
        @track_duplicate_execution(DuplicateOperationType.INTENT_ANALYSIS)
        async def analyze_intent(query: str, request_id: str):
            # Implementation
            pass
    """
    return operation_performance_tracker.track_function(operation_type, cache_key_generator)


# Cache key generators for common operations
def generate_intent_cache_key(query: str, request_id: str, **kwargs) -> str:
    """Generate cache key for intent analysis"""
    return hashlib.sha256(f"intent_analysis:{query}:{request_id}".encode()).hexdigest()


def generate_planning_cache_key(query: str, notebook_id: str, **kwargs) -> str:
    """Generate cache key for task planning"""
    return hashlib.sha256(f"task_planning:{query}:{notebook_id}".encode()).hexdigest()


def generate_embedding_cache_key(query: str, **kwargs) -> str:
    """Generate cache key for query embedding"""
    return hashlib.sha256(f"query_embedding:{query}".encode()).hexdigest()


def generate_extraction_cache_key(query: str, notebook_id: str, batch_config: dict, **kwargs) -> str:
    """Generate cache key for batch extraction"""
    config_str = json.dumps(batch_config, sort_keys=True)
    return hashlib.sha256(f"batch_extraction:{query}:{notebook_id}:{config_str}".encode()).hexdigest()