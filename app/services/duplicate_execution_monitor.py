"""
Duplicate Execution Monitor Service

Comprehensive monitoring for duplicate execution patterns to prevent future
redundant processing issues. Builds on existing monitoring infrastructure
to specifically track and alert on duplicate operations across the system.

Key Features:
- Real-time duplicate detection across all operation types
- Performance metrics tracking for expensive operations  
- Circuit breaker activation monitoring
- Cache consistency validation
- Operational dashboards for duplicate pattern analysis
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Set, Tuple, Deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import threading

from app.core.redis_client import get_redis_client
from app.services.radiating.optimization.performance_monitor import PerformanceMonitor, MetricType, AlertSeverity
from app.services.request_execution_state_tracker import ExecutionPhase

logger = logging.getLogger(__name__)


class DuplicateOperationType(Enum):
    """Types of operations that can have duplicates"""
    INTENT_ANALYSIS = "intent_analysis"
    TASK_PLANNING = "task_planning"
    QUERY_EMBEDDING = "query_embedding"
    BATCH_EXTRACTION = "batch_extraction"
    VERIFICATION = "verification"
    CACHE_OPERATION = "cache_operation"
    LLM_CALL = "llm_call"
    RAG_QUERY = "rag_query"
    NOTEBOOK_CHAT = "notebook_chat"


class DuplicateDetectionLevel(Enum):
    """Severity levels for duplicate detection"""
    INFO = "info"           # Minor duplicates, log only
    WARNING = "warning"     # Moderate duplicates, alert operators  
    ERROR = "error"         # Significant duplicates, investigate immediately
    CRITICAL = "critical"   # System-wide duplicate patterns, emergency response


@dataclass
class DuplicateEvent:
    """Single duplicate operation event"""
    event_id: str
    operation_type: DuplicateOperationType
    request_id: str
    conversation_id: Optional[str]
    notebook_id: Optional[str]
    operation_hash: str
    first_execution_time: datetime
    duplicate_execution_time: datetime
    time_between_duplicates: float  # seconds
    detection_level: DuplicateDetectionLevel
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class DuplicatePattern:
    """Detected duplicate execution pattern"""
    pattern_id: str
    operation_type: DuplicateOperationType
    frequency: int  # occurrences per hour
    average_gap: float  # average time between duplicates
    impact_score: float  # 0-1, impact on system performance
    affected_operations: List[str]  # request/conversation IDs
    first_detected: datetime
    last_detected: datetime
    recommendations: List[str] = field(default_factory=list)


@dataclass
class OperationMetrics:
    """Metrics for a specific operation type"""
    total_executions: int = 0
    unique_executions: int = 0
    duplicate_executions: int = 0
    duplicate_rate: float = 0.0
    average_execution_time: float = 0.0
    cache_hit_rate: float = 0.0
    timeout_occurrences: int = 0
    circuit_breaker_activations: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


class DuplicateExecutionMonitor:
    """
    Comprehensive monitoring service for duplicate execution patterns.
    Integrates with existing monitoring infrastructure while providing
    specialized duplicate detection and prevention capabilities.
    """
    
    # Redis key prefixes for different data types
    REDIS_PREFIXES = {
        'operations': 'duplicate_monitor:operations',
        'patterns': 'duplicate_monitor:patterns', 
        'metrics': 'duplicate_monitor:metrics',
        'events': 'duplicate_monitor:events',
        'alerts': 'duplicate_monitor:alerts',
        'dashboard': 'duplicate_monitor:dashboard'
    }
    
    # TTL settings for different data types (in seconds)
    TTL_SETTINGS = {
        'operation_tracking': 3600,      # 1 hour for operation tracking
        'pattern_detection': 86400,     # 24 hours for pattern detection
        'metrics_storage': 604800,      # 1 week for metrics
        'event_history': 259200,       # 3 days for event history
        'dashboard_data': 60            # 1 minute for dashboard data
    }
    
    # Detection thresholds
    DETECTION_THRESHOLDS = {
        'duplicate_gap_warning': 10,    # warn if duplicates within 10 seconds
        'duplicate_gap_error': 5,       # error if duplicates within 5 seconds
        'duplicate_gap_critical': 2,    # critical if duplicates within 2 seconds
        'pattern_frequency_warning': 5, # warn if 5+ duplicates per hour
        'pattern_frequency_error': 10,  # error if 10+ duplicates per hour
        'pattern_frequency_critical': 20 # critical if 20+ duplicates per hour
    }
    
    def __init__(self):
        """Initialize DuplicateExecutionMonitor"""
        self.redis_client = get_redis_client()
        # Initialize performance monitor only when needed to avoid event loop issues
        self._performance_monitor = None
        
        # Thread-safe operation tracking
        self.active_operations: Dict[str, datetime] = {}
        self.operation_hashes: Dict[str, Tuple[str, datetime]] = {}  # hash -> (operation_id, timestamp)
        self.operation_lock = threading.Lock()
        
        # Duplicate event tracking
        self.duplicate_events: Deque[DuplicateEvent] = deque(maxlen=1000)
        self.detected_patterns: Dict[str, DuplicatePattern] = {}
        
        # Metrics tracking per operation type
        self.operation_metrics: Dict[DuplicateOperationType, OperationMetrics] = {
            op_type: OperationMetrics() for op_type in DuplicateOperationType
        }
        
        # System statistics
        self.monitor_stats = {
            'monitoring_start': datetime.now(),
            'total_operations_tracked': 0,
            'total_duplicates_detected': 0,
            'total_patterns_identified': 0,
            'last_pattern_detection': None,
            'system_efficiency_score': 100.0
        }
        
        # Background tasks will be started when needed
        self._background_tasks_started = False
        
        logger.info("DuplicateExecutionMonitor initialized with comprehensive tracking")
    
    @property
    def performance_monitor(self):
        """Get performance monitor instance, creating if needed"""
        if self._performance_monitor is None:
            try:
                from app.services.radiating.optimization.performance_monitor import PerformanceMonitor
                self._performance_monitor = PerformanceMonitor()
            except Exception as e:
                logger.warning(f"Could not initialize performance monitor: {e}")
                # Create a simple fallback
                self._performance_monitor = self._create_fallback_monitor()
        return self._performance_monitor
    
    def _create_fallback_monitor(self):
        """Create a simple fallback monitor if PerformanceMonitor fails"""
        class FallbackMonitor:
            def start_operation(self, operation_id: str): pass
            def end_operation(self, operation_id: str, success: bool = True): return 0.0
            def record_error(self, error_type: str, component: str): pass
        return FallbackMonitor()
    
    def _ensure_background_tasks(self):
        """Start background tasks if not already started and event loop is available"""
        if not self._background_tasks_started:
            try:
                # Try to get current event loop
                loop = asyncio.get_running_loop()
                if loop and not loop.is_closed():
                    asyncio.create_task(self._duplicate_detector())
                    asyncio.create_task(self._pattern_analyzer())
                    asyncio.create_task(self._metrics_collector())
                    asyncio.create_task(self._dashboard_updater())
                    self._background_tasks_started = True
                    logger.debug("Background monitoring tasks started")
            except RuntimeError:
                # No event loop available yet - tasks will be started later
                pass
    
    def track_operation_start(
        self,
        operation_type: DuplicateOperationType,
        request_id: str,
        conversation_id: Optional[str] = None,
        notebook_id: Optional[str] = None,
        operation_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Track the start of an operation and check for duplicates
        
        Args:
            operation_type: Type of operation being tracked
            request_id: Unique request identifier
            conversation_id: Optional conversation identifier
            notebook_id: Optional notebook identifier  
            operation_context: Optional context for duplicate detection
            
        Returns:
            Operation tracking ID
        """
        # Ensure background tasks are running
        self._ensure_background_tasks()
        
        operation_id = f"{operation_type.value}:{request_id}:{uuid.uuid4().hex[:8]}"
        current_time = datetime.now()
        
        # Create operation hash for duplicate detection
        context_str = json.dumps(operation_context or {}, sort_keys=True)
        operation_hash = hashlib.sha256(
            f"{operation_type.value}:{request_id}:{conversation_id}:{notebook_id}:{context_str}".encode()
        ).hexdigest()
        
        with self.operation_lock:
            # Check for duplicate operations
            if operation_hash in self.operation_hashes:
                existing_op_id, existing_time = self.operation_hashes[operation_hash]
                time_gap = (current_time - existing_time).total_seconds()
                
                # Determine duplicate severity
                if time_gap < self.DETECTION_THRESHOLDS['duplicate_gap_critical']:
                    level = DuplicateDetectionLevel.CRITICAL
                elif time_gap < self.DETECTION_THRESHOLDS['duplicate_gap_error']:
                    level = DuplicateDetectionLevel.ERROR
                elif time_gap < self.DETECTION_THRESHOLDS['duplicate_gap_warning']:
                    level = DuplicateDetectionLevel.WARNING
                else:
                    level = DuplicateDetectionLevel.INFO
                
                # Record duplicate event
                duplicate_event = DuplicateEvent(
                    event_id=uuid.uuid4().hex,
                    operation_type=operation_type,
                    request_id=request_id,
                    conversation_id=conversation_id,
                    notebook_id=notebook_id,
                    operation_hash=operation_hash,
                    first_execution_time=existing_time,
                    duplicate_execution_time=current_time,
                    time_between_duplicates=time_gap,
                    detection_level=level,
                    context=operation_context or {}
                )
                
                self.duplicate_events.append(duplicate_event)
                self.monitor_stats['total_duplicates_detected'] += 1
                
                # Log based on severity
                if level == DuplicateDetectionLevel.CRITICAL:
                    logger.critical(f"CRITICAL DUPLICATE: {operation_type.value} for {request_id} within {time_gap:.2f}s")
                elif level == DuplicateDetectionLevel.ERROR:
                    logger.error(f"ERROR DUPLICATE: {operation_type.value} for {request_id} within {time_gap:.2f}s")
                elif level == DuplicateDetectionLevel.WARNING:
                    logger.warning(f"WARNING DUPLICATE: {operation_type.value} for {request_id} within {time_gap:.2f}s")
                else:
                    logger.info(f"INFO DUPLICATE: {operation_type.value} for {request_id} within {time_gap:.2f}s")
                
                # Update operation metrics
                self._update_operation_metrics(operation_type, is_duplicate=True)
                
                # Store duplicate event
                asyncio.create_task(self._store_duplicate_event(duplicate_event))
            
            # Track the operation
            self.active_operations[operation_id] = current_time
            self.operation_hashes[operation_hash] = (operation_id, current_time)
            self.monitor_stats['total_operations_tracked'] += 1
            
            # Update operation metrics
            self._update_operation_metrics(operation_type, is_duplicate=False)
        
        # Record in performance monitor
        self.performance_monitor.start_operation(operation_id)
        
        return operation_id
    
    def track_operation_end(
        self,
        operation_id: str,
        success: bool = True,
        execution_time: Optional[float] = None,
        cache_hit: Optional[bool] = None,
        timeout_occurred: bool = False,
        circuit_breaker_activated: bool = False
    ):
        """
        Track the end of an operation and record metrics
        
        Args:
            operation_id: Operation tracking ID from track_operation_start
            success: Whether the operation succeeded
            execution_time: Optional execution time override
            cache_hit: Whether this was a cache hit
            timeout_occurred: Whether a timeout occurred
            circuit_breaker_activated: Whether circuit breaker was activated
        """
        with self.operation_lock:
            if operation_id in self.active_operations:
                start_time = self.active_operations.pop(operation_id)
                
                if execution_time is None:
                    execution_time = (datetime.now() - start_time).total_seconds()
                
                # Extract operation type from ID
                operation_type_str = operation_id.split(':')[0]
                try:
                    operation_type = DuplicateOperationType(operation_type_str)
                    
                    # Update metrics
                    metrics = self.operation_metrics[operation_type]
                    metrics.total_executions += 1
                    
                    if cache_hit is not None:
                        # Update cache hit rate
                        total_cache_ops = metrics.total_executions
                        current_hits = metrics.cache_hit_rate * (total_cache_ops - 1)
                        if cache_hit:
                            current_hits += 1
                        metrics.cache_hit_rate = current_hits / total_cache_ops
                    
                    if timeout_occurred:
                        metrics.timeout_occurrences += 1
                    
                    if circuit_breaker_activated:
                        metrics.circuit_breaker_activations += 1
                    
                    # Update average execution time
                    total_time = metrics.average_execution_time * (metrics.total_executions - 1)
                    metrics.average_execution_time = (total_time + execution_time) / metrics.total_executions
                    metrics.last_updated = datetime.now()
                    
                except ValueError:
                    logger.warning(f"Unknown operation type in ID: {operation_id}")
        
        # Record in performance monitor
        duration = self.performance_monitor.end_operation(operation_id, success)
        
        # Record specific metrics
        if timeout_occurred:
            self.performance_monitor.record_error("timeout", "operation_execution")
        
        if circuit_breaker_activated:
            self.performance_monitor.record_error("circuit_breaker", "operation_execution")
    
    def _update_operation_metrics(self, operation_type: DuplicateOperationType, is_duplicate: bool):
        """Update metrics for an operation type"""
        metrics = self.operation_metrics[operation_type]
        
        if is_duplicate:
            metrics.duplicate_executions += 1
        else:
            metrics.unique_executions += 1
        
        # Recalculate duplicate rate
        total = metrics.duplicate_executions + metrics.unique_executions
        if total > 0:
            metrics.duplicate_rate = metrics.duplicate_executions / total
        
        metrics.last_updated = datetime.now()
    
    async def _store_duplicate_event(self, event: DuplicateEvent):
        """Store duplicate event in Redis for analysis"""
        try:
            # Store individual event
            event_key = f"{self.REDIS_PREFIXES['events']}:{event.event_id}"
            await self.redis_client.setex(
                event_key,
                self.TTL_SETTINGS['event_history'],
                json.dumps(asdict(event), default=str)
            )
            
            # Add to operation type index
            type_index_key = f"{self.REDIS_PREFIXES['events']}:by_type:{event.operation_type.value}"
            await self.redis_client.lpush(type_index_key, event.event_id)
            await self.redis_client.expire(type_index_key, self.TTL_SETTINGS['event_history'])
            
            # Add to severity index
            severity_index_key = f"{self.REDIS_PREFIXES['events']}:by_severity:{event.detection_level.value}"
            await self.redis_client.lpush(severity_index_key, event.event_id)
            await self.redis_client.expire(severity_index_key, self.TTL_SETTINGS['event_history'])
            
            # Add to request index for correlation
            request_index_key = f"{self.REDIS_PREFIXES['events']}:by_request:{event.request_id}"
            await self.redis_client.lpush(request_index_key, event.event_id)
            await self.redis_client.expire(request_index_key, self.TTL_SETTINGS['event_history'])
            
        except Exception as e:
            logger.error(f"Error storing duplicate event: {e}")
    
    async def _duplicate_detector(self):
        """Background task to detect and analyze duplicate patterns"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Analyze recent duplicate events for patterns
                await self._analyze_duplicate_patterns()
                
                # Clean up old tracking data
                await self._cleanup_old_tracking_data()
                
            except Exception as e:
                logger.error(f"Error in duplicate detector: {e}")
    
    async def _analyze_duplicate_patterns(self):
        """Analyze duplicate events to identify patterns"""
        try:
            # Group events by operation type in last hour
            cutoff_time = datetime.now() - timedelta(hours=1)
            recent_events = [
                event for event in self.duplicate_events
                if event.duplicate_execution_time > cutoff_time
            ]
            
            if not recent_events:
                return
            
            # Group by operation type
            events_by_type = defaultdict(list)
            for event in recent_events:
                events_by_type[event.operation_type].append(event)
            
            # Analyze each operation type for patterns
            for operation_type, type_events in events_by_type.items():
                if len(type_events) < 3:  # Need at least 3 events for pattern
                    continue
                
                # Calculate pattern metrics
                frequency = len(type_events)  # events per hour
                time_gaps = [
                    event.time_between_duplicates for event in type_events
                ]
                average_gap = sum(time_gaps) / len(time_gaps)
                
                # Calculate impact score based on frequency and timing
                impact_score = min(1.0, (frequency / 20) + (1 / max(average_gap, 1)))
                
                # Generate recommendations based on pattern
                recommendations = self._generate_pattern_recommendations(
                    operation_type, frequency, average_gap, impact_score
                )
                
                # Create or update pattern
                pattern_id = f"{operation_type.value}_pattern"
                if pattern_id in self.detected_patterns:
                    pattern = self.detected_patterns[pattern_id]
                    pattern.frequency = frequency
                    pattern.average_gap = average_gap
                    pattern.impact_score = impact_score
                    pattern.last_detected = datetime.now()
                    pattern.affected_operations = [e.request_id for e in type_events]
                    pattern.recommendations = recommendations
                else:
                    pattern = DuplicatePattern(
                        pattern_id=pattern_id,
                        operation_type=operation_type,
                        frequency=frequency,
                        average_gap=average_gap,
                        impact_score=impact_score,
                        affected_operations=[e.request_id for e in type_events],
                        first_detected=min(e.duplicate_execution_time for e in type_events),
                        last_detected=datetime.now(),
                        recommendations=recommendations
                    )
                    
                    self.detected_patterns[pattern_id] = pattern
                    self.monitor_stats['total_patterns_identified'] += 1
                    self.monitor_stats['last_pattern_detection'] = datetime.now()
                
                # Store pattern in Redis
                await self._store_duplicate_pattern(pattern)
                
                # Log pattern detection
                logger.warning(
                    f"DUPLICATE PATTERN DETECTED: {operation_type.value} - "
                    f"Frequency: {frequency}/hour, Average Gap: {average_gap:.2f}s, "
                    f"Impact: {impact_score:.2f}"
                )
        
        except Exception as e:
            logger.error(f"Error analyzing duplicate patterns: {e}")
    
    def _generate_pattern_recommendations(
        self,
        operation_type: DuplicateOperationType,
        frequency: int,
        average_gap: float,
        impact_score: float
    ) -> List[str]:
        """Generate recommendations based on detected patterns"""
        recommendations = []
        
        # Generic recommendations for all operation types
        if average_gap < 5:
            recommendations.append("Implement operation deduplication with stronger hashing")
            recommendations.append("Add request-level circuit breaker to prevent rapid duplicates")
        
        if frequency > 10:
            recommendations.append("Review caller code for unnecessary retry logic")
            recommendations.append("Implement exponential backoff for operation retries")
        
        if impact_score > 0.7:
            recommendations.append("Priority fix required - high performance impact")
            recommendations.append("Consider operation throttling or rate limiting")
        
        # Operation-specific recommendations
        if operation_type == DuplicateOperationType.INTENT_ANALYSIS:
            recommendations.extend([
                "Cache intent analysis results with longer TTL",
                "Implement query similarity check before intent analysis",
                "Add request-level intent analysis state tracking"
            ])
        
        elif operation_type == DuplicateOperationType.TASK_PLANNING:
            recommendations.extend([
                "Improve task plan caching with query fingerprinting",
                "Add plan reuse detection for similar queries",
                "Implement planning result validation before duplicate execution"
            ])
        
        elif operation_type == DuplicateOperationType.QUERY_EMBEDDING:
            recommendations.extend([
                "Implement embedding cache with semantic similarity",
                "Add embedding fingerprint matching",
                "Cache embeddings at request level, not just query level"
            ])
        
        elif operation_type == DuplicateOperationType.BATCH_EXTRACTION:
            recommendations.extend([
                "Implement batch extraction state tracking",
                "Add extraction result fingerprinting",
                "Cache extraction results with longer TTL",
                "Implement extraction progress tracking to prevent re-runs"
            ])
        
        elif operation_type == DuplicateOperationType.LLM_CALL:
            recommendations.extend([
                "Implement LLM response caching with prompt fingerprinting",
                "Add LLM call deduplication at conversation level",
                "Cache LLM responses for identical prompts"
            ])
        
        return recommendations
    
    async def _store_duplicate_pattern(self, pattern: DuplicatePattern):
        """Store detected pattern in Redis"""
        try:
            pattern_key = f"{self.REDIS_PREFIXES['patterns']}:{pattern.pattern_id}"
            await self.redis_client.setex(
                pattern_key,
                self.TTL_SETTINGS['pattern_detection'],
                json.dumps(asdict(pattern), default=str)
            )
            
            # Add to global pattern index
            await self.redis_client.lpush(
                f"{self.REDIS_PREFIXES['patterns']}:index",
                pattern.pattern_id
            )
            await self.redis_client.expire(
                f"{self.REDIS_PREFIXES['patterns']}:index",
                self.TTL_SETTINGS['pattern_detection']
            )
            
        except Exception as e:
            logger.error(f"Error storing duplicate pattern: {e}")
    
    async def _pattern_analyzer(self):
        """Background task for advanced pattern analysis"""
        while True:
            try:
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
                # Analyze cross-operation correlations
                await self._analyze_operation_correlations()
                
                # Update system efficiency score
                await self._calculate_system_efficiency()
                
                # Generate performance insights
                await self._generate_performance_insights()
                
            except Exception as e:
                logger.error(f"Error in pattern analyzer: {e}")
    
    async def _analyze_operation_correlations(self):
        """Analyze correlations between different operation types"""
        try:
            # Group recent events by request_id
            cutoff_time = datetime.now() - timedelta(hours=1)
            recent_events = [
                event for event in self.duplicate_events
                if event.duplicate_execution_time > cutoff_time
            ]
            
            # Group by request ID
            events_by_request = defaultdict(list)
            for event in recent_events:
                events_by_request[event.request_id].append(event)
            
            # Find requests with multiple operation type duplicates
            correlated_requests = []
            for request_id, request_events in events_by_request.items():
                operation_types = {event.operation_type for event in request_events}
                if len(operation_types) >= 2:  # Multiple operation types duplicated
                    correlated_requests.append({
                        'request_id': request_id,
                        'operation_types': [op.value for op in operation_types],
                        'total_duplicates': len(request_events),
                        'time_span': max(e.duplicate_execution_time for e in request_events) - 
                                   min(e.duplicate_execution_time for e in request_events)
                    })
            
            # Store correlation analysis
            if correlated_requests:
                await self.redis_client.setex(
                    f"{self.REDIS_PREFIXES['patterns']}:correlations",
                    self.TTL_SETTINGS['pattern_detection'],
                    json.dumps({
                        'timestamp': datetime.now().isoformat(),
                        'correlated_requests': correlated_requests,
                        'total_correlated': len(correlated_requests)
                    }, default=str)
                )
                
                logger.warning(
                    f"CORRELATION DETECTED: {len(correlated_requests)} requests with "
                    f"multiple operation type duplicates"
                )
        
        except Exception as e:
            logger.error(f"Error analyzing operation correlations: {e}")
    
    async def _calculate_system_efficiency(self):
        """Calculate overall system efficiency score"""
        try:
            total_operations = sum(
                metrics.total_executions for metrics in self.operation_metrics.values()
            )
            total_duplicates = sum(
                metrics.duplicate_executions for metrics in self.operation_metrics.values()
            )
            
            if total_operations > 0:
                duplicate_rate = total_duplicates / total_operations
                # Efficiency score: 100 - (duplicate_rate * 100)
                efficiency = max(0.0, 100.0 - (duplicate_rate * 100))
                self.monitor_stats['system_efficiency_score'] = efficiency
                
                # Alert if efficiency drops below thresholds
                if efficiency < 50:
                    logger.critical(f"SYSTEM EFFICIENCY CRITICAL: {efficiency:.1f}% - High duplicate rate")
                elif efficiency < 70:
                    logger.error(f"SYSTEM EFFICIENCY LOW: {efficiency:.1f}% - Review duplicate patterns")
                elif efficiency < 85:
                    logger.warning(f"SYSTEM EFFICIENCY DEGRADED: {efficiency:.1f}% - Monitor for trends")
        
        except Exception as e:
            logger.error(f"Error calculating system efficiency: {e}")
    
    async def _generate_performance_insights(self):
        """Generate performance insights from collected data"""
        try:
            insights = []
            
            # Analyze operation efficiency
            for operation_type, metrics in self.operation_metrics.items():
                if metrics.total_executions > 10:  # Only analyze with sufficient data
                    
                    # High duplicate rate insight
                    if metrics.duplicate_rate > 0.3:
                        insights.append({
                            'type': 'high_duplicate_rate',
                            'operation': operation_type.value,
                            'duplicate_rate': metrics.duplicate_rate,
                            'recommendation': f"Review {operation_type.value} caching and deduplication logic",
                            'priority': 'high' if metrics.duplicate_rate > 0.5 else 'medium'
                        })
                    
                    # Slow operation insight
                    if metrics.average_execution_time > 5.0:
                        insights.append({
                            'type': 'slow_operation',
                            'operation': operation_type.value,
                            'avg_time': metrics.average_execution_time,
                            'recommendation': f"Optimize {operation_type.value} performance or increase caching",
                            'priority': 'high' if metrics.average_execution_time > 10.0 else 'medium'
                        })
                    
                    # Poor cache performance insight
                    if metrics.cache_hit_rate < 0.5 and metrics.total_executions > 20:
                        insights.append({
                            'type': 'poor_cache_performance',
                            'operation': operation_type.value,
                            'hit_rate': metrics.cache_hit_rate,
                            'recommendation': f"Review and improve {operation_type.value} caching strategy",
                            'priority': 'medium'
                        })
                    
                    # Frequent timeouts insight
                    if metrics.timeout_occurrences > metrics.total_executions * 0.1:
                        insights.append({
                            'type': 'frequent_timeouts',
                            'operation': operation_type.value,
                            'timeout_rate': metrics.timeout_occurrences / metrics.total_executions,
                            'recommendation': f"Review timeout configuration for {operation_type.value}",
                            'priority': 'high'
                        })
            
            # Store insights
            if insights:
                await self.redis_client.setex(
                    f"{self.REDIS_PREFIXES['metrics']}:insights",
                    self.TTL_SETTINGS['metrics_storage'],
                    json.dumps({
                        'timestamp': datetime.now().isoformat(),
                        'insights': insights,
                        'total_insights': len(insights)
                    }, default=str)
                )
        
        except Exception as e:
            logger.error(f"Error generating performance insights: {e}")
    
    async def _metrics_collector(self):
        """Background task to collect and store metrics"""
        while True:
            try:
                await asyncio.sleep(60)  # Collect every minute
                
                # Collect current metrics
                metrics_snapshot = {
                    'timestamp': datetime.now().isoformat(),
                    'operation_metrics': {},
                    'system_stats': dict(self.monitor_stats),
                    'active_operations': len(self.active_operations),
                    'tracked_hashes': len(self.operation_hashes)
                }
                
                # Add operation metrics
                for operation_type, metrics in self.operation_metrics.items():
                    metrics_snapshot['operation_metrics'][operation_type.value] = asdict(metrics)
                
                # Store metrics snapshot
                await self.redis_client.setex(
                    f"{self.REDIS_PREFIXES['metrics']}:current",
                    self.TTL_SETTINGS['dashboard_data'],
                    json.dumps(metrics_snapshot, default=str)
                )
                
                # Add to historical metrics
                await self.redis_client.lpush(
                    f"{self.REDIS_PREFIXES['metrics']}:history",
                    json.dumps(metrics_snapshot, default=str)
                )
                
                # Trim history to reasonable size
                await self.redis_client.ltrim(
                    f"{self.REDIS_PREFIXES['metrics']}:history",
                    0, 1440  # Keep 24 hours of minute-by-minute data
                )
                
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
    
    async def _dashboard_updater(self):
        """Background task to update dashboard data"""
        while True:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                
                # Prepare dashboard data
                dashboard_data = await self._prepare_dashboard_data()
                
                # Store dashboard data
                await self.redis_client.setex(
                    f"{self.REDIS_PREFIXES['dashboard']}:current",
                    self.TTL_SETTINGS['dashboard_data'],
                    json.dumps(dashboard_data, default=str)
                )
                
            except Exception as e:
                logger.error(f"Error updating dashboard: {e}")
    
    async def _prepare_dashboard_data(self) -> Dict[str, Any]:
        """Prepare comprehensive dashboard data"""
        # Get recent duplicate events (last hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        recent_events = [
            event for event in self.duplicate_events
            if event.duplicate_execution_time > cutoff_time
        ]
        
        # Operation type breakdown
        operation_breakdown = defaultdict(int)
        severity_breakdown = defaultdict(int)
        for event in recent_events:
            operation_breakdown[event.operation_type.value] += 1
            severity_breakdown[event.detection_level.value] += 1
        
        # Top problematic operations
        top_operations = sorted(
            self.operation_metrics.items(),
            key=lambda x: x[1].duplicate_rate * x[1].total_executions,
            reverse=True
        )[:5]
        
        # Active patterns
        active_patterns = [
            {
                'operation': pattern.operation_type.value,
                'frequency': pattern.frequency,
                'impact': pattern.impact_score,
                'last_detected': pattern.last_detected.isoformat()
            }
            for pattern in self.detected_patterns.values()
            if pattern.last_detected > cutoff_time
        ]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_operations_tracked': self.monitor_stats['total_operations_tracked'],
                'total_duplicates_detected': self.monitor_stats['total_duplicates_detected'],
                'system_efficiency_score': self.monitor_stats['system_efficiency_score'],
                'active_operations': len(self.active_operations),
                'recent_duplicates': len(recent_events)
            },
            'recent_activity': {
                'operation_breakdown': dict(operation_breakdown),
                'severity_breakdown': dict(severity_breakdown),
                'total_patterns_active': len(active_patterns)
            },
            'top_problematic_operations': [
                {
                    'operation': op_type.value,
                    'duplicate_rate': metrics.duplicate_rate,
                    'total_executions': metrics.total_executions,
                    'avg_execution_time': metrics.average_execution_time,
                    'impact_score': metrics.duplicate_rate * metrics.total_executions
                }
                for op_type, metrics in top_operations
            ],
            'active_patterns': active_patterns,
            'performance_alerts': [
                {
                    'operation': op_type.value,
                    'issue': 'high_duplicate_rate',
                    'value': metrics.duplicate_rate,
                    'threshold': 0.2
                }
                for op_type, metrics in self.operation_metrics.items()
                if metrics.duplicate_rate > 0.2 and metrics.total_executions > 5
            ],
            'recommendations': [
                rec for pattern in self.detected_patterns.values()
                for rec in pattern.recommendations
                if pattern.last_detected > cutoff_time
            ][:10]  # Top 10 recommendations
        }
    
    async def _cleanup_old_tracking_data(self):
        """Clean up old tracking data to prevent memory leaks"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=2)
            
            with self.operation_lock:
                # Clean up old operation hashes
                old_hashes = [
                    op_hash for op_hash, (_, timestamp) in self.operation_hashes.items()
                    if timestamp < cutoff_time
                ]
                
                for op_hash in old_hashes:
                    del self.operation_hashes[op_hash]
                
                # Clean up old active operations
                old_operations = [
                    op_id for op_id, timestamp in self.active_operations.items()
                    if timestamp < cutoff_time
                ]
                
                for op_id in old_operations:
                    del self.active_operations[op_id]
            
            logger.debug(f"Cleaned up {len(old_hashes)} old hashes and {len(old_operations)} old operations")
        
        except Exception as e:
            logger.error(f"Error cleaning up tracking data: {e}")
    
    async def get_duplicate_report(
        self,
        operation_type: Optional[DuplicateOperationType] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Generate comprehensive duplicate execution report
        
        Args:
            operation_type: Optional filter by operation type
            hours: Number of hours to analyze
            
        Returns:
            Detailed duplicate execution report
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter events
        events = [
            event for event in self.duplicate_events
            if event.duplicate_execution_time > cutoff_time
        ]
        
        if operation_type:
            events = [e for e in events if e.operation_type == operation_type]
        
        # Generate report
        report = {
            'report_period': f"{hours} hours",
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_duplicates': len(events),
                'unique_requests_affected': len({e.request_id for e in events}),
                'operation_types_affected': len({e.operation_type for e in events}),
                'severity_distribution': {
                    level.value: len([e for e in events if e.detection_level == level])
                    for level in DuplicateDetectionLevel
                }
            },
            'operation_analysis': {},
            'patterns': [asdict(p) for p in self.detected_patterns.values()],
            'recommendations': [],
            'system_health': {
                'efficiency_score': self.monitor_stats['system_efficiency_score'],
                'active_operations': len(self.active_operations),
                'monitoring_uptime': (datetime.now() - self.monitor_stats['monitoring_start']).total_seconds()
            }
        }
        
        # Analyze by operation type
        for op_type, metrics in self.operation_metrics.items():
            if not operation_type or operation_type == op_type:
                type_events = [e for e in events if e.operation_type == op_type]
                
                report['operation_analysis'][op_type.value] = {
                    'metrics': asdict(metrics),
                    'recent_duplicates': len(type_events),
                    'avg_gap_between_duplicates': (
                        sum(e.time_between_duplicates for e in type_events) / len(type_events)
                        if type_events else 0
                    ),
                    'fastest_duplicate': min(e.time_between_duplicates for e in type_events) if type_events else 0,
                    'events': [asdict(e) for e in type_events[-10:]]  # Last 10 events
                }
        
        # Aggregate recommendations
        all_recommendations = []
        for pattern in self.detected_patterns.values():
            all_recommendations.extend(pattern.recommendations)
        
        # Deduplicate and prioritize recommendations
        unique_recommendations = list(set(all_recommendations))
        report['recommendations'] = unique_recommendations[:15]  # Top 15
        
        return report
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time duplicate monitoring metrics"""
        try:
            # Get cached dashboard data
            cached_data = await self.redis_client.get(f"{self.REDIS_PREFIXES['dashboard']}:current")
            if cached_data:
                return json.loads(cached_data)
            
            # Generate fresh data if cache miss
            return await self._prepare_dashboard_data()
        
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_operation_health_status(self) -> Dict[str, Any]:
        """Get health status for all monitored operations"""
        health_data = {}
        
        for operation_type, metrics in self.operation_metrics.items():
            # Calculate health score for this operation
            health_score = 100.0
            
            # Deduct for duplicates
            health_score -= metrics.duplicate_rate * 50  # Max 50 points for duplicates
            
            # Deduct for timeouts
            if metrics.total_executions > 0:
                timeout_rate = metrics.timeout_occurrences / metrics.total_executions
                health_score -= timeout_rate * 30  # Max 30 points for timeouts
            
            # Deduct for circuit breaker activations
            if metrics.total_executions > 0:
                cb_rate = metrics.circuit_breaker_activations / metrics.total_executions
                health_score -= cb_rate * 20  # Max 20 points for circuit breakers
            
            health_score = max(0, health_score)
            
            # Determine status
            if health_score >= 85:
                status = "healthy"
            elif health_score >= 70:
                status = "degraded"
            elif health_score >= 50:
                status = "unhealthy"
            else:
                status = "critical"
            
            health_data[operation_type.value] = {
                'status': status,
                'health_score': health_score,
                'metrics': asdict(metrics),
                'issues': []
            }
            
            # Add specific issues
            if metrics.duplicate_rate > 0.2:
                health_data[operation_type.value]['issues'].append(
                    f"High duplicate rate: {metrics.duplicate_rate:.1%}"
                )
            
            if metrics.average_execution_time > 5.0:
                health_data[operation_type.value]['issues'].append(
                    f"Slow execution: {metrics.average_execution_time:.2f}s avg"
                )
            
            if metrics.timeout_occurrences > 0:
                health_data[operation_type.value]['issues'].append(
                    f"Timeouts occurring: {metrics.timeout_occurrences} times"
                )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_health': min(data['health_score'] for data in health_data.values()) if health_data else 100,
            'operations': health_data,
            'system_stats': dict(self.monitor_stats)
        }


# Global instance
duplicate_execution_monitor = DuplicateExecutionMonitor()


def get_duplicate_execution_monitor() -> DuplicateExecutionMonitor:
    """Get the global duplicate execution monitor instance"""
    return duplicate_execution_monitor


# Convenience functions for common operations
def track_operation(
    operation_type: DuplicateOperationType,
    request_id: str,
    conversation_id: Optional[str] = None,
    notebook_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> str:
    """Convenience function to track operation start"""
    return duplicate_execution_monitor.track_operation_start(
        operation_type, request_id, conversation_id, notebook_id, context
    )


def end_operation(
    operation_id: str,
    success: bool = True,
    execution_time: Optional[float] = None,
    cache_hit: Optional[bool] = None,
    timeout_occurred: bool = False,
    circuit_breaker_activated: bool = False
):
    """Convenience function to track operation end"""
    duplicate_execution_monitor.track_operation_end(
        operation_id, success, execution_time, cache_hit, timeout_occurred, circuit_breaker_activated
    )