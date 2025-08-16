"""
Performance Monitor

Real-time performance monitoring for the radiating system with metrics tracking,
bottleneck detection, and performance reporting.
"""

import logging
import time
import psutil
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import json
import threading

from app.core.redis_client import get_redis_client
from app.core.radiating_settings_cache import get_radiating_settings

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics"""
    QUERY_TIME = "query_time"
    TRAVERSAL_TIME = "traversal_time"
    CACHE_HIT_RATE = "cache_hit_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    QUEUE_SIZE = "queue_size"
    ACTIVE_CONNECTIONS = "active_connections"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert"""
    id: str
    metric_type: MetricType
    severity: AlertSeverity
    threshold: float
    actual_value: float
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class BottleneckInfo:
    """Information about detected bottleneck"""
    component: str
    metric: MetricType
    severity: str
    impact: float  # 0-1 scale
    description: str
    recommendations: List[str]
    detected_at: datetime


class MetricCollector:
    """Collects and stores performance metrics"""
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize metric collector
        
        Args:
            window_size: Size of sliding window for metrics
        """
        self.window_size = window_size
        self.metrics: Dict[MetricType, Deque[MetricPoint]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.lock = threading.Lock()
    
    def record(
        self,
        metric_type: MetricType,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a metric value"""
        with self.lock:
            point = MetricPoint(
                timestamp=datetime.now(),
                value=value,
                metadata=metadata or {}
            )
            self.metrics[metric_type].append(point)
    
    def get_recent(
        self,
        metric_type: MetricType,
        duration: Optional[timedelta] = None
    ) -> List[MetricPoint]:
        """Get recent metric points"""
        with self.lock:
            points = list(self.metrics.get(metric_type, []))
            
            if duration:
                cutoff = datetime.now() - duration
                points = [p for p in points if p.timestamp > cutoff]
            
            return points
    
    def get_statistics(
        self,
        metric_type: MetricType,
        duration: Optional[timedelta] = None
    ) -> Dict[str, float]:
        """Get statistics for a metric"""
        points = self.get_recent(metric_type, duration)
        
        if not points:
            return {
                'count': 0,
                'mean': 0,
                'min': 0,
                'max': 0,
                'std': 0,
                'p50': 0,
                'p95': 0,
                'p99': 0
            }
        
        values = [p.value for p in points]
        values.sort()
        
        import numpy as np
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'min': np.min(values),
            'max': np.max(values),
            'std': np.std(values),
            'p50': np.percentile(values, 50),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }


class PerformanceMonitor:
    """
    Monitors radiating system performance with real-time metrics,
    bottleneck detection, and alerting.
    """
    
    # Alert thresholds
    ALERT_THRESHOLDS = {
        MetricType.QUERY_TIME: {
            AlertSeverity.WARNING: 1.0,      # 1 second
            AlertSeverity.ERROR: 5.0,        # 5 seconds
            AlertSeverity.CRITICAL: 10.0     # 10 seconds
        },
        MetricType.CACHE_HIT_RATE: {
            AlertSeverity.WARNING: 0.5,      # 50% hit rate
            AlertSeverity.ERROR: 0.3,        # 30% hit rate
            AlertSeverity.CRITICAL: 0.1      # 10% hit rate
        },
        MetricType.MEMORY_USAGE: {
            AlertSeverity.WARNING: 70,       # 70% memory
            AlertSeverity.ERROR: 85,         # 85% memory
            AlertSeverity.CRITICAL: 95       # 95% memory
        },
        MetricType.CPU_USAGE: {
            AlertSeverity.WARNING: 70,       # 70% CPU
            AlertSeverity.ERROR: 85,         # 85% CPU
            AlertSeverity.CRITICAL: 95       # 95% CPU
        },
        MetricType.ERROR_RATE: {
            AlertSeverity.WARNING: 0.01,     # 1% errors
            AlertSeverity.ERROR: 0.05,       # 5% errors
            AlertSeverity.CRITICAL: 0.1      # 10% errors
        }
    }
    
    def __init__(self):
        """Initialize PerformanceMonitor"""
        self.redis_client = get_redis_client()
        self.collector = MetricCollector()
        
        # System info
        self.process = psutil.Process()
        self.cpu_count = psutil.cpu_count()
        
        # Active monitoring
        self.active_operations: Dict[str, datetime] = {}
        self.operation_lock = threading.Lock()
        
        # Alerts
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: List[PerformanceAlert] = []
        
        # Bottleneck detection
        self.detected_bottlenecks: List[BottleneckInfo] = []
        
        # Statistics
        self.stats = {
            'monitoring_start': datetime.now(),
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_processing_time': 0.0
        }
        
        # Start background monitoring
        asyncio.create_task(self._background_monitor())
        asyncio.create_task(self._bottleneck_detector())
        asyncio.create_task(self._alert_manager())
    
    def start_operation(self, operation_id: str) -> None:
        """Mark the start of an operation"""
        with self.operation_lock:
            self.active_operations[operation_id] = datetime.now()
            self.stats['total_operations'] += 1
    
    def end_operation(
        self,
        operation_id: str,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Mark the end of an operation
        
        Returns:
            Operation duration in seconds
        """
        with self.operation_lock:
            if operation_id not in self.active_operations:
                logger.warning(f"Unknown operation: {operation_id}")
                return 0.0
            
            start_time = self.active_operations.pop(operation_id)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            if success:
                self.stats['successful_operations'] += 1
            else:
                self.stats['failed_operations'] += 1
            
            self.stats['total_processing_time'] += duration
            
            # Record metrics
            self.collector.record(
                MetricType.LATENCY,
                duration,
                metadata
            )
            
            return duration
    
    def record_query_time(self, query: str, duration: float):
        """Record query execution time"""
        self.collector.record(
            MetricType.QUERY_TIME,
            duration,
            {'query': query[:100]}  # Truncate query for storage
        )
    
    def record_traversal_time(self, depth: int, duration: float, nodes_visited: int):
        """Record traversal time"""
        self.collector.record(
            MetricType.TRAVERSAL_TIME,
            duration,
            {
                'depth': depth,
                'nodes_visited': nodes_visited,
                'nodes_per_second': nodes_visited / duration if duration > 0 else 0
            }
        )
    
    def record_cache_hit(self, hit: bool, cache_tier: str):
        """Record cache hit/miss"""
        self.collector.record(
            MetricType.CACHE_HIT_RATE,
            1.0 if hit else 0.0,
            {'tier': cache_tier}
        )
    
    def record_error(self, error_type: str, component: str):
        """Record an error occurrence"""
        self.collector.record(
            MetricType.ERROR_RATE,
            1.0,
            {
                'error_type': error_type,
                'component': component
            }
        )
    
    async def _background_monitor(self):
        """Background monitoring task"""
        while True:
            try:
                # Collect system metrics
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                memory_percent = self.process.memory_percent()
                
                # Record system metrics
                self.collector.record(MetricType.CPU_USAGE, cpu_percent)
                self.collector.record(
                    MetricType.MEMORY_USAGE,
                    memory_percent,
                    {
                        'rss_mb': memory_info.rss / 1024 / 1024,
                        'vms_mb': memory_info.vms / 1024 / 1024
                    }
                )
                
                # Calculate throughput
                with self.operation_lock:
                    active_count = len(self.active_operations)
                    self.collector.record(
                        MetricType.QUEUE_SIZE,
                        active_count
                    )
                
                # Calculate recent throughput
                recent_ops = self.collector.get_recent(
                    MetricType.LATENCY,
                    timedelta(minutes=1)
                )
                throughput = len(recent_ops) / 60.0  # Operations per second
                self.collector.record(MetricType.THROUGHPUT, throughput)
                
                # Store metrics in Redis
                await self._store_metrics()
                
                # Sleep for monitoring interval
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in background monitor: {e}")
                await asyncio.sleep(5)
    
    async def _bottleneck_detector(self):
        """Detect performance bottlenecks"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                bottlenecks = []
                
                # Check query performance
                query_stats = self.collector.get_statistics(
                    MetricType.QUERY_TIME,
                    timedelta(minutes=5)
                )
                
                if query_stats['p95'] > 5.0:  # 95th percentile > 5 seconds
                    bottlenecks.append(BottleneckInfo(
                        component="Query Engine",
                        metric=MetricType.QUERY_TIME,
                        severity="high",
                        impact=min(1.0, query_stats['p95'] / 10.0),
                        description=f"Query p95 latency is {query_stats['p95']:.2f}s",
                        recommendations=[
                            "Review and optimize slow queries",
                            "Add or optimize database indexes",
                            "Consider query result caching",
                            "Enable query parallelization"
                        ],
                        detected_at=datetime.now()
                    ))
                
                # Check cache performance
                cache_stats = self.collector.get_statistics(
                    MetricType.CACHE_HIT_RATE,
                    timedelta(minutes=5)
                )
                
                if cache_stats['mean'] < 0.5:  # Less than 50% hit rate
                    bottlenecks.append(BottleneckInfo(
                        component="Cache System",
                        metric=MetricType.CACHE_HIT_RATE,
                        severity="medium",
                        impact=1.0 - cache_stats['mean'],
                        description=f"Cache hit rate is only {cache_stats['mean']*100:.1f}%",
                        recommendations=[
                            "Increase cache size",
                            "Implement cache warming",
                            "Review cache eviction policy",
                            "Optimize cache key strategy"
                        ],
                        detected_at=datetime.now()
                    ))
                
                # Check memory usage
                memory_stats = self.collector.get_statistics(
                    MetricType.MEMORY_USAGE,
                    timedelta(minutes=5)
                )
                
                if memory_stats['mean'] > 80:  # Over 80% memory usage
                    bottlenecks.append(BottleneckInfo(
                        component="Memory",
                        metric=MetricType.MEMORY_USAGE,
                        severity="high",
                        impact=memory_stats['mean'] / 100.0,
                        description=f"Memory usage at {memory_stats['mean']:.1f}%",
                        recommendations=[
                            "Reduce batch sizes",
                            "Implement memory-efficient algorithms",
                            "Increase system memory",
                            "Enable memory profiling"
                        ],
                        detected_at=datetime.now()
                    ))
                
                # Check CPU usage
                cpu_stats = self.collector.get_statistics(
                    MetricType.CPU_USAGE,
                    timedelta(minutes=5)
                )
                
                if cpu_stats['mean'] > 80:  # Over 80% CPU usage
                    bottlenecks.append(BottleneckInfo(
                        component="CPU",
                        metric=MetricType.CPU_USAGE,
                        severity="high",
                        impact=cpu_stats['mean'] / 100.0,
                        description=f"CPU usage at {cpu_stats['mean']:.1f}%",
                        recommendations=[
                            "Optimize algorithms",
                            "Enable parallel processing",
                            "Scale horizontally",
                            "Profile CPU hotspots"
                        ],
                        detected_at=datetime.now()
                    ))
                
                # Store detected bottlenecks
                if bottlenecks:
                    self.detected_bottlenecks.extend(bottlenecks)
                    # Keep only recent bottlenecks
                    cutoff = datetime.now() - timedelta(hours=1)
                    self.detected_bottlenecks = [
                        b for b in self.detected_bottlenecks
                        if b.detected_at > cutoff
                    ]
                    
                    # Log bottlenecks
                    for bottleneck in bottlenecks:
                        logger.warning(
                            f"Bottleneck detected in {bottleneck.component}: "
                            f"{bottleneck.description}"
                        )
                
            except Exception as e:
                logger.error(f"Error in bottleneck detector: {e}")
    
    async def _alert_manager(self):
        """Manage performance alerts"""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Check metrics against thresholds
                for metric_type, thresholds in self.ALERT_THRESHOLDS.items():
                    stats = self.collector.get_statistics(
                        metric_type,
                        timedelta(minutes=1)
                    )
                    
                    if stats['count'] == 0:
                        continue
                    
                    current_value = stats['mean']
                    
                    # Determine alert severity
                    triggered_severity = None
                    triggered_threshold = None
                    
                    for severity in [AlertSeverity.CRITICAL, AlertSeverity.ERROR, AlertSeverity.WARNING]:
                        if severity in thresholds:
                            threshold = thresholds[severity]
                            
                            # For cache hit rate, lower is worse
                            if metric_type == MetricType.CACHE_HIT_RATE:
                                if current_value < threshold:
                                    triggered_severity = severity
                                    triggered_threshold = threshold
                                    break
                            else:
                                if current_value > threshold:
                                    triggered_severity = severity
                                    triggered_threshold = threshold
                                    break
                    
                    # Create or update alert
                    if triggered_severity:
                        alert_id = f"{metric_type.value}_{triggered_severity.value}"
                        
                        if alert_id not in self.active_alerts:
                            alert = PerformanceAlert(
                                id=alert_id,
                                metric_type=metric_type,
                                severity=triggered_severity,
                                threshold=triggered_threshold,
                                actual_value=current_value,
                                message=f"{metric_type.value} alert: {current_value:.2f} exceeds threshold {triggered_threshold:.2f}",
                                timestamp=datetime.now()
                            )
                            
                            self.active_alerts[alert_id] = alert
                            self.alert_history.append(alert)
                            
                            logger.warning(f"Performance alert: {alert.message}")
                    else:
                        # Check if we can resolve any alerts
                        for alert_id in list(self.active_alerts.keys()):
                            if alert_id.startswith(metric_type.value):
                                alert = self.active_alerts[alert_id]
                                alert.resolved = True
                                alert.resolution_time = datetime.now()
                                del self.active_alerts[alert_id]
                                
                                logger.info(f"Performance alert resolved: {alert_id}")
                
            except Exception as e:
                logger.error(f"Error in alert manager: {e}")
    
    async def _store_metrics(self):
        """Store metrics in Redis for persistence"""
        try:
            # Prepare metrics data
            metrics_data = {}
            
            for metric_type in MetricType:
                stats = self.collector.get_statistics(metric_type)
                if stats['count'] > 0:
                    metrics_data[metric_type.value] = stats
            
            # Add system stats
            metrics_data['system'] = {
                'uptime': (datetime.now() - self.stats['monitoring_start']).total_seconds(),
                'total_operations': self.stats['total_operations'],
                'successful_operations': self.stats['successful_operations'],
                'failed_operations': self.stats['failed_operations'],
                'avg_processing_time': (
                    self.stats['total_processing_time'] / self.stats['total_operations']
                    if self.stats['total_operations'] > 0 else 0
                )
            }
            
            # Add bottleneck info
            metrics_data['bottlenecks'] = [
                {
                    'component': b.component,
                    'metric': b.metric.value,
                    'severity': b.severity,
                    'impact': b.impact,
                    'description': b.description,
                    'detected_at': b.detected_at.isoformat()
                }
                for b in self.detected_bottlenecks[-10:]  # Last 10 bottlenecks
            ]
            
            # Add active alerts
            metrics_data['alerts'] = [
                {
                    'id': a.id,
                    'metric': a.metric_type.value,
                    'severity': a.severity.value,
                    'threshold': a.threshold,
                    'actual_value': a.actual_value,
                    'message': a.message,
                    'timestamp': a.timestamp.isoformat()
                }
                for a in self.active_alerts.values()
            ]
            
            # Store in Redis
            await self.redis_client.setex(
                "radiating:performance:metrics",
                60,  # 1 minute TTL
                json.dumps(metrics_data, default=str)
            )
            
            # Store historical data
            await self.redis_client.lpush(
                "radiating:performance:history",
                json.dumps({
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics_data
                }, default=str)
            )
            
            # Trim history to last 1000 entries
            await self.redis_client.ltrim("radiating:performance:history", 0, 999)
            
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        metrics = {}
        
        for metric_type in MetricType:
            stats = self.collector.get_statistics(metric_type)
            if stats['count'] > 0:
                metrics[metric_type.value] = stats
        
        return metrics
    
    def get_bottlenecks(self) -> List[BottleneckInfo]:
        """Get detected bottlenecks"""
        return self.detected_bottlenecks
    
    def get_alerts(self) -> List[PerformanceAlert]:
        """Get active alerts"""
        return list(self.active_alerts.values())
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        # Calculate health score (0-100)
        health_score = 100.0
        
        # Deduct for active alerts
        for alert in self.active_alerts.values():
            if alert.severity == AlertSeverity.CRITICAL:
                health_score -= 30
            elif alert.severity == AlertSeverity.ERROR:
                health_score -= 20
            elif alert.severity == AlertSeverity.WARNING:
                health_score -= 10
        
        # Deduct for bottlenecks
        for bottleneck in self.detected_bottlenecks:
            health_score -= bottleneck.impact * 20
        
        health_score = max(0, health_score)
        
        # Determine status
        if health_score >= 80:
            status = "healthy"
        elif health_score >= 60:
            status = "degraded"
        elif health_score >= 40:
            status = "unhealthy"
        else:
            status = "critical"
        
        return {
            'status': status,
            'health_score': health_score,
            'active_alerts': len(self.active_alerts),
            'detected_bottlenecks': len(self.detected_bottlenecks),
            'uptime': (datetime.now() - self.stats['monitoring_start']).total_seconds(),
            'success_rate': (
                self.stats['successful_operations'] / self.stats['total_operations']
                if self.stats['total_operations'] > 0 else 1.0
            )
        }