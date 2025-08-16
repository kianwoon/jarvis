"""
Dashboard Metrics

Collects and prepares metrics for the radiating system monitoring dashboard.
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np

from app.core.redis_client import get_redis_client
from app.services.radiating.optimization.performance_monitor import PerformanceMonitor
from app.services.radiating.optimization.cache_strategy import CacheStrategy
from app.services.radiating.quality.quality_metrics import QualityMetrics
from app.services.radiating.quality.feedback_integrator import FeedbackIntegrator

logger = logging.getLogger(__name__)


class DashboardMetricType(Enum):
    """Types of dashboard metrics"""
    REAL_TIME = "real_time"
    HISTORICAL = "historical"
    AGGREGATED = "aggregated"
    ALERT = "alert"


@dataclass
class DashboardMetric:
    """Individual dashboard metric"""
    name: str
    value: Any
    unit: str
    type: DashboardMetricType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardData:
    """Complete dashboard data"""
    real_time_metrics: Dict[str, Any]
    historical_data: Dict[str, List[Dict]]
    alerts: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class DashboardMetrics:
    """
    Collects and prepares metrics for the radiating system dashboard,
    providing real-time statistics, historical data, and alerts.
    """
    
    # Metric collection intervals
    COLLECTION_INTERVALS = {
        'real_time': 5,      # 5 seconds
        'aggregated': 60,    # 1 minute
        'historical': 300    # 5 minutes
    }
    
    # Dashboard sections
    DASHBOARD_SECTIONS = [
        'performance',
        'quality',
        'cache',
        'feedback',
        'system'
    ]
    
    # Alert thresholds for dashboard
    ALERT_THRESHOLDS = {
        'response_time': 2.0,      # 2 seconds
        'error_rate': 0.05,        # 5%
        'cache_hit_rate': 0.5,     # 50%
        'quality_score': 0.6,      # 60%
        'cpu_usage': 80,           # 80%
        'memory_usage': 80         # 80%
    }
    
    def __init__(self):
        """Initialize DashboardMetrics"""
        self.redis_client = get_redis_client()
        
        # Component instances
        self.performance_monitor = PerformanceMonitor()
        self.cache_strategy = CacheStrategy()
        self.quality_metrics = QualityMetrics()
        self.feedback_integrator = FeedbackIntegrator()
        
        # Metric storage
        self.real_time_buffer: Dict[str, List[DashboardMetric]] = {
            section: [] for section in self.DASHBOARD_SECTIONS
        }
        
        self.historical_buffer: Dict[str, List[Dict]] = {
            section: [] for section in self.DASHBOARD_SECTIONS
        }
        
        # Active alerts
        self.active_alerts: List[Dict[str, Any]] = []
        
        # Statistics
        self.dashboard_stats = {
            'collections': 0,
            'alerts_triggered': 0,
            'last_update': None
        }
        
        # Start collection tasks
        asyncio.create_task(self._real_time_collector())
        asyncio.create_task(self._aggregated_collector())
        asyncio.create_task(self._historical_collector())
        asyncio.create_task(self._alert_monitor())
    
    async def get_dashboard_data(self) -> DashboardData:
        """
        Get complete dashboard data
        
        Returns:
            DashboardData with all metrics
        """
        # Collect real-time metrics
        real_time = await self._collect_real_time_metrics()
        
        # Get historical data
        historical = await self._get_historical_data()
        
        # Get active alerts
        alerts = self._get_active_alerts()
        
        # Get statistics
        statistics = await self._collect_statistics()
        
        dashboard_data = DashboardData(
            real_time_metrics=real_time,
            historical_data=historical,
            alerts=alerts,
            statistics=statistics
        )
        
        # Store in Redis for external access
        await self._store_dashboard_data(dashboard_data)
        
        return dashboard_data
    
    async def _collect_real_time_metrics(self) -> Dict[str, Any]:
        """Collect real-time metrics"""
        metrics = {}
        
        # Performance metrics
        perf_metrics = self.performance_monitor.get_current_metrics()
        metrics['performance'] = {
            'query_time': perf_metrics.get('query_time', {}).get('mean', 0),
            'throughput': perf_metrics.get('throughput', {}).get('mean', 0),
            'latency': perf_metrics.get('latency', {}).get('mean', 0),
            'error_rate': perf_metrics.get('error_rate', {}).get('mean', 0),
            'cpu_usage': perf_metrics.get('cpu_usage', {}).get('mean', 0),
            'memory_usage': perf_metrics.get('memory_usage', {}).get('mean', 0)
        }
        
        # Cache metrics
        cache_stats = self.cache_strategy.get_statistics()
        metrics['cache'] = {
            'memory_hit_rate': cache_stats['memory']['hit_rate'],
            'redis_hit_rate': cache_stats['redis']['hit_rate'],
            'total_hits': cache_stats['memory']['hits'] + cache_stats['redis']['hits'],
            'total_misses': cache_stats['memory']['misses'] + cache_stats['redis']['misses'],
            'evictions': cache_stats['memory']['evictions'],
            'avg_response_time': cache_stats['memory']['avg_response_time_ms']
        }
        
        # Quality metrics
        quality_stats = self.quality_metrics.get_statistics()
        metrics['quality'] = {
            'average_score': quality_stats['average_quality_score'],
            'current_level': quality_stats['quality_level'],
            'evaluations': quality_stats['evaluations_performed']
        }
        
        # Add current metrics
        if 'current_metrics' in quality_stats:
            for metric_name, metric_data in quality_stats['current_metrics'].items():
                metrics['quality'][f"{metric_name}_current"] = metric_data.get('current', 0)
                metrics['quality'][f"{metric_name}_target"] = metric_data.get('target', 0)
        
        # Feedback metrics
        feedback_stats = self.feedback_integrator.get_statistics()
        metrics['feedback'] = {
            'total_feedback': feedback_stats['total_feedback'],
            'positive_rate': feedback_stats['positive_rate'],
            'corrections_applied': feedback_stats['corrections_applied'],
            'weights_adjusted': feedback_stats['weights_adjusted']
        }
        
        # System health
        health = self.performance_monitor.get_health_status()
        metrics['system'] = {
            'status': health['status'],
            'health_score': health['health_score'],
            'uptime': health['uptime'],
            'success_rate': health['success_rate'],
            'active_alerts': health['active_alerts']
        }
        
        return metrics
    
    async def _get_historical_data(self, hours: int = 24) -> Dict[str, List[Dict]]:
        """Get historical data for charts"""
        historical = {}
        
        # Get from Redis
        for section in self.DASHBOARD_SECTIONS:
            key = f"radiating:dashboard:history:{section}"
            data = await self.redis_client.lrange(key, 0, -1)
            
            historical[section] = []
            for item in data[-288:]:  # Last 24 hours at 5-minute intervals
                try:
                    historical[section].append(json.loads(item))
                except:
                    continue
        
        # Add trend analysis
        for section, data in historical.items():
            if data:
                historical[f"{section}_trend"] = self._calculate_trend(data)
        
        return historical
    
    def _calculate_trend(self, data: List[Dict]) -> Dict[str, Any]:
        """Calculate trend from historical data"""
        if len(data) < 2:
            return {'direction': 'stable', 'change': 0}
        
        # Get values from last hour vs previous hour
        recent = data[-12:] if len(data) >= 12 else data
        older = data[-24:-12] if len(data) >= 24 else data[:len(recent)]
        
        # Calculate average values
        recent_values = [d.get('value', 0) for d in recent if 'value' in d]
        older_values = [d.get('value', 0) for d in older if 'value' in d]
        
        if not recent_values or not older_values:
            return {'direction': 'stable', 'change': 0}
        
        recent_avg = np.mean(recent_values)
        older_avg = np.mean(older_values)
        
        # Calculate trend
        if older_avg == 0:
            change = 0
        else:
            change = ((recent_avg - older_avg) / older_avg) * 100
        
        if change > 5:
            direction = 'increasing'
        elif change < -5:
            direction = 'decreasing'
        else:
            direction = 'stable'
        
        return {
            'direction': direction,
            'change': change,
            'recent_avg': recent_avg,
            'older_avg': older_avg
        }
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts for dashboard"""
        alerts = []
        
        # Get performance alerts
        perf_alerts = self.performance_monitor.get_alerts()
        for alert in perf_alerts:
            alerts.append({
                'id': alert.id,
                'type': 'performance',
                'severity': alert.severity.value,
                'metric': alert.metric_type.value,
                'message': alert.message,
                'value': alert.actual_value,
                'threshold': alert.threshold,
                'timestamp': alert.timestamp.isoformat()
            })
        
        # Check custom thresholds
        metrics = asyncio.run(self._collect_real_time_metrics())
        
        # Response time alert
        if metrics['performance']['latency'] > self.ALERT_THRESHOLDS['response_time']:
            alerts.append({
                'id': 'response_time_high',
                'type': 'performance',
                'severity': 'warning',
                'metric': 'response_time',
                'message': f"Response time is {metrics['performance']['latency']:.2f}s",
                'value': metrics['performance']['latency'],
                'threshold': self.ALERT_THRESHOLDS['response_time'],
                'timestamp': datetime.now().isoformat()
            })
        
        # Cache hit rate alert
        if metrics['cache']['memory_hit_rate'] < self.ALERT_THRESHOLDS['cache_hit_rate']:
            alerts.append({
                'id': 'cache_hit_low',
                'type': 'cache',
                'severity': 'warning',
                'metric': 'cache_hit_rate',
                'message': f"Cache hit rate is {metrics['cache']['memory_hit_rate']:.1%}",
                'value': metrics['cache']['memory_hit_rate'],
                'threshold': self.ALERT_THRESHOLDS['cache_hit_rate'],
                'timestamp': datetime.now().isoformat()
            })
        
        # Quality score alert
        if metrics['quality']['average_score'] < self.ALERT_THRESHOLDS['quality_score']:
            alerts.append({
                'id': 'quality_low',
                'type': 'quality',
                'severity': 'warning',
                'metric': 'quality_score',
                'message': f"Quality score is {metrics['quality']['average_score']:.2f}",
                'value': metrics['quality']['average_score'],
                'threshold': self.ALERT_THRESHOLDS['quality_score'],
                'timestamp': datetime.now().isoformat()
            })
        
        # Sort by severity and timestamp
        severity_order = {'critical': 0, 'error': 1, 'warning': 2, 'info': 3}
        alerts.sort(key=lambda x: (severity_order.get(x['severity'], 99), x['timestamp']))
        
        return alerts
    
    async def _collect_statistics(self) -> Dict[str, Any]:
        """Collect overall statistics"""
        stats = {
            'last_update': datetime.now().isoformat(),
            'collections': self.dashboard_stats['collections']
        }
        
        # Add component statistics
        stats['performance'] = self.performance_monitor.get_statistics()
        stats['cache'] = self.cache_strategy.get_statistics()
        stats['quality'] = self.quality_metrics.get_statistics()
        stats['feedback'] = self.feedback_integrator.get_statistics()
        
        # Calculate aggregates
        stats['summary'] = {
            'total_operations': stats['performance'].get('total_operations', 0),
            'total_cache_operations': (
                stats['cache']['memory']['hits'] +
                stats['cache']['memory']['misses']
            ),
            'total_quality_evaluations': stats['quality']['evaluations_performed'],
            'total_feedback_collected': stats['feedback']['total_feedback']
        }
        
        return stats
    
    async def _store_dashboard_data(self, data: DashboardData):
        """Store dashboard data in Redis"""
        try:
            # Store current dashboard data
            await self.redis_client.setex(
                "radiating:dashboard:current",
                60,  # 1 minute TTL
                json.dumps({
                    'real_time': data.real_time_metrics,
                    'alerts': data.alerts,
                    'statistics': data.statistics,
                    'timestamp': data.timestamp.isoformat()
                }, default=str)
            )
            
            # Store historical snapshot
            for section, metrics in data.real_time_metrics.items():
                snapshot = {
                    'timestamp': data.timestamp.isoformat(),
                    'metrics': metrics
                }
                
                await self.redis_client.lpush(
                    f"radiating:dashboard:history:{section}",
                    json.dumps(snapshot)
                )
                
                # Trim to keep only recent history
                await self.redis_client.ltrim(
                    f"radiating:dashboard:history:{section}",
                    0,
                    288  # Keep 24 hours at 5-minute intervals
                )
        
        except Exception as e:
            logger.error(f"Error storing dashboard data: {e}")
    
    async def _real_time_collector(self):
        """Background task to collect real-time metrics"""
        while True:
            try:
                await asyncio.sleep(self.COLLECTION_INTERVALS['real_time'])
                
                # Collect metrics
                metrics = await self._collect_real_time_metrics()
                
                # Store in buffer
                for section, section_metrics in metrics.items():
                    for metric_name, value in section_metrics.items():
                        metric = DashboardMetric(
                            name=metric_name,
                            value=value,
                            unit=self._get_metric_unit(metric_name),
                            type=DashboardMetricType.REAL_TIME
                        )
                        
                        self.real_time_buffer[section].append(metric)
                        
                        # Keep only recent metrics
                        self.real_time_buffer[section] = self.real_time_buffer[section][-100:]
                
                self.dashboard_stats['collections'] += 1
                self.dashboard_stats['last_update'] = datetime.now()
                
            except Exception as e:
                logger.error(f"Error in real-time collector: {e}")
    
    async def _aggregated_collector(self):
        """Background task to collect aggregated metrics"""
        while True:
            try:
                await asyncio.sleep(self.COLLECTION_INTERVALS['aggregated'])
                
                # Aggregate real-time metrics
                for section in self.DASHBOARD_SECTIONS:
                    if self.real_time_buffer[section]:
                        # Calculate aggregates
                        values_by_metric = {}
                        
                        for metric in self.real_time_buffer[section]:
                            if metric.name not in values_by_metric:
                                values_by_metric[metric.name] = []
                            values_by_metric[metric.name].append(metric.value)
                        
                        # Store aggregates
                        aggregates = {}
                        for metric_name, values in values_by_metric.items():
                            if values and all(isinstance(v, (int, float)) for v in values):
                                aggregates[metric_name] = {
                                    'mean': np.mean(values),
                                    'min': np.min(values),
                                    'max': np.max(values),
                                    'std': np.std(values)
                                }
                        
                        # Store in Redis
                        await self.redis_client.setex(
                            f"radiating:dashboard:aggregated:{section}",
                            300,  # 5 minute TTL
                            json.dumps(aggregates, default=str)
                        )
                
            except Exception as e:
                logger.error(f"Error in aggregated collector: {e}")
    
    async def _historical_collector(self):
        """Background task to collect historical data"""
        while True:
            try:
                await asyncio.sleep(self.COLLECTION_INTERVALS['historical'])
                
                # Get current dashboard data
                data = await self.get_dashboard_data()
                
                # Historical data is already stored in get_dashboard_data
                
            except Exception as e:
                logger.error(f"Error in historical collector: {e}")
    
    async def _alert_monitor(self):
        """Background task to monitor alerts"""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Get current alerts
                alerts = self._get_active_alerts()
                
                # Check for new alerts
                current_alert_ids = {a['id'] for a in alerts}
                previous_alert_ids = {a['id'] for a in self.active_alerts}
                
                new_alerts = current_alert_ids - previous_alert_ids
                
                if new_alerts:
                    self.dashboard_stats['alerts_triggered'] += len(new_alerts)
                    
                    # Log new alerts
                    for alert in alerts:
                        if alert['id'] in new_alerts:
                            logger.warning(f"Dashboard alert: {alert['message']}")
                
                # Update active alerts
                self.active_alerts = alerts
                
            except Exception as e:
                logger.error(f"Error in alert monitor: {e}")
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get unit for metric"""
        units = {
            'query_time': 'seconds',
            'throughput': 'ops/sec',
            'latency': 'seconds',
            'error_rate': 'percentage',
            'cpu_usage': 'percentage',
            'memory_usage': 'percentage',
            'hit_rate': 'percentage',
            'response_time': 'milliseconds',
            'quality_score': 'score',
            'feedback': 'count',
            'uptime': 'seconds'
        }
        
        for key, unit in units.items():
            if key in metric_name:
                return unit
        
        return 'value'
    
    async def get_metric_history(
        self,
        metric_name: str,
        section: str,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get history for a specific metric
        
        Args:
            metric_name: Name of the metric
            section: Dashboard section
            hours: Number of hours of history
            
        Returns:
            List of historical data points
        """
        key = f"radiating:dashboard:history:{section}"
        data = await self.redis_client.lrange(key, 0, -1)
        
        history = []
        cutoff = datetime.now() - timedelta(hours=hours)
        
        for item in data:
            try:
                point = json.loads(item)
                timestamp = datetime.fromisoformat(point['timestamp'])
                
                if timestamp > cutoff:
                    if metric_name in point.get('metrics', {}):
                        history.append({
                            'timestamp': timestamp.isoformat(),
                            'value': point['metrics'][metric_name]
                        })
            except:
                continue
        
        return history
    
    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration"""
        return {
            'sections': self.DASHBOARD_SECTIONS,
            'update_intervals': self.COLLECTION_INTERVALS,
            'alert_thresholds': self.ALERT_THRESHOLDS,
            'metrics': {
                'performance': [
                    'query_time', 'throughput', 'latency',
                    'error_rate', 'cpu_usage', 'memory_usage'
                ],
                'cache': [
                    'memory_hit_rate', 'redis_hit_rate',
                    'total_hits', 'evictions', 'avg_response_time'
                ],
                'quality': [
                    'average_score', 'precision', 'recall',
                    'coverage', 'diversity', 'coherence'
                ],
                'feedback': [
                    'total_feedback', 'positive_rate',
                    'corrections_applied', 'weights_adjusted'
                ],
                'system': [
                    'status', 'health_score', 'uptime',
                    'success_rate', 'active_alerts'
                ]
            }
        }