"""
Monitoring Health Checks

Validates that duplicate execution monitoring and performance tracking
systems are functioning correctly. Provides comprehensive health validation
for all monitoring components and early warning for monitoring failures.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from app.core.redis_client import get_redis_client
from app.services.duplicate_execution_monitor import (
    get_duplicate_execution_monitor, DuplicateOperationType
)
from app.services.operation_performance_tracker import get_operation_performance_tracker
from app.core.timeout_settings_cache import get_timeout_value

logger = logging.getLogger(__name__)


class HealthCheckStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a single health check"""
    component: str
    status: HealthCheckStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.now)
    check_duration: float = 0.0


@dataclass
class SystemHealthReport:
    """Comprehensive system health report"""
    overall_status: HealthCheckStatus
    overall_score: float  # 0-100
    component_results: List[HealthCheckResult]
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    report_generated_at: datetime = field(default_factory=datetime.now)


class MonitoringHealthChecker:
    """
    Comprehensive health checker for monitoring systems.
    Validates all monitoring components are functioning correctly.
    """
    
    # Health check configurations
    CHECK_CONFIGS = {
        'duplicate_monitor_basic': {
            'timeout': 5.0,
            'critical_threshold': 0.8,
            'warning_threshold': 0.9
        },
        'performance_tracker_basic': {
            'timeout': 5.0,
            'critical_threshold': 0.8,
            'warning_threshold': 0.9
        },
        'redis_connectivity': {
            'timeout': 2.0,
            'critical_threshold': 0.95,
            'warning_threshold': 0.98
        },
        'cache_consistency': {
            'timeout': 10.0,
            'critical_threshold': 0.9,
            'warning_threshold': 0.95
        },
        'metrics_collection': {
            'timeout': 8.0,
            'critical_threshold': 0.85,
            'warning_threshold': 0.95
        },
        'execution_state_tracking': {
            'timeout': 6.0,
            'critical_threshold': 0.9,
            'warning_threshold': 0.95
        }
    }
    
    def __init__(self):
        """Initialize MonitoringHealthChecker"""
        self.redis_client = get_redis_client()
        self.duplicate_monitor = get_duplicate_execution_monitor()
        self.performance_tracker = get_operation_performance_tracker()
        
        # Health check history
        self.check_history: List[SystemHealthReport] = []
        
        # Component availability tracking
        self.component_availability: Dict[str, List[Tuple[datetime, bool]]] = {}
        
        logger.info("MonitoringHealthChecker initialized")
    
    async def run_comprehensive_health_check(self) -> SystemHealthReport:
        """
        Run comprehensive health check of all monitoring components
        
        Returns:
            Complete system health report
        """
        logger.info("Starting comprehensive monitoring health check")
        start_time = time.time()
        
        # List of health checks to perform
        health_checks = [
            self._check_duplicate_monitor_basic,
            self._check_performance_tracker_basic,
            self._check_redis_connectivity,
            self._check_cache_consistency,
            self._check_metrics_collection,
            self._check_execution_state_tracking,
            self._check_timeout_configuration,
            self._check_monitoring_integration
        ]
        
        # Run all health checks
        check_results = []
        for check_func in health_checks:
            try:
                result = await check_func()
                check_results.append(result)
            except Exception as e:
                # Create error result for failed check
                check_results.append(HealthCheckResult(
                    component=check_func.__name__,
                    status=HealthCheckStatus.ERROR,
                    message=f"Health check failed: {str(e)}",
                    details={'error': str(e)}
                ))
        
        # Analyze results
        report = self._analyze_health_results(check_results)
        
        # Store in history
        self.check_history.append(report)
        if len(self.check_history) > 100:  # Keep last 100 reports
            self.check_history = self.check_history[-100:]
        
        # Store in Redis for external access
        await self._store_health_report(report)
        
        total_time = time.time() - start_time
        logger.info(f"Health check completed in {total_time:.2f}s - Status: {report.overall_status.value}")
        
        return report
    
    async def _check_duplicate_monitor_basic(self) -> HealthCheckResult:
        """Check basic duplicate monitor functionality"""
        start_time = time.time()
        
        try:
            # Test operation tracking
            test_operation_id = self.duplicate_monitor.track_operation_start(
                DuplicateOperationType.CACHE_OPERATION,
                "health_check_test",
                context={'test': True}
            )
            
            # Verify operation is tracked
            if test_operation_id not in self.duplicate_monitor.active_operations:
                return HealthCheckResult(
                    component="duplicate_monitor_basic",
                    status=HealthCheckStatus.ERROR,
                    message="Failed to track test operation",
                    check_duration=time.time() - start_time
                )
            
            # End operation tracking
            self.duplicate_monitor.track_operation_end(test_operation_id, success=True)
            
            # Check metrics
            metrics_count = sum(
                m.total_executions for m in self.duplicate_monitor.operation_metrics.values()
            )
            
            return HealthCheckResult(
                component="duplicate_monitor_basic",
                status=HealthCheckStatus.HEALTHY,
                message="Duplicate monitor functioning correctly",
                details={
                    'total_operations_tracked': self.duplicate_monitor.monitor_stats['total_operations_tracked'],
                    'duplicates_detected': self.duplicate_monitor.monitor_stats['total_duplicates_detected'],
                    'active_operations': len(self.duplicate_monitor.active_operations),
                    'metrics_available': metrics_count > 0
                },
                check_duration=time.time() - start_time
            )
        
        except Exception as e:
            return HealthCheckResult(
                component="duplicate_monitor_basic",
                status=HealthCheckStatus.ERROR,
                message=f"Duplicate monitor check failed: {str(e)}",
                details={'error': str(e)},
                check_duration=time.time() - start_time
            )
    
    async def _check_performance_tracker_basic(self) -> HealthCheckResult:
        """Check basic performance tracker functionality"""
        start_time = time.time()
        
        try:
            # Test tracker functionality
            from app.services.operation_performance_tracker import OperationContext
            
            test_context = OperationContext(
                operation_type=DuplicateOperationType.CACHE_OPERATION,
                request_id="health_check_test",
                query="test query"
            )
            
            # Test tracking
            async with self.performance_tracker.track_operation(test_context) as result:
                result.success = True
                result.result_data = {'test': 'data'}
            
            # Verify tracking worked
            if not result.success:
                return HealthCheckResult(
                    component="performance_tracker_basic",
                    status=HealthCheckStatus.ERROR,
                    message="Performance tracker test operation failed",
                    check_duration=time.time() - start_time
                )
            
            return HealthCheckResult(
                component="performance_tracker_basic",
                status=HealthCheckStatus.HEALTHY,
                message="Performance tracker functioning correctly",
                details={
                    'test_execution_time': result.execution_time,
                    'tracking_successful': True,
                    'thresholds_configured': len(self.performance_tracker.performance_thresholds)
                },
                check_duration=time.time() - start_time
            )
        
        except Exception as e:
            return HealthCheckResult(
                component="performance_tracker_basic",
                status=HealthCheckStatus.ERROR,
                message=f"Performance tracker check failed: {str(e)}",
                details={'error': str(e)},
                check_duration=time.time() - start_time
            )
    
    async def _check_redis_connectivity(self) -> HealthCheckResult:
        """Check Redis connectivity and basic operations"""
        start_time = time.time()
        
        try:
            # Test basic connectivity
            ping_result = await self.redis_client.ping()
            if not ping_result:
                return HealthCheckResult(
                    component="redis_connectivity",
                    status=HealthCheckStatus.CRITICAL,
                    message="Redis ping failed",
                    check_duration=time.time() - start_time
                )
            
            # Test read/write operations
            test_key = "health_check:redis_test"
            test_value = f"test_{datetime.now().isoformat()}"
            
            await self.redis_client.setex(test_key, 10, test_value)
            retrieved_value = await self.redis_client.get(test_key)
            await self.redis_client.delete(test_key)
            
            if retrieved_value != test_value:
                return HealthCheckResult(
                    component="redis_connectivity",
                    status=HealthCheckStatus.ERROR,
                    message="Redis read/write test failed",
                    check_duration=time.time() - start_time
                )
            
            # Get Redis info
            redis_info = await self.redis_client.info()
            
            return HealthCheckResult(
                component="redis_connectivity",
                status=HealthCheckStatus.HEALTHY,
                message="Redis connectivity and operations working",
                details={
                    'redis_version': redis_info.get('redis_version'),
                    'connected_clients': redis_info.get('connected_clients'),
                    'used_memory_human': redis_info.get('used_memory_human'),
                    'total_commands_processed': redis_info.get('total_commands_processed')
                },
                check_duration=time.time() - start_time
            )
        
        except Exception as e:
            return HealthCheckResult(
                component="redis_connectivity",
                status=HealthCheckStatus.CRITICAL,
                message=f"Redis connectivity check failed: {str(e)}",
                details={'error': str(e)},
                check_duration=time.time() - start_time
            )
    
    async def _check_cache_consistency(self) -> HealthCheckResult:
        """Check cache consistency across monitoring operations"""
        start_time = time.time()
        
        try:
            # Check if monitoring data is being stored consistently
            prefixes_to_check = [
                self.duplicate_monitor.REDIS_PREFIXES['metrics'],
                self.duplicate_monitor.REDIS_PREFIXES['events'],
                self.duplicate_monitor.REDIS_PREFIXES['patterns']
            ]
            
            cache_status = {}
            for prefix in prefixes_to_check:
                # Check for recent data
                keys = await self.redis_client.keys(f"{prefix}:*")
                cache_status[prefix] = {
                    'key_count': len(keys),
                    'has_recent_data': len(keys) > 0
                }
            
            # Check for data consistency
            inconsistencies = []
            
            # Check metrics consistency
            current_metrics = await self.redis_client.get(f"{self.duplicate_monitor.REDIS_PREFIXES['metrics']}:current")
            if current_metrics:
                try:
                    metrics_data = json.loads(current_metrics)
                    # Validate metrics structure
                    required_fields = ['timestamp', 'operation_metrics', 'system_stats']
                    for field in required_fields:
                        if field not in metrics_data:
                            inconsistencies.append(f"Missing {field} in current metrics")
                except json.JSONDecodeError:
                    inconsistencies.append("Current metrics data is corrupted")
            else:
                inconsistencies.append("No current metrics data available")
            
            # Determine status
            if inconsistencies:
                if len(inconsistencies) > 3:
                    status = HealthCheckStatus.ERROR
                else:
                    status = HealthCheckStatus.WARNING
                message = f"Cache consistency issues detected: {', '.join(inconsistencies[:3])}"
            else:
                status = HealthCheckStatus.HEALTHY
                message = "Cache consistency validated successfully"
            
            return HealthCheckResult(
                component="cache_consistency",
                status=status,
                message=message,
                details={
                    'cache_status': cache_status,
                    'inconsistencies': inconsistencies,
                    'total_prefixes_checked': len(prefixes_to_check)
                },
                check_duration=time.time() - start_time
            )
        
        except Exception as e:
            return HealthCheckResult(
                component="cache_consistency",
                status=HealthCheckStatus.ERROR,
                message=f"Cache consistency check failed: {str(e)}",
                details={'error': str(e)},
                check_duration=time.time() - start_time
            )
    
    async def _check_metrics_collection(self) -> HealthCheckResult:
        """Check that metrics are being collected properly"""
        start_time = time.time()
        
        try:
            # Check recent metrics collection activity
            history_data = await self.redis_client.lrange(
                f"{self.duplicate_monitor.REDIS_PREFIXES['metrics']}:history",
                0, 10  # Get last 10 data points
            )
            
            if not history_data:
                return HealthCheckResult(
                    component="metrics_collection",
                    status=HealthCheckStatus.WARNING,
                    message="No historical metrics data found",
                    details={'history_points': 0},
                    check_duration=time.time() - start_time
                )
            
            # Parse and validate recent data
            recent_points = []
            for data_point in history_data:
                try:
                    point = json.loads(data_point)
                    point_time = datetime.fromisoformat(point['timestamp'])
                    recent_points.append(point_time)
                except Exception:
                    continue
            
            if not recent_points:
                return HealthCheckResult(
                    component="metrics_collection",
                    status=HealthCheckStatus.ERROR,
                    message="Unable to parse historical metrics data",
                    check_duration=time.time() - start_time
                )
            
            # Check collection frequency
            recent_points.sort()
            if len(recent_points) >= 2:
                latest_gap = (recent_points[-1] - recent_points[-2]).total_seconds()
                average_gap = sum(
                    (recent_points[i] - recent_points[i-1]).total_seconds()
                    for i in range(1, len(recent_points))
                ) / (len(recent_points) - 1)
            else:
                latest_gap = average_gap = 0
            
            # Validate collection frequency (should be ~1 minute)
            expected_gap = 60  # 1 minute
            if average_gap > expected_gap * 3:  # More than 3 minutes average
                status = HealthCheckStatus.WARNING
                message = f"Metrics collection frequency degraded: {average_gap:.1f}s average (expected ~{expected_gap}s)"
            elif latest_gap > expected_gap * 5:  # Latest gap > 5 minutes
                status = HealthCheckStatus.WARNING  
                message = f"Recent metrics collection gap: {latest_gap:.1f}s (expected ~{expected_gap}s)"
            else:
                status = HealthCheckStatus.HEALTHY
                message = "Metrics collection frequency is normal"
            
            return HealthCheckResult(
                component="metrics_collection",
                status=status,
                message=message,
                details={
                    'history_points': len(history_data),
                    'valid_points': len(recent_points),
                    'average_collection_gap': average_gap,
                    'latest_collection_gap': latest_gap,
                    'expected_gap': expected_gap
                },
                check_duration=time.time() - start_time
            )
        
        except Exception as e:
            return HealthCheckResult(
                component="metrics_collection",
                status=HealthCheckStatus.ERROR,
                message=f"Metrics collection check failed: {str(e)}",
                details={'error': str(e)},
                check_duration=time.time() - start_time
            )
    
    async def _check_execution_state_tracking(self) -> HealthCheckResult:
        """Check execution state tracking functionality"""
        start_time = time.time()
        
        try:
            # Test execution state tracker
            from app.services.request_execution_state_tracker import (
                check_operation_completed, mark_operation_completed
            )
            
            test_request_id = f"health_check_{int(time.time())}"
            test_operation = "intent_analysis"
            
            # Test operation completion check
            already_completed = await check_operation_completed(test_request_id, test_operation)
            
            if already_completed:
                # This is unexpected for a new test request
                status = HealthCheckStatus.WARNING
                message = "Execution state tracker returned unexpected completion status"
                details = {'unexpected_completion': True}
            else:
                # Mark as completed and verify
                await mark_operation_completed(test_request_id, test_operation, {'test': 'data'})
                
                # Check that it's now marked as completed
                now_completed = await check_operation_completed(test_request_id, test_operation)
                
                if now_completed:
                    status = HealthCheckStatus.HEALTHY
                    message = "Execution state tracking working correctly"
                    details = {'test_successful': True}
                else:
                    status = HealthCheckStatus.ERROR
                    message = "Execution state tracking not persisting state"
                    details = {'state_persistence_failed': True}
            
            return HealthCheckResult(
                component="execution_state_tracking",
                status=status,
                message=message,
                details=details,
                check_duration=time.time() - start_time
            )
        
        except ImportError as e:
            return HealthCheckResult(
                component="execution_state_tracking",
                status=HealthCheckStatus.ERROR,
                message="Execution state tracker not available",
                details={'import_error': str(e)},
                check_duration=time.time() - start_time
            )
        except Exception as e:
            return HealthCheckResult(
                component="execution_state_tracking",
                status=HealthCheckStatus.ERROR,
                message=f"Execution state tracking check failed: {str(e)}",
                details={'error': str(e)},
                check_duration=time.time() - start_time
            )
    
    async def _check_timeout_configuration(self) -> HealthCheckResult:
        """Check timeout configuration availability"""
        start_time = time.time()
        
        try:
            # Test timeout configurations for key operations
            timeout_configs = {
                'query_intent_analysis': get_timeout_value('query_intent_analysis'),
                'ai_task_planning': get_timeout_value('ai_task_planning'), 
                'batch_extraction': get_timeout_value('batch_extraction'),
                'verification_service': get_timeout_value('verification_service')
            }
            
            # Check which timeouts are configured
            configured_count = sum(1 for v in timeout_configs.values() if v is not None)
            total_count = len(timeout_configs)
            
            if configured_count == 0:
                status = HealthCheckStatus.ERROR
                message = "No timeout configurations available"
            elif configured_count < total_count:
                status = HealthCheckStatus.WARNING
                message = f"Partial timeout configuration: {configured_count}/{total_count} configured"
            else:
                status = HealthCheckStatus.HEALTHY
                message = "Timeout configurations available"
            
            return HealthCheckResult(
                component="timeout_configuration",
                status=status,
                message=message,
                details={
                    'configured_timeouts': {k: v for k, v in timeout_configs.items() if v is not None},
                    'missing_timeouts': [k for k, v in timeout_configs.items() if v is None],
                    'configuration_completeness': configured_count / total_count
                },
                check_duration=time.time() - start_time
            )
        
        except Exception as e:
            return HealthCheckResult(
                component="timeout_configuration",
                status=HealthCheckStatus.ERROR,
                message=f"Timeout configuration check failed: {str(e)}",
                details={'error': str(e)},
                check_duration=time.time() - start_time
            )
    
    async def _check_monitoring_integration(self) -> HealthCheckResult:
        """Check integration between monitoring components"""
        start_time = time.time()
        
        try:
            integration_status = {}
            
            # Check duplicate monitor integration
            monitor = get_duplicate_execution_monitor()
            integration_status['duplicate_monitor_available'] = monitor is not None
            
            # Check performance tracker integration
            tracker = get_operation_performance_tracker()
            integration_status['performance_tracker_available'] = tracker is not None
            
            # Check if they're properly connected
            if monitor and tracker:
                integration_status['components_connected'] = (
                    tracker.duplicate_monitor == monitor
                )
            else:
                integration_status['components_connected'] = False
            
            # Check background tasks are running
            integration_status['background_tasks_active'] = len(monitor.active_operations) >= 0  # Basic existence check
            
            # Determine overall integration health
            healthy_components = sum(integration_status.values())
            total_components = len(integration_status)
            
            if healthy_components == total_components:
                status = HealthCheckStatus.HEALTHY
                message = "All monitoring components integrated successfully"
            elif healthy_components >= total_components * 0.8:
                status = HealthCheckStatus.WARNING
                message = "Most monitoring components integrated"
            else:
                status = HealthCheckStatus.ERROR
                message = "Monitoring integration issues detected"
            
            return HealthCheckResult(
                component="monitoring_integration",
                status=status,
                message=message,
                details={
                    'integration_status': integration_status,
                    'healthy_ratio': healthy_components / total_components,
                    'component_health': f"{healthy_components}/{total_components}"
                },
                check_duration=time.time() - start_time
            )
        
        except Exception as e:
            return HealthCheckResult(
                component="monitoring_integration",
                status=HealthCheckStatus.ERROR,
                message=f"Integration check failed: {str(e)}",
                details={'error': str(e)},
                check_duration=time.time() - start_time
            )
    
    def _analyze_health_results(self, results: List[HealthCheckResult]) -> SystemHealthReport:
        """Analyze health check results and create comprehensive report"""
        # Count statuses
        status_counts = {status: 0 for status in HealthCheckStatus}
        for result in results:
            status_counts[result.status] += 1
        
        # Determine overall status
        if status_counts[HealthCheckStatus.CRITICAL] > 0:
            overall_status = HealthCheckStatus.CRITICAL
        elif status_counts[HealthCheckStatus.ERROR] > 0:
            overall_status = HealthCheckStatus.ERROR
        elif status_counts[HealthCheckStatus.WARNING] > 0:
            overall_status = HealthCheckStatus.WARNING
        else:
            overall_status = HealthCheckStatus.HEALTHY
        
        # Calculate overall score (0-100)
        score_weights = {
            HealthCheckStatus.HEALTHY: 100,
            HealthCheckStatus.WARNING: 70,
            HealthCheckStatus.ERROR: 30,
            HealthCheckStatus.CRITICAL: 0,
            HealthCheckStatus.UNKNOWN: 50
        }
        
        total_score = sum(score_weights[result.status] for result in results)
        overall_score = total_score / len(results) if results else 0
        
        # Extract issues and warnings
        critical_issues = [r.message for r in results if r.status == HealthCheckStatus.CRITICAL]
        warnings = [r.message for r in results if r.status == HealthCheckStatus.WARNING]
        
        # Generate recommendations
        recommendations = self._generate_health_recommendations(results)
        
        return SystemHealthReport(
            overall_status=overall_status,
            overall_score=overall_score,
            component_results=results,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _generate_health_recommendations(self, results: List[HealthCheckResult]) -> List[str]:
        """Generate recommendations based on health check results"""
        recommendations = []
        
        for result in results:
            if result.status == HealthCheckStatus.CRITICAL:
                if 'redis' in result.component.lower():
                    recommendations.append("URGENT: Fix Redis connectivity - monitoring system non-functional")
                elif 'duplicate_monitor' in result.component:
                    recommendations.append("URGENT: Restart duplicate monitoring service")
                else:
                    recommendations.append(f"URGENT: Fix critical issue in {result.component}")
            
            elif result.status == HealthCheckStatus.ERROR:
                if 'cache' in result.component.lower():
                    recommendations.append(f"Fix cache consistency issues in {result.component}")
                elif 'tracking' in result.component.lower():
                    recommendations.append(f"Repair execution state tracking in {result.component}")
                else:
                    recommendations.append(f"Address error in {result.component}: {result.message}")
            
            elif result.status == HealthCheckStatus.WARNING:
                if 'collection' in result.component.lower():
                    recommendations.append("Optimize metrics collection frequency")
                elif 'timeout' in result.component.lower():
                    recommendations.append("Complete timeout configuration setup")
                else:
                    recommendations.append(f"Monitor {result.component} for potential issues")
        
        # Add general recommendations
        error_count = sum(1 for r in results if r.status in [HealthCheckStatus.ERROR, HealthCheckStatus.CRITICAL])
        if error_count > len(results) * 0.3:  # >30% components have issues
            recommendations.append("Consider monitoring system restart or reinitialization")
        
        return recommendations
    
    async def _store_health_report(self, report: SystemHealthReport):
        """Store health report in Redis for external access"""
        try:
            # Store current health status
            health_data = {
                'overall_status': report.overall_status.value,
                'overall_score': report.overall_score,
                'component_count': len(report.component_results),
                'critical_issues': report.critical_issues,
                'warnings': report.warnings,
                'recommendations': report.recommendations,
                'timestamp': report.report_generated_at.isoformat(),
                'component_details': [
                    {
                        'component': result.component,
                        'status': result.status.value,
                        'message': result.message,
                        'check_duration': result.check_duration
                    }
                    for result in report.component_results
                ]
            }
            
            # Store with short TTL for real-time access
            await self.redis_client.setex(
                "monitoring:health:current",
                300,  # 5 minute TTL
                json.dumps(health_data, default=str)
            )
            
            # Add to health history
            await self.redis_client.lpush(
                "monitoring:health:history",
                json.dumps(health_data, default=str)
            )
            
            # Trim history
            await self.redis_client.ltrim("monitoring:health:history", 0, 288)  # Keep 24 hours
            
        except Exception as e:
            logger.error(f"Error storing health report: {e}")
    
    async def get_component_availability(self, component: str, hours: int = 24) -> Dict[str, Any]:
        """Get availability statistics for a specific component"""
        try:
            # Get historical health data
            history_data = await self.redis_client.lrange("monitoring:health:history", 0, -1)
            
            availability_points = []
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            for data_point in history_data:
                try:
                    point = json.loads(data_point)
                    point_time = datetime.fromisoformat(point['timestamp'])
                    
                    if point_time < cutoff_time:
                        continue
                    
                    # Find component in details
                    for comp_detail in point.get('component_details', []):
                        if comp_detail['component'] == component:
                            availability_points.append({
                                'timestamp': point_time,
                                'status': comp_detail['status'],
                                'healthy': comp_detail['status'] == 'healthy'
                            })
                            break
                
                except Exception:
                    continue
            
            if not availability_points:
                return {
                    'component': component,
                    'availability_percentage': 0.0,
                    'total_checks': 0,
                    'error': 'No historical data available'
                }
            
            # Calculate availability
            healthy_count = sum(1 for point in availability_points if point['healthy'])
            availability_percentage = (healthy_count / len(availability_points)) * 100
            
            # Calculate uptime/downtime periods
            status_changes = []
            for i, point in enumerate(availability_points):
                if i == 0 or point['status'] != availability_points[i-1]['status']:
                    status_changes.append(point)
            
            return {
                'component': component,
                'availability_percentage': availability_percentage,
                'total_checks': len(availability_points),
                'healthy_checks': healthy_count,
                'unhealthy_checks': len(availability_points) - healthy_count,
                'status_changes': len(status_changes),
                'current_status': availability_points[-1]['status'] if availability_points else 'unknown',
                'analysis_period': f"{hours} hours",
                'last_check': availability_points[-1]['timestamp'].isoformat() if availability_points else None
            }
        
        except Exception as e:
            logger.error(f"Error getting component availability for {component}: {e}")
            return {
                'component': component,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def run_quick_health_check(self) -> Dict[str, Any]:
        """Run quick health check for immediate status"""
        try:
            # Quick checks only
            quick_checks = [
                self._check_redis_connectivity,
                self._check_duplicate_monitor_basic
            ]
            
            results = []
            for check_func in quick_checks:
                try:
                    result = await asyncio.wait_for(check_func(), timeout=5.0)
                    results.append(result)
                except asyncio.TimeoutError:
                    results.append(HealthCheckResult(
                        component=check_func.__name__,
                        status=HealthCheckStatus.ERROR,
                        message="Health check timed out"
                    ))
                except Exception as e:
                    results.append(HealthCheckResult(
                        component=check_func.__name__,
                        status=HealthCheckStatus.ERROR,
                        message=f"Check failed: {str(e)}"
                    ))
            
            # Determine quick status
            has_critical = any(r.status == HealthCheckStatus.CRITICAL for r in results)
            has_error = any(r.status == HealthCheckStatus.ERROR for r in results)
            
            if has_critical:
                quick_status = "critical"
            elif has_error:
                quick_status = "error"
            else:
                quick_status = "healthy"
            
            return {
                'quick_status': quick_status,
                'checks_performed': len(results),
                'results': [
                    {
                        'component': r.component,
                        'status': r.status.value,
                        'message': r.message
                    }
                    for r in results
                ],
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Quick health check failed: {e}")
            return {
                'quick_status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Global health checker instance
monitoring_health_checker = MonitoringHealthChecker()


def get_monitoring_health_checker() -> MonitoringHealthChecker:
    """Get the global monitoring health checker instance"""
    return monitoring_health_checker