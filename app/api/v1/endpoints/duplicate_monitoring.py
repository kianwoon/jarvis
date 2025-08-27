"""
Duplicate Execution Monitoring API Endpoints

Provides REST API endpoints for accessing duplicate execution monitoring data,
performance metrics, and operational dashboards. Integrates with existing
FastAPI patterns and authentication infrastructure.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

from app.services.duplicate_execution_monitor import (
    get_duplicate_execution_monitor, DuplicateExecutionMonitor, DuplicateOperationType
)
from app.services.operation_performance_tracker import (
    get_operation_performance_tracker, OperationPerformanceTracker
)
from app.services.monitoring_health_checks import (
    get_monitoring_health_checker, MonitoringHealthChecker
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/duplicate-monitoring", tags=["duplicate-monitoring"])


@router.get("/dashboard")
async def get_monitoring_dashboard():
    """
    Get comprehensive monitoring dashboard data
    
    Returns:
        Dashboard data with real-time metrics, alerts, and recommendations
    """
    try:
        tracker = get_operation_performance_tracker()
        dashboard_data = await tracker.create_monitoring_dashboard_data()
        
        return {
            "status": "success",
            "data": dashboard_data,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting monitoring dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")


@router.get("/metrics/real-time")
async def get_real_time_metrics():
    """
    Get real-time duplicate monitoring metrics
    
    Returns:
        Current metrics including duplicate rates, active operations, and alerts
    """
    try:
        monitor = get_duplicate_execution_monitor()
        metrics = await monitor.get_real_time_metrics()
        
        return {
            "status": "success", 
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting real-time metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/report/duplicates")
async def get_duplicate_report(
    hours: int = Query(24, ge=1, le=168, description="Hours of data to analyze"),
    operation_type: Optional[str] = Query(None, description="Filter by operation type")
):
    """
    Get comprehensive duplicate execution report
    
    Args:
        hours: Number of hours to analyze (1-168)
        operation_type: Optional filter by operation type
        
    Returns:
        Detailed duplicate execution analysis report
    """
    try:
        # Validate operation type
        op_type_enum = None
        if operation_type:
            try:
                op_type_enum = DuplicateOperationType(operation_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid operation type: {operation_type}. Valid types: {[t.value for t in DuplicateOperationType]}"
                )
        
        tracker = get_operation_performance_tracker()
        report = await tracker.get_duplicate_prevention_report(hours, op_type_enum)
        
        return {
            "status": "success",
            "report": report,
            "metadata": {
                "hours_analyzed": hours,
                "operation_filter": operation_type,
                "generated_at": datetime.now().isoformat()
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating duplicate report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@router.get("/health/operations")
async def get_operation_health():
    """
    Get health status for all monitored operations
    
    Returns:
        Health status, scores, and issues for each operation type
    """
    try:
        monitor = get_duplicate_execution_monitor()
        health_status = monitor.get_operation_health_status()
        
        return {
            "status": "success",
            "health": health_status,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting operation health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get health status: {str(e)}")


@router.get("/patterns/active")
async def get_active_patterns():
    """
    Get currently active duplicate patterns
    
    Returns:
        List of detected duplicate patterns with analysis and recommendations
    """
    try:
        monitor = get_duplicate_execution_monitor()
        
        # Get patterns detected in last 4 hours
        cutoff_time = datetime.now() - timedelta(hours=4)
        active_patterns = [
            {
                'pattern_id': pattern.pattern_id,
                'operation_type': pattern.operation_type.value,
                'frequency': pattern.frequency,
                'average_gap': pattern.average_gap,
                'impact_score': pattern.impact_score,
                'affected_operations': len(pattern.affected_operations),
                'first_detected': pattern.first_detected.isoformat(),
                'last_detected': pattern.last_detected.isoformat(),
                'recommendations': pattern.recommendations,
                'time_since_last': (datetime.now() - pattern.last_detected).total_seconds()
            }
            for pattern in monitor.detected_patterns.values()
            if pattern.last_detected > cutoff_time
        ]
        
        return {
            "status": "success",
            "patterns": active_patterns,
            "total_patterns": len(active_patterns),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting active patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get patterns: {str(e)}")


@router.get("/analysis/operation/{operation_type}")
async def get_operation_analysis(
    operation_type: str = Path(..., description="Operation type to analyze"),
    hours: int = Query(24, ge=1, le=168, description="Hours of data to analyze")
):
    """
    Get detailed analysis for a specific operation type
    
    Args:
        operation_type: Type of operation to analyze
        hours: Hours of historical data to include
        
    Returns:
        Detailed operation analysis including metrics, patterns, and recommendations
    """
    try:
        # Validate operation type
        try:
            op_type_enum = DuplicateOperationType(operation_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid operation type: {operation_type}. Valid types: {[t.value for t in DuplicateOperationType]}"
            )
        
        monitor = get_duplicate_execution_monitor()
        tracker = get_operation_performance_tracker()
        
        # Get operation metrics
        operation_metrics = monitor.operation_metrics[op_type_enum]
        
        # Get duplicate report for this operation
        duplicate_report = await monitor.get_duplicate_report(op_type_enum, hours)
        
        # Get performance insights
        performance_insights = await tracker._generate_performance_insights(hours, op_type_enum)
        
        # Find related patterns
        related_patterns = [
            pattern for pattern in monitor.detected_patterns.values()
            if pattern.operation_type == op_type_enum
        ]
        
        analysis = {
            'operation_type': operation_type,
            'analysis_period': f"{hours} hours",
            'metrics': {
                'total_executions': operation_metrics.total_executions,
                'unique_executions': operation_metrics.unique_executions,
                'duplicate_executions': operation_metrics.duplicate_executions,
                'duplicate_rate': operation_metrics.duplicate_rate,
                'average_execution_time': operation_metrics.average_execution_time,
                'cache_hit_rate': operation_metrics.cache_hit_rate,
                'timeout_occurrences': operation_metrics.timeout_occurrences,
                'circuit_breaker_activations': operation_metrics.circuit_breaker_activations,
                'last_updated': operation_metrics.last_updated.isoformat()
            },
            'performance_analysis': performance_insights,
            'duplicate_patterns': [
                {
                    'pattern_id': pattern.pattern_id,
                    'frequency': pattern.frequency,
                    'average_gap': pattern.average_gap,
                    'impact_score': pattern.impact_score,
                    'affected_operations': len(pattern.affected_operations),
                    'recommendations': pattern.recommendations,
                    'detection_period': {
                        'first': pattern.first_detected.isoformat(),
                        'last': pattern.last_detected.isoformat()
                    }
                }
                for pattern in related_patterns
            ],
            'efficiency_assessment': {
                'efficiency_score': tracker._calculate_operation_efficiency(operation_metrics),
                'primary_issues': [],
                'improvement_potential': {}
            }
        }
        
        # Add efficiency issues
        if operation_metrics.duplicate_rate > 0.1:
            analysis['efficiency_assessment']['primary_issues'].append({
                'issue': 'high_duplicate_rate',
                'severity': 'high' if operation_metrics.duplicate_rate > 0.3 else 'medium',
                'impact': f"{operation_metrics.duplicate_rate:.1%} of operations are duplicates"
            })
        
        if operation_metrics.average_execution_time > tracker.performance_thresholds.get(op_type_enum, {}).get('warning_time', float('inf')):
            analysis['efficiency_assessment']['primary_issues'].append({
                'issue': 'slow_performance',
                'severity': 'high',
                'impact': f"Average execution time {operation_metrics.average_execution_time:.2f}s exceeds threshold"
            })
        
        return {
            "status": "success",
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing operation {operation_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze operation: {str(e)}")


@router.get("/alerts/active")
async def get_active_alerts():
    """
    Get currently active monitoring alerts
    
    Returns:
        List of active alerts with severity and recommendations
    """
    try:
        monitor = get_duplicate_execution_monitor()
        tracker = get_operation_performance_tracker()
        
        # Get system health data which includes alerts
        health_data = monitor.get_operation_health_status()
        
        # Get real-time metrics for additional alerts
        real_time_data = await monitor.get_real_time_metrics()
        
        alerts = []
        
        # Extract alerts from health data
        for operation, health_info in health_data.get('operations', {}).items():
            if health_info.get('issues'):
                for issue in health_info['issues']:
                    alerts.append({
                        'id': f"{operation}_{issue.replace(' ', '_').replace(':', '')}",
                        'operation': operation,
                        'severity': health_info['status'],
                        'issue': issue,
                        'health_score': health_info['health_score'],
                        'detected_at': datetime.now().isoformat(),
                        'type': 'operation_health'
                    })
        
        # Add performance alerts from real-time data
        for perf_alert in real_time_data.get('performance_alerts', []):
            alerts.append({
                'id': f"perf_{perf_alert['operation']}_{perf_alert['issue']}",
                'operation': perf_alert['operation'],
                'severity': 'warning' if perf_alert['value'] < perf_alert['threshold'] * 2 else 'error',
                'issue': f"{perf_alert['issue']}: {perf_alert['value']:.2f} (threshold: {perf_alert['threshold']})",
                'detected_at': datetime.now().isoformat(),
                'type': 'performance_threshold'
            })
        
        # Sort by severity
        severity_order = {'critical': 0, 'error': 1, 'warning': 2, 'info': 3}
        alerts.sort(key=lambda x: severity_order.get(x['severity'], 99))
        
        return {
            "status": "success",
            "alerts": alerts,
            "total_alerts": len(alerts),
            "severity_breakdown": {
                severity: len([a for a in alerts if a['severity'] == severity])
                for severity in severity_order.keys()
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@router.get("/insights/performance")
async def get_performance_insights(
    hours: int = Query(24, ge=1, le=168, description="Hours of data to analyze"),
    operation_type: Optional[str] = Query(None, description="Filter by operation type")
):
    """
    Get performance insights and optimization recommendations
    
    Args:
        hours: Hours of data to analyze
        operation_type: Optional filter by operation type
        
    Returns:
        Performance insights with actionable recommendations
    """
    try:
        # Validate operation type
        op_type_enum = None
        if operation_type:
            try:
                op_type_enum = DuplicateOperationType(operation_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid operation type: {operation_type}"
                )
        
        tracker = get_operation_performance_tracker()
        insights = await tracker._generate_performance_insights(hours, op_type_enum)
        
        # Get prevention recommendations
        monitor = get_duplicate_execution_monitor()
        duplicate_report = await monitor.get_duplicate_report(op_type_enum, hours)
        recommendations = await tracker._generate_prevention_recommendations(duplicate_report)
        
        return {
            "status": "success",
            "insights": insights,
            "recommendations": recommendations,
            "analysis_period": f"{hours} hours",
            "operation_filter": operation_type,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance insights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")


@router.get("/metrics/historical")
async def get_historical_metrics(
    hours: int = Query(24, ge=1, le=168, description="Hours of historical data"),
    operation_type: Optional[str] = Query(None, description="Filter by operation type"),
    metric_type: Optional[str] = Query(None, description="Filter by metric type")
):
    """
    Get historical performance metrics for trend analysis
    
    Args:
        hours: Hours of historical data to retrieve
        operation_type: Optional filter by operation type
        metric_type: Optional filter by specific metric (duplicate_rate, execution_time, cache_hit_rate)
        
    Returns:
        Historical metrics data suitable for charting and trend analysis
    """
    try:
        tracker = get_operation_performance_tracker()
        historical_data = await tracker._get_recent_performance_data(hours)
        
        # Apply filters
        filtered_data = {}
        
        for metric_name, data_points in historical_data.items():
            if metric_type and metric_type not in metric_name:
                continue
            
            if operation_type:
                filtered_points = [
                    point for point in data_points
                    if point.get('operation') == operation_type
                ]
                if filtered_points:
                    filtered_data[metric_name] = filtered_points
            else:
                filtered_data[metric_name] = data_points
        
        return {
            "status": "success",
            "historical_data": filtered_data,
            "metadata": {
                "hours_analyzed": hours,
                "operation_filter": operation_type,
                "metric_filter": metric_type,
                "data_points": sum(len(points) for points in filtered_data.values()),
                "generated_at": datetime.now().isoformat()
            }
        }
    
    except Exception as e:
        logger.error(f"Error getting historical metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get historical data: {str(e)}")


@router.get("/status/implementation")
async def get_implementation_status():
    """
    Get implementation status of duplicate prevention measures
    
    Returns:
        Status of various duplicate prevention components and readiness assessment
    """
    try:
        tracker = get_operation_performance_tracker()
        status = await tracker._check_implementation_status()
        
        return {
            "status": "success",
            "implementation_status": status,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting implementation status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/operation-types")
async def get_operation_types():
    """
    Get list of all supported operation types for monitoring
    
    Returns:
        List of operation types with descriptions
    """
    operation_descriptions = {
        DuplicateOperationType.INTENT_ANALYSIS: "AI-powered query intent understanding and classification",
        DuplicateOperationType.TASK_PLANNING: "AI task execution planning and strategy generation",
        DuplicateOperationType.QUERY_EMBEDDING: "Vector embedding generation for semantic search",
        DuplicateOperationType.BATCH_EXTRACTION: "Large-scale data extraction and processing",
        DuplicateOperationType.VERIFICATION: "AI verification and quality assessment",
        DuplicateOperationType.CACHE_OPERATION: "Cache read/write operations and management", 
        DuplicateOperationType.LLM_CALL: "Large Language Model API calls and responses",
        DuplicateOperationType.RAG_QUERY: "Retrieval Augmented Generation query processing",
        DuplicateOperationType.NOTEBOOK_CHAT: "Notebook-specific chat and conversation processing"
    }
    
    return {
        "status": "success",
        "operation_types": [
            {
                "type": op_type.value,
                "description": operation_descriptions.get(op_type, "Standard operation type"),
                "monitoring_active": True
            }
            for op_type in DuplicateOperationType
        ],
        "total_types": len(DuplicateOperationType),
        "timestamp": datetime.now().isoformat()
    }


@router.post("/alerts/resolve/{alert_id}")
async def resolve_alert(alert_id: str = Path(..., description="Alert ID to resolve")):
    """
    Mark an alert as resolved
    
    Args:
        alert_id: ID of the alert to resolve
        
    Returns:
        Confirmation of alert resolution
    """
    try:
        # Note: This is a placeholder for alert resolution logic
        # In a full implementation, you would update the alert status in Redis
        logger.info(f"Alert {alert_id} marked as resolved")
        
        return {
            "status": "success",
            "message": f"Alert {alert_id} has been resolved",
            "resolved_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error resolving alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {str(e)}")


@router.get("/recommendations/top")
async def get_top_recommendations(
    limit: int = Query(10, ge=1, le=50, description="Number of recommendations to return")
):
    """
    Get top duplicate prevention recommendations
    
    Args:
        limit: Maximum number of recommendations to return
        
    Returns:
        Prioritized list of recommendations for preventing duplicate executions
    """
    try:
        monitor = get_duplicate_execution_monitor()
        tracker = get_operation_performance_tracker()
        
        # Get comprehensive duplicate report
        duplicate_report = await monitor.get_duplicate_report(None, 24)  # Last 24 hours
        
        # Generate recommendations
        recommendations = await tracker._generate_prevention_recommendations(duplicate_report)
        
        # Return top recommendations
        top_recommendations = recommendations[:limit]
        
        return {
            "status": "success",
            "recommendations": top_recommendations,
            "total_available": len(recommendations),
            "returned": len(top_recommendations),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@router.get("/health/comprehensive")
async def run_comprehensive_health_check():
    """
    Run comprehensive health check of all monitoring systems
    
    Returns:
        Complete health report with component status, issues, and recommendations
    """
    try:
        from app.services.monitoring_health_checks import get_monitoring_health_checker
        health_checker = get_monitoring_health_checker()
        health_report = await health_checker.run_comprehensive_health_check()
        
        return {
            "status": "success",
            "health_report": {
                'overall_status': health_report.overall_status.value,
                'overall_score': health_report.overall_score,
                'component_results': [
                    {
                        'component': result.component,
                        'status': result.status.value,
                        'message': result.message,
                        'details': result.details,
                        'check_duration': result.check_duration,
                        'checked_at': result.checked_at.isoformat()
                    }
                    for result in health_report.component_results
                ],
                'critical_issues': health_report.critical_issues,
                'warnings': health_report.warnings,
                'recommendations': health_report.recommendations,
                'generated_at': health_report.report_generated_at.isoformat()
            }
        }
    
    except Exception as e:
        logger.error(f"Error running comprehensive health check: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/health/quick")
async def run_quick_health_check():
    """
    Run quick health check for immediate status
    
    Returns:
        Quick health status for immediate monitoring needs
    """
    try:
        from app.services.monitoring_health_checks import get_monitoring_health_checker
        health_checker = get_monitoring_health_checker()
        quick_status = await health_checker.run_quick_health_check()
        
        return {
            "status": "success",
            "quick_health": quick_status
        }
    
    except Exception as e:
        logger.error(f"Error running quick health check: {e}")
        raise HTTPException(status_code=500, detail=f"Quick health check failed: {str(e)}")


@router.get("/availability/{component}")
async def get_component_availability(
    component: str = Path(..., description="Component name to check availability"),
    hours: int = Query(24, ge=1, le=168, description="Hours of availability data")
):
    """
    Get availability statistics for a specific monitoring component
    
    Args:
        component: Name of the component to check
        hours: Hours of historical availability data
        
    Returns:
        Component availability statistics and uptime metrics
    """
    try:
        from app.services.monitoring_health_checks import get_monitoring_health_checker
        health_checker = get_monitoring_health_checker()
        availability = await health_checker.get_component_availability(component, hours)
        
        return {
            "status": "success",
            "availability": availability,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting availability for {component}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get availability: {str(e)}")


@router.get("/config/dashboard")
async def get_dashboard_configuration():
    """
    Get dashboard configuration settings
    
    Returns:
        Complete dashboard configuration including thresholds, sections, and display settings
    """
    try:
        from app.config.monitoring_dashboard_config import get_dashboard_config
        config = get_dashboard_config()
        
        return {
            "status": "success",
            "configuration": config,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting dashboard configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")


@router.get("/config/operation/{operation_type}")
async def get_operation_configuration(
    operation_type: str = Path(..., description="Operation type to get configuration for")
):
    """
    Get configuration for a specific operation type
    
    Args:
        operation_type: Type of operation to configure
        
    Returns:
        Operation-specific monitoring configuration
    """
    try:
        from app.config.monitoring_dashboard_config import get_operation_monitoring_config
        config = get_operation_monitoring_config(operation_type)
        
        return {
            "status": "success",
            "operation_type": operation_type,
            "configuration": config,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting operation configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get operation config: {str(e)}")


@router.get("/integration/status")
async def get_integration_status():
    """
    Get monitoring integration status across the system
    
    Returns:
        Integration status with coverage metrics and recommendations
    """
    try:
        from app.services.monitoring_integrations import get_integration_manager
        manager = get_integration_manager()
        status = await manager.get_integration_status()
        
        return {
            "status": "success",
            "integration_status": status,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting integration status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get integration status: {str(e)}")


@router.get("/integration/guide") 
async def get_integration_guide():
    """
    Get integration guide for adding monitoring to existing functions
    
    Returns:
        Step-by-step guide for integrating monitoring with code examples
    """
    try:
        from app.services.monitoring_integrations import get_integration_manager
        manager = get_integration_manager()
        guide = await manager.create_integration_guide()
        
        return {
            "status": "success",
            "guide": guide,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting integration guide: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get guide: {str(e)}")


@router.get("/summary/efficiency")
async def get_efficiency_summary():
    """
    Get system efficiency summary with key metrics
    
    Returns:
        High-level efficiency metrics and trends
    """
    try:
        monitor = get_duplicate_execution_monitor()
        
        # Calculate efficiency metrics
        total_operations = sum(
            metrics.total_executions for metrics in monitor.operation_metrics.values()
        )
        total_duplicates = sum(
            metrics.duplicate_executions for metrics in monitor.operation_metrics.values()
        )
        total_unique = sum(
            metrics.unique_executions for metrics in monitor.operation_metrics.values()
        )
        
        # Efficiency calculations
        duplicate_rate = total_duplicates / total_operations if total_operations > 0 else 0
        efficiency_score = monitor.monitor_stats['system_efficiency_score']
        
        # Operation efficiency breakdown
        operation_efficiency = {}
        for op_type, metrics in monitor.operation_metrics.items():
            if metrics.total_executions > 0:
                operation_efficiency[op_type.value] = {
                    'efficiency_score': get_operation_performance_tracker()._calculate_operation_efficiency(metrics),
                    'duplicate_rate': metrics.duplicate_rate,
                    'executions': metrics.total_executions
                }
        
        # Identify most and least efficient operations
        if operation_efficiency:
            most_efficient = max(operation_efficiency.items(), key=lambda x: x[1]['efficiency_score'])
            least_efficient = min(operation_efficiency.items(), key=lambda x: x[1]['efficiency_score'])
        else:
            most_efficient = least_efficient = None
        
        return {
            "status": "success",
            "efficiency_summary": {
                "overall_efficiency_score": efficiency_score,
                "total_operations": total_operations,
                "unique_operations": total_unique,
                "duplicate_operations": total_duplicates,
                "duplicate_rate": duplicate_rate,
                "operations_tracked": len(monitor.operation_metrics),
                "patterns_detected": len(monitor.detected_patterns),
                "monitoring_uptime": (
                    datetime.now() - monitor.monitor_stats['monitoring_start']
                ).total_seconds(),
                "most_efficient_operation": {
                    "operation": most_efficient[0],
                    "score": most_efficient[1]['efficiency_score']
                } if most_efficient else None,
                "least_efficient_operation": {
                    "operation": least_efficient[0], 
                    "score": least_efficient[1]['efficiency_score']
                } if least_efficient else None
            },
            "operation_breakdown": operation_efficiency,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting efficiency summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")