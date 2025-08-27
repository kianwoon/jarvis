"""
Monitoring Dashboard Configuration

Configuration settings for the duplicate execution monitoring dashboard.
Defines thresholds, display settings, and alert configurations that can be
customized without code changes.
"""

from typing import Dict, Any, List
from datetime import timedelta, datetime

# Dashboard update intervals (in seconds)
DASHBOARD_UPDATE_INTERVALS = {
    'real_time_metrics': 10,      # Update every 10 seconds
    'performance_trends': 60,     # Update every 1 minute  
    'health_checks': 300,         # Update every 5 minutes
    'pattern_analysis': 1800,     # Update every 30 minutes
    'efficiency_reports': 3600    # Update every 1 hour
}

# Alert thresholds for monitoring dashboard
MONITORING_ALERT_THRESHOLDS = {
    'duplicate_rate': {
        'warning': 0.1,      # 10% duplicate rate
        'error': 0.25,       # 25% duplicate rate
        'critical': 0.5      # 50% duplicate rate
    },
    'system_efficiency': {
        'warning': 85.0,     # 85% efficiency
        'error': 70.0,       # 70% efficiency  
        'critical': 50.0     # 50% efficiency
    },
    'operation_time': {
        'intent_analysis': {
            'warning': 5.0,   # 5 seconds
            'error': 10.0,    # 10 seconds
            'critical': 20.0  # 20 seconds
        },
        'task_planning': {
            'warning': 8.0,   # 8 seconds
            'error': 15.0,    # 15 seconds
            'critical': 30.0  # 30 seconds
        },
        'batch_extraction': {
            'warning': 30.0,  # 30 seconds
            'error': 60.0,    # 60 seconds
            'critical': 120.0 # 120 seconds
        },
        'query_embedding': {
            'warning': 3.0,   # 3 seconds
            'error': 8.0,     # 8 seconds
            'critical': 15.0  # 15 seconds
        },
        'verification': {
            'warning': 12.0,  # 12 seconds
            'error': 25.0,    # 25 seconds
            'critical': 50.0  # 50 seconds
        }
    },
    'cache_hit_rate': {
        'warning': 0.5,      # 50% cache hit rate
        'error': 0.3,        # 30% cache hit rate
        'critical': 0.1      # 10% cache hit rate
    },
    'pattern_frequency': {
        'warning': 5,        # 5 duplicates per hour
        'error': 10,         # 10 duplicates per hour
        'critical': 20       # 20 duplicates per hour
    }
}

# Dashboard section configurations
DASHBOARD_SECTIONS = {
    'overview': {
        'title': 'System Overview',
        'widgets': [
            'total_operations_tracked',
            'duplicates_detected', 
            'system_efficiency_score',
            'active_patterns',
            'health_status'
        ],
        'refresh_interval': 10
    },
    'real_time_metrics': {
        'title': 'Real-Time Metrics',
        'widgets': [
            'operation_breakdown',
            'duplicate_rates_chart',
            'execution_times_chart',
            'cache_performance_chart'
        ],
        'refresh_interval': 15
    },
    'performance_trends': {
        'title': 'Performance Trends',
        'widgets': [
            'efficiency_trend_chart',
            'duplicate_pattern_trend',
            'operation_time_trend',
            'cache_hit_trend'
        ],
        'refresh_interval': 60
    },
    'alerts_and_patterns': {
        'title': 'Alerts & Patterns',
        'widgets': [
            'active_alerts',
            'detected_patterns',
            'pattern_analysis',
            'recommendations'
        ],
        'refresh_interval': 30
    },
    'health_monitoring': {
        'title': 'System Health',
        'widgets': [
            'component_health',
            'availability_metrics',
            'health_trends',
            'system_diagnostics'
        ],
        'refresh_interval': 60
    }
}

# Chart configurations for performance visualization
CHART_CONFIGURATIONS = {
    'duplicate_rates_chart': {
        'type': 'line',
        'title': 'Duplicate Execution Rates',
        'x_axis': 'timestamp',
        'y_axis': 'duplicate_rate',
        'data_points': 50,
        'time_range': '1h',
        'color_scheme': 'red_gradient'
    },
    'execution_times_chart': {
        'type': 'multi_line',
        'title': 'Operation Execution Times', 
        'x_axis': 'timestamp',
        'y_axis': 'execution_time',
        'data_points': 100,
        'time_range': '2h',
        'color_scheme': 'blue_gradient'
    },
    'efficiency_trend_chart': {
        'type': 'area',
        'title': 'System Efficiency Trend',
        'x_axis': 'timestamp', 
        'y_axis': 'efficiency_score',
        'data_points': 72,
        'time_range': '6h',
        'color_scheme': 'green_gradient'
    },
    'cache_performance_chart': {
        'type': 'stacked_bar',
        'title': 'Cache Hit/Miss Rates',
        'x_axis': 'operation_type',
        'y_axis': 'rate',
        'stacks': ['cache_hits', 'cache_misses'],
        'color_scheme': 'cache_performance'
    }
}

# Color schemes for dashboard charts
COLOR_SCHEMES = {
    'red_gradient': ['#ffebee', '#ffcdd2', '#ef5350', '#c62828', '#b71c1c'],
    'blue_gradient': ['#e3f2fd', '#bbdefb', '#42a5f5', '#1976d2', '#0d47a1'],
    'green_gradient': ['#e8f5e8', '#c8e6c9', '#66bb6a', '#388e3c', '#1b5e20'],
    'cache_performance': {
        'cache_hits': '#4caf50',     # Green for hits
        'cache_misses': '#f44336'    # Red for misses
    }
}

# Widget display configurations
WIDGET_CONFIGURATIONS = {
    'total_operations_tracked': {
        'type': 'metric_card',
        'title': 'Total Operations',
        'format': 'number',
        'icon': 'analytics',
        'color': 'primary'
    },
    'duplicates_detected': {
        'type': 'metric_card', 
        'title': 'Duplicates Detected',
        'format': 'number',
        'icon': 'warning',
        'color': 'warning',
        'trend': True
    },
    'system_efficiency_score': {
        'type': 'progress_card',
        'title': 'System Efficiency',
        'format': 'percentage',
        'icon': 'speed',
        'color': 'success',
        'thresholds': {
            'excellent': 90,
            'good': 75,
            'fair': 60,
            'poor': 40
        }
    },
    'active_patterns': {
        'type': 'metric_card',
        'title': 'Active Patterns', 
        'format': 'number',
        'icon': 'pattern',
        'color': 'info'
    },
    'health_status': {
        'type': 'status_card',
        'title': 'System Health',
        'format': 'status_indicator',
        'icon': 'health_and_safety',
        'statuses': {
            'healthy': {'color': 'success', 'text': 'All Systems Operational'},
            'warning': {'color': 'warning', 'text': 'Minor Issues Detected'},
            'error': {'color': 'error', 'text': 'Problems Require Attention'},
            'critical': {'color': 'error', 'text': 'Critical Issues - Immediate Action Required'}
        }
    }
}

# Data retention settings
DATA_RETENTION = {
    'real_time_metrics': timedelta(hours=4),      # 4 hours of real-time data
    'performance_metrics': timedelta(days=7),    # 1 week of performance data
    'duplicate_events': timedelta(days=3),       # 3 days of duplicate events  
    'health_reports': timedelta(days=1),         # 24 hours of health reports
    'pattern_data': timedelta(days=7),          # 1 week of pattern data
    'efficiency_reports': timedelta(days=30)    # 30 days of efficiency data
}

# Export settings for external tools
EXPORT_CONFIGURATIONS = {
    'duplicate_report': {
        'formats': ['json', 'csv', 'xlsx'],
        'default_time_range': '24h',
        'include_recommendations': True,
        'include_raw_events': False
    },
    'performance_report': {
        'formats': ['json', 'pdf'],
        'default_time_range': '7d',
        'include_charts': True,
        'include_trends': True
    },
    'health_report': {
        'formats': ['json', 'html'],
        'include_component_details': True,
        'include_recommendations': True,
        'include_history': True
    }
}

# Integration settings
INTEGRATION_SETTINGS = {
    'auto_integration': {
        'enabled': True,
        'target_functions': [
            'ai_task_planner.understand_and_plan',
            'query_intent_analyzer.analyze_intent', 
            'notebook_rag_service.query_rag',
            'ai_verification_service.verify_completeness'
        ],
        'default_caching': True,
        'performance_tracking': True
    },
    'monitoring_overhead': {
        'max_acceptable_overhead': 0.05,  # 5% maximum overhead
        'disable_if_overhead_exceeds': 0.1,  # 10% disable threshold
        'lightweight_mode_threshold': 0.08  # 8% switch to lightweight mode
    }
}

# Notification settings
NOTIFICATION_SETTINGS = {
    'duplicate_alerts': {
        'enabled': True,
        'channels': ['log', 'metrics'],  # Could extend to ['email', 'slack', 'webhook']
        'severity_filters': {
            'log': ['info', 'warning', 'error', 'critical'],
            'metrics': ['warning', 'error', 'critical']
        }
    },
    'performance_alerts': {
        'enabled': True,
        'channels': ['log', 'metrics'],
        'severity_filters': {
            'log': ['warning', 'error', 'critical'],
            'metrics': ['error', 'critical']  
        }
    },
    'health_alerts': {
        'enabled': True,
        'channels': ['log'],
        'severity_filters': {
            'log': ['error', 'critical']
        }
    }
}

# Dashboard API configurations
API_CONFIGURATIONS = {
    'rate_limiting': {
        'enabled': True,
        'requests_per_minute': 60,
        'burst_limit': 10
    },
    'caching': {
        'enabled': True,
        'default_ttl': 30,  # 30 seconds
        'endpoint_ttls': {
            '/monitoring/dashboard': 15,
            '/monitoring/metrics/real-time': 10,
            '/monitoring/health/quick': 60,
            '/monitoring/report/duplicates': 300
        }
    },
    'response_optimization': {
        'compression_enabled': True,
        'max_response_size': 1048576,  # 1MB
        'paginate_large_results': True,
        'default_page_size': 100
    }
}


def get_dashboard_config() -> Dict[str, Any]:
    """Get complete dashboard configuration"""
    return {
        'update_intervals': DASHBOARD_UPDATE_INTERVALS,
        'alert_thresholds': MONITORING_ALERT_THRESHOLDS,
        'sections': DASHBOARD_SECTIONS,
        'charts': CHART_CONFIGURATIONS,
        'colors': COLOR_SCHEMES,
        'widgets': WIDGET_CONFIGURATIONS,
        'data_retention': {k: v.total_seconds() for k, v in DATA_RETENTION.items()},
        'export': EXPORT_CONFIGURATIONS,
        'integration': INTEGRATION_SETTINGS,
        'notifications': NOTIFICATION_SETTINGS,
        'api': API_CONFIGURATIONS
    }


def get_operation_monitoring_config(operation_type: str) -> Dict[str, Any]:
    """Get monitoring configuration for specific operation type"""
    base_config = get_dashboard_config()
    
    # Operation-specific configurations
    operation_configs = {
        'intent_analysis': {
            'target_execution_time': 2.0,
            'cache_ttl': 300,  # 5 minutes
            'duplicate_detection_window': 60,  # 1 minute
            'priority': 'high'
        },
        'task_planning': {
            'target_execution_time': 3.0,
            'cache_ttl': 600,  # 10 minutes
            'duplicate_detection_window': 120,  # 2 minutes
            'priority': 'high'
        },
        'query_embedding': {
            'target_execution_time': 1.0,
            'cache_ttl': 1800,  # 30 minutes
            'duplicate_detection_window': 30,  # 30 seconds
            'priority': 'medium'
        },
        'batch_extraction': {
            'target_execution_time': 10.0,
            'cache_ttl': 3600,  # 1 hour
            'duplicate_detection_window': 300,  # 5 minutes
            'priority': 'critical'
        },
        'verification': {
            'target_execution_time': 5.0,
            'cache_ttl': 900,  # 15 minutes
            'duplicate_detection_window': 180,  # 3 minutes
            'priority': 'high'
        }
    }
    
    operation_config = operation_configs.get(operation_type, {
        'target_execution_time': 5.0,
        'cache_ttl': 600,
        'duplicate_detection_window': 60,
        'priority': 'medium'
    })
    
    return {
        **base_config,
        'operation_specific': operation_config,
        'operation_type': operation_type
    }


def get_health_check_config() -> Dict[str, Any]:
    """Get health check configuration"""
    return {
        'check_intervals': {
            'quick_health': 30,        # 30 seconds
            'comprehensive_health': 300, # 5 minutes
            'component_health': 120,    # 2 minutes
            'availability_check': 600   # 10 minutes
        },
        'timeout_limits': {
            'quick_check': 5.0,        # 5 seconds
            'component_check': 10.0,   # 10 seconds
            'comprehensive_check': 30.0 # 30 seconds
        },
        'failure_thresholds': {
            'consecutive_failures_warning': 3,
            'consecutive_failures_error': 5,
            'failure_rate_warning': 0.1,   # 10% failure rate
            'failure_rate_error': 0.25     # 25% failure rate
        },
        'component_weights': {
            'duplicate_monitor': 0.3,      # 30% weight
            'performance_tracker': 0.25,   # 25% weight
            'redis_connectivity': 0.2,     # 20% weight
            'cache_consistency': 0.15,     # 15% weight
            'metrics_collection': 0.1      # 10% weight
        }
    }


# Pre-configured monitoring templates for common use cases
MONITORING_TEMPLATES = {
    'notebook_service_monitoring': {
        'description': 'Complete monitoring setup for notebook services',
        'decorators': [
            {
                'function_pattern': 'query.*',
                'decorator': 'monitor_notebook_operation',
                'operation_type': 'RAG_QUERY'
            },
            {
                'function_pattern': 'chat.*',
                'decorator': 'monitor_notebook_operation', 
                'operation_type': 'NOTEBOOK_CHAT'
            }
        ],
        'health_checks': ['notebook_operations', 'rag_performance', 'chat_efficiency'],
        'alerts': ['duplicate_queries', 'slow_responses', 'cache_misses']
    },
    'ai_service_monitoring': {
        'description': 'Complete monitoring setup for AI services',
        'decorators': [
            {
                'function_pattern': '.*intent.*',
                'decorator': 'monitor_ai_service_operation',
                'operation_type': 'INTENT_ANALYSIS'
            },
            {
                'function_pattern': '.*plan.*',
                'decorator': 'monitor_ai_service_operation',
                'operation_type': 'TASK_PLANNING'
            },
            {
                'function_pattern': '.*verify.*',
                'decorator': 'monitor_ai_service_operation',
                'operation_type': 'VERIFICATION'
            }
        ],
        'health_checks': ['ai_response_times', 'llm_connectivity', 'planning_accuracy'],
        'alerts': ['slow_ai_operations', 'planning_failures', 'verification_errors']
    },
    'data_processing_monitoring': {
        'description': 'Complete monitoring setup for data processing operations',
        'decorators': [
            {
                'function_pattern': '.*extract.*',
                'decorator': 'monitor_duplicate_execution',
                'operation_type': 'BATCH_EXTRACTION'
            },
            {
                'function_pattern': '.*embed.*',
                'decorator': 'monitor_duplicate_execution',
                'operation_type': 'QUERY_EMBEDDING'
            }
        ],
        'health_checks': ['extraction_performance', 'embedding_quality', 'batch_efficiency'],
        'alerts': ['extraction_timeouts', 'embedding_failures', 'batch_duplicates']
    }
}


def get_monitoring_template(template_name: str) -> Dict[str, Any]:
    """Get pre-configured monitoring template"""
    if template_name not in MONITORING_TEMPLATES:
        raise ValueError(f"Unknown monitoring template: {template_name}")
    
    template = MONITORING_TEMPLATES[template_name].copy()
    template['template_name'] = template_name
    template['retrieved_at'] = datetime.now().isoformat()
    
    return template


def get_all_monitoring_configurations() -> Dict[str, Any]:
    """Get all monitoring configurations for setup and validation"""
    # Import here to avoid circular imports
    try:
        from app.services.duplicate_execution_monitor import DuplicateOperationType
        operation_types = [op.value for op in DuplicateOperationType]
    except ImportError:
        operation_types = [
            'intent_analysis', 'task_planning', 'query_embedding', 'batch_extraction',
            'verification', 'cache_operation', 'llm_call', 'rag_query', 'notebook_chat'
        ]
    
    return {
        'dashboard_config': get_dashboard_config(),
        'health_check_config': get_health_check_config(),
        'monitoring_templates': MONITORING_TEMPLATES,
        'available_operation_types': operation_types,
        'configuration_version': '1.0.0',
        'last_updated': datetime.now().isoformat()
    }