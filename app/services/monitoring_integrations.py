"""
Monitoring Integration Utilities

Provides easy integration of duplicate execution monitoring and performance
tracking with existing services. Offers decorators and utilities that can
be seamlessly added to existing functions without major code changes.
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Dict, Any, Optional, Callable, Union
from functools import wraps
import inspect

from app.services.duplicate_execution_monitor import DuplicateOperationType, get_duplicate_execution_monitor
from app.services.operation_performance_tracker import (
    OperationContext, get_operation_performance_tracker
)

logger = logging.getLogger(__name__)


def monitor_duplicate_execution(
    operation_type: DuplicateOperationType,
    extract_request_id: Optional[Callable] = None,
    extract_conversation_id: Optional[Callable] = None,
    extract_notebook_id: Optional[Callable] = None,
    extract_query: Optional[Callable] = None,
    cache_key_generator: Optional[Callable] = None,
    enable_caching: bool = True
):
    """
    Decorator to add duplicate execution monitoring to existing functions.
    
    Args:
        operation_type: Type of operation being monitored
        extract_request_id: Function to extract request_id from args/kwargs
        extract_conversation_id: Function to extract conversation_id from args/kwargs
        extract_notebook_id: Function to extract notebook_id from args/kwargs
        extract_query: Function to extract query from args/kwargs
        cache_key_generator: Function to generate cache key from args/kwargs
        enable_caching: Whether to enable result caching
    
    Usage:
        @monitor_duplicate_execution(
            DuplicateOperationType.INTENT_ANALYSIS,
            extract_request_id=lambda *args, **kwargs: kwargs.get('request_id'),
            extract_query=lambda *args, **kwargs: args[0] if args else None
        )
        async def analyze_intent(query: str, request_id: str = None):
            # Existing function implementation
            pass
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract context using provided functions or defaults
            request_id = _safe_extract(extract_request_id, args, kwargs, 'request_id')
            conversation_id = _safe_extract(extract_conversation_id, args, kwargs, 'conversation_id')
            notebook_id = _safe_extract(extract_notebook_id, args, kwargs, 'notebook_id')
            query = _safe_extract(extract_query, args, kwargs, 'query')
            
            # Generate cache key if generator provided
            cache_key = None
            if cache_key_generator and enable_caching:
                try:
                    cache_key = cache_key_generator(*args, **kwargs)
                except Exception as e:
                    logger.debug(f"Cache key generation failed for {func.__name__}: {e}")
            
            # Create operation context
            context = OperationContext(
                operation_type=operation_type,
                request_id=str(request_id) if request_id else 'unknown',
                conversation_id=conversation_id,
                notebook_id=notebook_id,
                query=query,
                cache_key=cache_key,
                user_context=kwargs
            )
            
            # Track operation
            tracker = get_operation_performance_tracker()
            async with tracker.track_operation(context) as result:
                if not result.cache_hit:
                    # Execute original function
                    if asyncio.iscoroutinefunction(func):
                        result.result_data = await func(*args, **kwargs)
                    else:
                        result.result_data = func(*args, **kwargs)
                    result.success = True
            
            return result.result_data
        
        return wrapper
    return decorator


def _safe_extract(extractor: Optional[Callable], args: tuple, kwargs: dict, default_key: str) -> Any:
    """Safely extract value using extractor function or default key lookup"""
    if extractor:
        try:
            return extractor(*args, **kwargs)
        except Exception:
            pass
    
    # Fallback to default key lookup in kwargs
    return kwargs.get(default_key)


def monitor_notebook_operation(
    operation_type: DuplicateOperationType,
    enable_caching: bool = True
):
    """
    Specialized decorator for notebook operations that follows notebook patterns.
    Automatically extracts notebook_id, request_id, and conversation_id.
    
    Usage:
        @monitor_notebook_operation(DuplicateOperationType.RAG_QUERY)
        async def query_notebook(notebook_id: str, query: str, conversation_id: str = None):
            # Implementation
            pass
    """
    return monitor_duplicate_execution(
        operation_type=operation_type,
        extract_request_id=lambda *args, **kwargs: kwargs.get('request_id') or kwargs.get('conversation_id'),
        extract_conversation_id=lambda *args, **kwargs: kwargs.get('conversation_id'),
        extract_notebook_id=lambda *args, **kwargs: kwargs.get('notebook_id') or (args[0] if args else None),
        extract_query=lambda *args, **kwargs: kwargs.get('query') or (args[1] if len(args) > 1 else None),
        cache_key_generator=_notebook_cache_key_generator if enable_caching else None,
        enable_caching=enable_caching
    )


def monitor_ai_service_operation(
    operation_type: DuplicateOperationType,
    enable_caching: bool = True
):
    """
    Specialized decorator for AI service operations (task planning, intent analysis, etc.).
    
    Usage:
        @monitor_ai_service_operation(DuplicateOperationType.INTENT_ANALYSIS)
        async def analyze_query_intent(query: str, request_id: str = None):
            # Implementation  
            pass
    """
    return monitor_duplicate_execution(
        operation_type=operation_type,
        extract_request_id=lambda *args, **kwargs: kwargs.get('request_id'),
        extract_query=lambda *args, **kwargs: kwargs.get('query') or (args[0] if args else None),
        cache_key_generator=_ai_service_cache_key_generator if enable_caching else None,
        enable_caching=enable_caching
    )


def _notebook_cache_key_generator(*args, **kwargs) -> str:
    """Generate cache key for notebook operations"""
    notebook_id = kwargs.get('notebook_id') or (args[0] if args else 'unknown')
    query = kwargs.get('query') or (args[1] if len(args) > 1 else '')
    conversation_id = kwargs.get('conversation_id', '')
    
    cache_input = f"notebook:{notebook_id}:{query}:{conversation_id}"
    return hashlib.sha256(cache_input.encode()).hexdigest()


def _ai_service_cache_key_generator(*args, **kwargs) -> str:
    """Generate cache key for AI service operations"""
    query = kwargs.get('query') or (args[0] if args else '')
    request_id = kwargs.get('request_id', '')
    
    cache_input = f"ai_service:{query}:{request_id}"
    return hashlib.sha256(cache_input.encode()).hexdigest()


class MonitoringIntegrationManager:
    """
    Manager for integrating monitoring with existing services.
    Provides utilities for adding monitoring to existing code with minimal changes.
    """
    
    def __init__(self):
        """Initialize MonitoringIntegrationManager"""
        self.duplicate_monitor = get_duplicate_execution_monitor()
        self.performance_tracker = get_operation_performance_tracker()
        
        # Integration status tracking
        self.integrated_functions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("MonitoringIntegrationManager initialized")
    
    async def validate_integration(self, function_name: str) -> Dict[str, Any]:
        """
        Validate that a function has been properly integrated with monitoring
        
        Args:
            function_name: Name of the function to validate
            
        Returns:
            Validation results and recommendations
        """
        validation_result = {
            'function_name': function_name,
            'is_integrated': function_name in self.integrated_functions,
            'integration_details': self.integrated_functions.get(function_name, {}),
            'recommendations': [],
            'validation_timestamp': time.time()
        }
        
        if not validation_result['is_integrated']:
            validation_result['recommendations'].extend([
                f"Add duplicate execution monitoring to {function_name}",
                f"Implement performance tracking for {function_name}",
                f"Consider adding caching if {function_name} is expensive"
            ])
        
        return validation_result
    
    def register_integration(
        self,
        function_name: str,
        operation_type: DuplicateOperationType,
        integration_details: Dict[str, Any]
    ):
        """
        Register that a function has been integrated with monitoring
        
        Args:
            function_name: Name of the integrated function
            operation_type: Type of operation being monitored
            integration_details: Details about the integration
        """
        self.integrated_functions[function_name] = {
            'operation_type': operation_type.value,
            'integration_timestamp': time.time(),
            'details': integration_details
        }
        
        logger.info(f"Registered monitoring integration for {function_name}")
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """
        Get overall integration status across the system
        
        Returns:
            Comprehensive integration status report
        """
        try:
            # Get list of functions that should be monitored
            key_functions_to_monitor = [
                'analyze_query_intent',  # Query intent analysis
                'generate_execution_plan',  # Task planning
                'embed_query',  # Query embedding
                'extract_batch_data',  # Batch extraction
                'verify_completeness',  # Verification
                'notebook_chat',  # Notebook chat
                'rag_query'  # RAG queries
            ]
            
            integration_status = {
                'total_functions_monitored': len(self.integrated_functions),
                'key_functions_status': {},
                'integration_coverage': 0.0,
                'missing_integrations': [],
                'integration_health': 'unknown'
            }
            
            # Check each key function
            for func_name in key_functions_to_monitor:
                is_integrated = func_name in self.integrated_functions
                integration_status['key_functions_status'][func_name] = {
                    'integrated': is_integrated,
                    'details': self.integrated_functions.get(func_name, {})
                }
                
                if not is_integrated:
                    integration_status['missing_integrations'].append(func_name)
            
            # Calculate coverage
            integrated_count = sum(
                1 for status in integration_status['key_functions_status'].values()
                if status['integrated']
            )
            integration_status['integration_coverage'] = integrated_count / len(key_functions_to_monitor)
            
            # Determine health
            if integration_status['integration_coverage'] >= 0.9:
                integration_status['integration_health'] = 'excellent'
            elif integration_status['integration_coverage'] >= 0.7:
                integration_status['integration_health'] = 'good'
            elif integration_status['integration_coverage'] >= 0.5:
                integration_status['integration_health'] = 'fair'
            else:
                integration_status['integration_health'] = 'poor'
            
            return integration_status
        
        except Exception as e:
            logger.error(f"Error getting integration status: {e}")
            return {
                'error': str(e),
                'timestamp': time.time()
            }
    
    async def create_integration_guide(self) -> Dict[str, Any]:
        """
        Create integration guide for adding monitoring to existing functions
        
        Returns:
            Step-by-step guide for integrating monitoring
        """
        return {
            'integration_guide': {
                'overview': 'How to add duplicate execution monitoring to existing functions',
                'steps': [
                    {
                        'step': 1,
                        'title': 'Import monitoring decorators',
                        'code': 'from app.services.monitoring_integrations import monitor_duplicate_execution, DuplicateOperationType'
                    },
                    {
                        'step': 2,
                        'title': 'Add decorator to function',
                        'code': '@monitor_duplicate_execution(DuplicateOperationType.INTENT_ANALYSIS)\nasync def analyze_intent(query: str, request_id: str):\n    # existing implementation'
                    },
                    {
                        'step': 3,
                        'title': 'Configure extraction functions (optional)',
                        'code': '@monitor_duplicate_execution(\n    DuplicateOperationType.TASK_PLANNING,\n    extract_request_id=lambda *args, **kwargs: kwargs.get("request_id"),\n    extract_query=lambda *args, **kwargs: args[0]\n)'
                    },
                    {
                        'step': 4,
                        'title': 'Add cache key generator (optional)',
                        'code': 'cache_key_gen = lambda query, request_id, **kw: f"intent:{hash(query)}:{request_id}"\n@monitor_duplicate_execution(\n    DuplicateOperationType.INTENT_ANALYSIS,\n    cache_key_generator=cache_key_gen\n)'
                    }
                ],
                'operation_types': [
                    {
                        'type': op_type.value,
                        'description': self._get_operation_description(op_type),
                        'typical_functions': self._get_typical_functions(op_type)
                    }
                    for op_type in DuplicateOperationType
                ],
                'best_practices': [
                    'Always provide request_id extraction for proper tracking',
                    'Use caching for expensive operations that can be repeated',
                    'Choose appropriate operation types for accurate monitoring',
                    'Test integration with existing functionality',
                    'Monitor performance impact of monitoring overhead'
                ],
                'common_patterns': {
                    'notebook_services': {
                        'example': '@monitor_notebook_operation(DuplicateOperationType.RAG_QUERY)',
                        'description': 'For notebook-related operations with automatic parameter extraction'
                    },
                    'ai_services': {
                        'example': '@monitor_ai_service_operation(DuplicateOperationType.INTENT_ANALYSIS)',
                        'description': 'For AI service operations like intent analysis and task planning'
                    },
                    'custom_integration': {
                        'example': 'async with tracker.track_operation(context) as result:',
                        'description': 'For manual integration with full control over context'
                    }
                }
            }
        }
    
    def _get_operation_description(self, operation_type: DuplicateOperationType) -> str:
        """Get description for operation type"""
        descriptions = {
            DuplicateOperationType.INTENT_ANALYSIS: "AI-powered understanding of user query intent and requirements",
            DuplicateOperationType.TASK_PLANNING: "AI generation of execution plans for complex queries",
            DuplicateOperationType.QUERY_EMBEDDING: "Vector embedding generation for semantic search",
            DuplicateOperationType.BATCH_EXTRACTION: "Large-scale data extraction and processing operations",
            DuplicateOperationType.VERIFICATION: "AI verification and completeness checking",
            DuplicateOperationType.CACHE_OPERATION: "Cache read/write and invalidation operations",
            DuplicateOperationType.LLM_CALL: "Direct Large Language Model API calls",
            DuplicateOperationType.RAG_QUERY: "Retrieval Augmented Generation query processing",
            DuplicateOperationType.NOTEBOOK_CHAT: "Notebook-specific chat and conversation handling"
        }
        return descriptions.get(operation_type, "Standard operation type")
    
    def _get_typical_functions(self, operation_type: DuplicateOperationType) -> List[str]:
        """Get typical function names for operation type"""
        typical_functions = {
            DuplicateOperationType.INTENT_ANALYSIS: [
                'analyze_query_intent', 'classify_query_type', 'extract_user_intent'
            ],
            DuplicateOperationType.TASK_PLANNING: [
                'generate_execution_plan', 'create_task_plan', 'plan_query_execution'
            ],
            DuplicateOperationType.QUERY_EMBEDDING: [
                'embed_query', 'generate_embedding', 'create_query_vector'
            ],
            DuplicateOperationType.BATCH_EXTRACTION: [
                'extract_batch_data', 'process_batch_query', 'execute_batch_retrieval'
            ],
            DuplicateOperationType.VERIFICATION: [
                'verify_completeness', 'validate_results', 'check_result_quality'
            ],
            DuplicateOperationType.RAG_QUERY: [
                'query_rag', 'execute_rag_query', 'notebook_rag_query'
            ],
            DuplicateOperationType.NOTEBOOK_CHAT: [
                'notebook_chat', 'process_notebook_message', 'handle_notebook_conversation'
            ]
        }
        return typical_functions.get(operation_type, [])


# Convenience functions for common integration patterns
async def track_ai_task_planning(func: Callable, query: str, request_id: str = None, **kwargs):
    """
    Track AI task planning operation
    
    Usage:
        result = await track_ai_task_planning(
            ai_task_planner.generate_execution_plan,
            query="user query",
            request_id="req_123"
        )
    """
    context = OperationContext(
        operation_type=DuplicateOperationType.TASK_PLANNING,
        request_id=request_id or 'unknown',
        query=query,
        cache_key=f"task_planning:{hashlib.sha256(f'{query}:{request_id}'.encode()).hexdigest()}",
        user_context=kwargs
    )
    
    tracker = get_operation_performance_tracker()
    async with tracker.track_operation(context) as result:
        if not result.cache_hit:
            if asyncio.iscoroutinefunction(func):
                result.result_data = await func(query, request_id=request_id, **kwargs)
            else:
                result.result_data = func(query, request_id=request_id, **kwargs)
            result.success = True
    
    return result.result_data


async def track_query_intent_analysis(func: Callable, query: str, request_id: str = None, **kwargs):
    """
    Track query intent analysis operation
    
    Usage:
        result = await track_query_intent_analysis(
            intent_analyzer.analyze_intent,
            query="user query",
            request_id="req_123"
        )
    """
    context = OperationContext(
        operation_type=DuplicateOperationType.INTENT_ANALYSIS,
        request_id=request_id or 'unknown',
        query=query,
        cache_key=f"intent_analysis:{hashlib.sha256(f'{query}:{request_id}'.encode()).hexdigest()}",
        user_context=kwargs
    )
    
    tracker = get_operation_performance_tracker()
    async with tracker.track_operation(context) as result:
        if not result.cache_hit:
            if asyncio.iscoroutinefunction(func):
                result.result_data = await func(query, request_id=request_id, **kwargs)
            else:
                result.result_data = func(query, request_id=request_id, **kwargs)
            result.success = True
    
    return result.result_data


async def track_batch_extraction(func: Callable, query: str, notebook_id: str, config: dict, **kwargs):
    """
    Track batch extraction operation
    
    Usage:
        result = await track_batch_extraction(
            batch_extractor.extract_data,
            query="extraction query",
            notebook_id="nb_123",
            config=extraction_config
        )
    """
    context = OperationContext(
        operation_type=DuplicateOperationType.BATCH_EXTRACTION,
        request_id=kwargs.get('request_id', 'unknown'),
        notebook_id=notebook_id,
        query=query,
        cache_key=f"batch_extraction:{hashlib.sha256(f'{query}:{notebook_id}:{json.dumps(config, sort_keys=True)}'.encode()).hexdigest()}",
        user_context={'config': config, **kwargs}
    )
    
    tracker = get_operation_performance_tracker()
    async with tracker.track_operation(context) as result:
        if not result.cache_hit:
            if asyncio.iscoroutinefunction(func):
                result.result_data = await func(query, notebook_id, config, **kwargs)
            else:
                result.result_data = func(query, notebook_id, config, **kwargs)
            result.success = True
    
    return result.result_data


class PerformanceMetricsCollector:
    """
    Utility for collecting custom performance metrics during operations.
    Can be used within monitored functions to add additional context.
    """
    
    def __init__(self):
        self.custom_metrics: Dict[str, List[Tuple[datetime, Any]]] = {}
        self.metrics_lock = asyncio.Lock()
    
    async def record_metric(self, metric_name: str, value: Any, context: Optional[Dict[str, Any]] = None):
        """
        Record a custom metric during operation execution
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            context: Optional context information
        """
        async with self.metrics_lock:
            if metric_name not in self.custom_metrics:
                self.custom_metrics[metric_name] = []
            
            self.custom_metrics[metric_name].append((
                datetime.now(),
                {
                    'value': value,
                    'context': context or {}
                }
            ))
            
            # Keep only recent metrics (last hour)
            cutoff = datetime.now() - timedelta(hours=1)
            self.custom_metrics[metric_name] = [
                (timestamp, data) for timestamp, data in self.custom_metrics[metric_name]
                if timestamp > cutoff
            ]
    
    async def get_metric_summary(self, metric_name: str) -> Dict[str, Any]:
        """Get summary statistics for a custom metric"""
        async with self.metrics_lock:
            if metric_name not in self.custom_metrics:
                return {'error': f'Metric {metric_name} not found'}
            
            data_points = self.custom_metrics[metric_name]
            if not data_points:
                return {'metric_name': metric_name, 'data_points': 0}
            
            values = []
            for _, data in data_points:
                value = data['value']
                if isinstance(value, (int, float)):
                    values.append(value)
            
            if values:
                return {
                    'metric_name': metric_name,
                    'data_points': len(data_points),
                    'numeric_points': len(values),
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'latest_value': values[-1] if values else None,
                    'collection_timespan': (data_points[-1][0] - data_points[0][0]).total_seconds() if len(data_points) > 1 else 0
                }
            else:
                return {
                    'metric_name': metric_name,
                    'data_points': len(data_points),
                    'numeric_points': 0,
                    'note': 'No numeric values recorded'
                }


# Global instances
integration_manager = MonitoringIntegrationManager()
custom_metrics_collector = PerformanceMetricsCollector()


def get_integration_manager() -> MonitoringIntegrationManager:
    """Get the global integration manager instance"""
    return integration_manager


def get_custom_metrics_collector() -> PerformanceMetricsCollector:
    """Get the global custom metrics collector instance"""
    return custom_metrics_collector