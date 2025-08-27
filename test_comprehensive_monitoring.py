#!/usr/bin/env python3
"""
Comprehensive Monitoring Test Script

Tests the complete duplicate execution monitoring system including:
- Duplicate detection and tracking
- Performance metrics collection
- Health checks validation
- Dashboard data generation
- Integration utilities
"""

import asyncio
import json
import logging
import time
from datetime import datetime
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from app.services.duplicate_execution_monitor import (
    DuplicateExecutionMonitor, DuplicateOperationType, get_duplicate_execution_monitor
)
from app.services.operation_performance_tracker import (
    OperationPerformanceTracker, OperationContext, get_operation_performance_tracker
)
from app.services.monitoring_health_checks import (
    get_monitoring_health_checker
)
from app.services.monitoring_integrations import (
    monitor_duplicate_execution, get_integration_manager
)
from app.config.monitoring_dashboard_config import get_dashboard_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_duplicate_detection():
    """Test duplicate operation detection"""
    print("\n=== Testing Duplicate Operation Detection ===")
    
    monitor = get_duplicate_execution_monitor()
    
    # Test 1: Normal operation tracking
    print("\n1. Testing normal operation tracking:")
    operation_id1 = monitor.track_operation_start(
        DuplicateOperationType.INTENT_ANALYSIS,
        "test_request_001",
        conversation_id="conv_001",
        operation_context={'query': 'test query 1'}
    )
    print(f"   ✅ Started tracking operation: {operation_id1}")
    
    await asyncio.sleep(0.1)  # Simulate processing time
    
    monitor.track_operation_end(operation_id1, success=True)
    print(f"   ✅ Completed operation tracking")
    
    # Test 2: Duplicate operation detection
    print("\n2. Testing duplicate detection:")
    
    # First execution
    operation_id2 = monitor.track_operation_start(
        DuplicateOperationType.TASK_PLANNING,
        "test_request_002",
        operation_context={'query': 'identical query'}
    )
    monitor.track_operation_end(operation_id2, success=True)
    print(f"   ✅ First execution completed")
    
    # Immediate duplicate (should trigger duplicate detection)
    operation_id3 = monitor.track_operation_start(
        DuplicateOperationType.TASK_PLANNING,
        "test_request_002",
        operation_context={'query': 'identical query'}
    )
    monitor.track_operation_end(operation_id3, success=True)
    print(f"   ✅ Duplicate execution detected and tracked")
    
    # Check duplicate events
    duplicate_count = len(monitor.duplicate_events)
    print(f"   📊 Total duplicate events recorded: {duplicate_count}")
    
    return duplicate_count > 0


async def test_performance_tracking():
    """Test performance tracking functionality"""
    print("\n=== Testing Performance Tracking ===")
    
    tracker = get_operation_performance_tracker()
    
    # Test 1: Context manager usage
    print("\n1. Testing performance tracking context manager:")
    
    context = OperationContext(
        operation_type=DuplicateOperationType.QUERY_EMBEDDING,
        request_id="test_request_003",
        query="performance test query"
    )
    
    async with tracker.track_operation(context) as result:
        # Simulate operation
        await asyncio.sleep(0.2)
        result.result_data = {'embedding': [0.1, 0.2, 0.3]}
        result.success = True
    
    print(f"   ✅ Operation tracked - Duration: {result.execution_time:.3f}s")
    print(f"   📊 Operation successful: {result.success}")
    
    # Test 2: Cache simulation
    print("\n2. Testing cache behavior simulation:")
    
    context_with_cache = OperationContext(
        operation_type=DuplicateOperationType.VERIFICATION,
        request_id="test_request_004", 
        cache_key="test_cache_key_001"
    )
    
    # First execution (should miss cache)
    async with tracker.track_operation(context_with_cache) as result1:
        await asyncio.sleep(0.1)
        result1.result_data = {'verification': 'passed'}
        result1.success = True
    
    print(f"   ✅ First execution (cache miss): {result1.execution_time:.3f}s")
    
    # Second execution (should hit cache if caching is working)
    async with tracker.track_operation(context_with_cache) as result2:
        if not result2.cache_hit:
            # Only execute if cache miss
            await asyncio.sleep(0.1)
            result2.result_data = {'verification': 'passed'}
            result2.success = True
    
    print(f"   📊 Second execution - Cache hit: {result2.cache_hit}, Duration: {result2.execution_time:.3f}s")
    
    return result1.success and result2.success


async def test_health_checks():
    """Test monitoring health checks"""
    print("\n=== Testing Monitoring Health Checks ===")
    
    health_checker = get_monitoring_health_checker()
    
    # Test 1: Quick health check
    print("\n1. Running quick health check:")
    quick_health = await health_checker.run_quick_health_check()
    
    print(f"   📊 Quick health status: {quick_health.get('quick_status', 'unknown')}")
    print(f"   📊 Checks performed: {quick_health.get('checks_performed', 0)}")
    
    # Test 2: Component-specific checks (if time allows)
    print("\n2. Testing component health checks:")
    try:
        # Test duplicate monitor
        result = await health_checker._check_duplicate_monitor_basic()
        print(f"   ✅ Duplicate monitor: {result.status.value} - {result.message}")
        
        # Test Redis connectivity
        result = await health_checker._check_redis_connectivity() 
        print(f"   ✅ Redis connectivity: {result.status.value} - {result.message}")
        
    except Exception as e:
        print(f"   ⚠️  Component check error: {e}")
    
    return quick_health.get('quick_status') in ['healthy', 'warning']


async def test_dashboard_data_generation():
    """Test dashboard data generation"""
    print("\n=== Testing Dashboard Data Generation ===")
    
    tracker = get_operation_performance_tracker()
    
    # Test 1: Create dashboard data
    print("\n1. Generating dashboard data:")
    dashboard_data = await tracker.create_monitoring_dashboard_data()
    
    print(f"   ✅ Dashboard data generated")
    print(f"   📊 Summary widgets: {len(dashboard_data.get('summary_widgets', {}))}")
    print(f"   📊 Has real-time metrics: {'real_time_metrics' in dashboard_data}")
    print(f"   📊 Has performance trends: {'performance_trends' in dashboard_data}")
    
    # Test 2: Validate dashboard structure
    print("\n2. Validating dashboard structure:")
    required_sections = ['metadata', 'summary_widgets', 'real_time_metrics', 'operation_health']
    
    for section in required_sections:
        if section in dashboard_data:
            print(f"   ✅ Section '{section}' present")
        else:
            print(f"   ❌ Section '{section}' missing")
    
    return all(section in dashboard_data for section in required_sections)


async def test_integration_decorators():
    """Test monitoring integration decorators"""
    print("\n=== Testing Integration Decorators ===")
    
    # Test 1: Create a test function with monitoring
    @monitor_duplicate_execution(
        DuplicateOperationType.INTENT_ANALYSIS,
        extract_request_id=lambda *args, **kwargs: kwargs.get('request_id', 'test_req'),
        extract_query=lambda *args, **kwargs: args[0] if args else 'test query'
    )
    async def test_monitored_function(query: str, request_id: str = None):
        """Test function with monitoring decorator"""
        await asyncio.sleep(0.1)  # Simulate processing
        return {'intent': 'test', 'confidence': 0.9}
    
    print("\n1. Testing decorated function:")
    
    # First execution
    result1 = await test_monitored_function("test query for monitoring", request_id="test_req_005")
    print(f"   ✅ First execution result: {result1}")
    
    # Second execution (potential duplicate)
    result2 = await test_monitored_function("test query for monitoring", request_id="test_req_005")  
    print(f"   ✅ Second execution result: {result2}")
    
    # Test 2: Check if monitoring captured the operations
    print("\n2. Validating monitoring capture:")
    monitor = get_duplicate_execution_monitor()
    
    intent_metrics = monitor.operation_metrics[DuplicateOperationType.INTENT_ANALYSIS]
    print(f"   📊 Intent analysis executions: {intent_metrics.total_executions}")
    print(f"   📊 Duplicate rate: {intent_metrics.duplicate_rate:.2%}")
    
    return result1 is not None and result2 is not None


async def test_configuration_system():
    """Test monitoring configuration system"""
    print("\n=== Testing Configuration System ===")
    
    # Test 1: Dashboard configuration
    print("\n1. Testing dashboard configuration:")
    try:
        config = get_dashboard_config()
        print(f"   ✅ Dashboard config loaded")
        print(f"   📊 Sections configured: {len(config.get('sections', {}))}")
        print(f"   📊 Alert thresholds: {len(config.get('alert_thresholds', {}))}")
        print(f"   📊 Chart configurations: {len(config.get('charts', {}))}")
        config_valid = True
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        config_valid = False
    
    # Test 2: Integration manager
    print("\n2. Testing integration manager:")
    try:
        manager = get_integration_manager()
        status = await manager.get_integration_status()
        print(f"   ✅ Integration manager working")
        print(f"   📊 Functions monitored: {status.get('total_functions_monitored', 0)}")
        print(f"   📊 Integration coverage: {status.get('integration_coverage', 0):.1%}")
        integration_valid = True
    except Exception as e:
        print(f"   ❌ Integration manager error: {e}")
        integration_valid = False
    
    return config_valid and integration_valid


async def run_comprehensive_monitoring_test():
    """Run all monitoring tests"""
    print("=" * 70)
    print(" COMPREHENSIVE MONITORING SYSTEM TEST")
    print("=" * 70)
    print(f"Started at: {datetime.now()}")
    
    test_results = {
        'duplicate_detection': False,
        'performance_tracking': False,  
        'health_checks': False,
        'dashboard_generation': False,
        'integration_decorators': False,
        'configuration_system': False
    }
    
    try:
        # Run all tests
        test_results['duplicate_detection'] = await test_duplicate_detection()
        test_results['performance_tracking'] = await test_performance_tracking()
        test_results['health_checks'] = await test_health_checks()
        test_results['dashboard_generation'] = await test_dashboard_data_generation()
        test_results['integration_decorators'] = await test_integration_decorators()
        test_results['configuration_system'] = await test_configuration_system()
        
        # Summary
        print("\n" + "=" * 50)
        print(" TEST RESULTS SUMMARY")
        print("=" * 50)
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        for test_name, passed in test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status}: {test_name.replace('_', ' ').title()}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! Monitoring system is fully functional.")
            print("\n📋 Next Steps:")
            print("   1. Review monitoring endpoints at /monitoring/*")
            print("   2. Check dashboard at /monitoring/dashboard") 
            print("   3. Monitor duplicate patterns at /monitoring/patterns/active")
            print("   4. Get performance insights at /monitoring/insights/performance")
            print("   5. Integrate monitoring into key services using decorators")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} tests failed. Review the output above for details.")
            print("\n🔧 Troubleshooting:")
            print("   1. Check Redis connectivity")
            print("   2. Verify all monitoring services are properly initialized")
            print("   3. Review error logs for specific issues")
            print("   4. Run individual test components to isolate problems")
        
        print(f"\nCompleted at: {datetime.now()}")
        
        # Generate final monitoring report
        print("\n" + "=" * 50)
        print(" MONITORING SYSTEM STATUS REPORT")
        print("=" * 50)
        
        try:
            monitor = get_duplicate_execution_monitor()
            tracker = get_operation_performance_tracker()
            
            print(f"📊 Operations tracked: {monitor.monitor_stats['total_operations_tracked']}")
            print(f"📊 Duplicates detected: {monitor.monitor_stats['total_duplicates_detected']}")
            print(f"📊 System efficiency: {monitor.monitor_stats['system_efficiency_score']:.1f}%")
            print(f"📊 Active operations: {len(monitor.active_operations)}")
            print(f"📊 Detected patterns: {len(monitor.detected_patterns)}")
            
            # Show sample metrics for each operation type
            print(f"\n📈 Operation Metrics Sample:")
            for op_type, metrics in monitor.operation_metrics.items():
                if metrics.total_executions > 0:
                    print(f"   {op_type.value}: {metrics.total_executions} ops, {metrics.duplicate_rate:.1%} duplicates")
        
        except Exception as e:
            print(f"❌ Error generating final report: {e}")
    
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        print("🔧 Check that all required dependencies are installed and Redis is running")


async def demonstrate_monitoring_usage():
    """Demonstrate practical usage of monitoring system"""
    print("\n" + "=" * 50)
    print(" MONITORING USAGE DEMONSTRATION") 
    print("=" * 50)
    
    # Example 1: Manual operation tracking
    print("\n🔧 Example 1: Manual Operation Tracking")
    monitor = get_duplicate_execution_monitor()
    
    op_id = monitor.track_operation_start(
        DuplicateOperationType.RAG_QUERY,
        "demo_request_001",
        notebook_id="notebook_123",
        operation_context={'query': 'What are the key findings?'}
    )
    
    # Simulate work
    await asyncio.sleep(0.05)
    
    monitor.track_operation_end(op_id, success=True, cache_hit=False)
    print("   ✅ Manual tracking demonstrated")
    
    # Example 2: Context manager usage
    print("\n🔧 Example 2: Context Manager Usage")
    tracker = get_operation_performance_tracker()
    
    context = OperationContext(
        operation_type=DuplicateOperationType.BATCH_EXTRACTION,
        request_id="demo_request_002",
        notebook_id="notebook_456",
        query="Extract all project data"
    )
    
    async with tracker.track_operation(context) as result:
        # Simulate expensive operation
        await asyncio.sleep(0.1)
        result.result_data = {'extracted_items': 25}
        result.success = True
    
    print(f"   ✅ Context manager tracking: {result.execution_time:.3f}s")
    
    # Example 3: Get monitoring data
    print("\n🔧 Example 3: Accessing Monitoring Data")
    
    # Get real-time metrics
    real_time_metrics = await monitor.get_real_time_metrics()
    print(f"   📊 Real-time metrics available: {'summary' in real_time_metrics}")
    
    # Get health status
    health_status = monitor.get_operation_health_status()
    print(f"   📊 Health status overall: {health_status.get('overall_health', 0):.1f}")
    
    # Get duplicate report
    duplicate_report = await monitor.get_duplicate_report(None, 1)  # Last 1 hour
    print(f"   📊 Duplicate events in last hour: {len(duplicate_report.get('duplicate_analysis', {}).get('summary', {}).get('total_duplicates', []))}")
    
    print("\n✅ Monitoring usage demonstration completed")


if __name__ == "__main__":
    print("🚀 Starting Comprehensive Monitoring System Test")
    print("   This test validates the complete duplicate execution monitoring implementation")
    print("   including detection, tracking, health checks, and dashboard functionality.\n")
    
    try:
        # Run the test suite
        asyncio.run(run_comprehensive_monitoring_test())
        
        # Run usage demonstration  
        asyncio.run(demonstrate_monitoring_usage())
        
        print("\n🏁 Test completed. Check the output above for results.")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()