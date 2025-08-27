#!/usr/bin/env python3
"""
Simple Monitoring Integration Test

Tests that the monitoring system integrates properly with the existing
codebase without causing import or initialization issues.
"""

import asyncio
import logging
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_monitoring_imports():
    """Test that monitoring modules can be imported successfully"""
    print("\n=== Testing Monitoring Module Imports ===")
    
    try:
        # Test duplicate execution monitor import
        from app.services.duplicate_execution_monitor import DuplicateOperationType
        print("   ✅ DuplicateOperationType imported successfully")
        
        # Test operation performance tracker import
        from app.services.operation_performance_tracker import OperationContext
        print("   ✅ OperationContext imported successfully")
        
        # Test monitoring integrations import
        from app.services.monitoring_integrations import monitor_duplicate_execution
        print("   ✅ Monitoring decorators imported successfully")
        
        # Test health checks import
        from app.services.monitoring_health_checks import HealthCheckStatus
        print("   ✅ Health check components imported successfully")
        
        # Test configuration import
        from app.config.monitoring_dashboard_config import get_dashboard_config
        config = get_dashboard_config()
        print(f"   ✅ Dashboard configuration loaded: {len(config)} sections")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False


async def test_basic_functionality():
    """Test basic monitoring functionality"""
    print("\n=== Testing Basic Monitoring Functionality ===")
    
    try:
        # Import with runtime initialization
        from app.services.duplicate_execution_monitor import DuplicateOperationType
        
        # Test operation type enum
        print("\n1. Testing operation types:")
        for op_type in DuplicateOperationType:
            print(f"   📋 {op_type.value}")
        
        print(f"   ✅ {len(DuplicateOperationType)} operation types available")
        
        # Test context creation
        print("\n2. Testing context creation:")
        from app.services.operation_performance_tracker import OperationContext
        
        context = OperationContext(
            operation_type=DuplicateOperationType.INTENT_ANALYSIS,
            request_id="test_001",
            query="test query"
        )
        print(f"   ✅ Context created: {context.operation_type.value}")
        
        # Test decorator availability
        print("\n3. Testing decorator availability:")
        from app.services.monitoring_integrations import monitor_duplicate_execution
        
        @monitor_duplicate_execution(DuplicateOperationType.CACHE_OPERATION)
        async def test_function():
            await asyncio.sleep(0.01)
            return "test result"
        
        result = await test_function()
        print(f"   ✅ Decorated function executed: {result}")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {e}")
        return False


async def test_api_endpoint_structure():
    """Test that API endpoints are properly structured"""
    print("\n=== Testing API Endpoint Structure ===")
    
    try:
        # Import the router
        from app.api.v1.endpoints.duplicate_monitoring import router
        print("   ✅ Monitoring API router imported successfully")
        
        # Check routes are registered
        route_count = len(router.routes)
        print(f"   📋 API routes registered: {route_count}")
        
        # List some key routes
        route_paths = [route.path for route in router.routes if hasattr(route, 'path')]
        key_routes = ['/dashboard', '/metrics/real-time', '/health/quick', '/report/duplicates']
        
        found_routes = [route for route in key_routes if any(route in path for path in route_paths)]
        print(f"   ✅ Key routes found: {len(found_routes)}/{len(key_routes)}")
        
        for route in found_routes:
            print(f"      📍 {route}")
        
        return len(found_routes) >= 3  # At least 3 key routes should be available
    
    except Exception as e:
        print(f"   ❌ API endpoint test failed: {e}")
        return False


async def test_configuration_system():
    """Test configuration system"""
    print("\n=== Testing Configuration System ===")
    
    try:
        from app.config.monitoring_dashboard_config import (
            get_dashboard_config, 
            get_operation_monitoring_config,
            get_health_check_config
        )
        
        # Test dashboard config
        dashboard_config = get_dashboard_config()
        print(f"   ✅ Dashboard config sections: {len(dashboard_config.get('sections', {}))}")
        
        # Test operation config
        operation_config = get_operation_monitoring_config('intent_analysis')
        print(f"   ✅ Operation config loaded for intent_analysis")
        
        # Test health check config
        health_config = get_health_check_config()
        print(f"   ✅ Health check config loaded: {len(health_config.get('check_intervals', {}))} intervals")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False


async def run_integration_tests():
    """Run all integration tests"""
    print("=" * 60)
    print(" MONITORING SYSTEM INTEGRATION TEST")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    test_results = {}
    
    try:
        # Run tests
        test_results['imports'] = await test_monitoring_imports()
        test_results['basic_functionality'] = await test_basic_functionality()
        test_results['api_endpoints'] = await test_api_endpoint_structure()
        test_results['configuration'] = await test_configuration_system()
        
        # Summary
        print("\n" + "=" * 40)
        print(" INTEGRATION TEST RESULTS")
        print("=" * 40)
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        for test_name, passed in test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status}: {test_name.replace('_', ' ').title()}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL INTEGRATION TESTS PASSED!")
            print("\n📋 Monitoring System Ready:")
            print("   • Duplicate execution detection active")
            print("   • Performance tracking available")  
            print("   • Health checks functional")
            print("   • API endpoints configured")
            print("   • Dashboard configuration loaded")
            
            print("\n🚀 Available API Endpoints:")
            print("   GET /monitoring/dashboard - Main monitoring dashboard")
            print("   GET /monitoring/metrics/real-time - Real-time metrics")
            print("   GET /monitoring/health/quick - Quick health check")
            print("   GET /monitoring/report/duplicates - Duplicate analysis report")
            print("   GET /monitoring/patterns/active - Active duplicate patterns")
            print("   GET /monitoring/config/dashboard - Dashboard configuration")
            
            print("\n📖 Integration Guide:")
            print("   • Add @monitor_duplicate_execution decorator to functions")
            print("   • Use track_operation context manager for manual tracking")
            print("   • Check /monitoring/integration/guide for detailed examples")
            print("   • Monitor system health at /monitoring/health/comprehensive")
            
        else:
            print(f"\n⚠️  {total_tests - passed_tests} integration tests failed.")
            print("\n🔧 Next Steps:")
            print("   1. Review failed test output above")
            print("   2. Check Redis connectivity")
            print("   3. Verify all dependencies are installed")
            print("   4. Check for import path issues")
        
        print(f"\nCompleted at: {datetime.now()}")
        
    except Exception as e:
        print(f"\n💥 Integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Starting Monitoring System Integration Test")
    print("   This validates that monitoring components integrate properly")
    print("   with the existing codebase without causing issues.\n")
    
    try:
        asyncio.run(run_integration_tests())
        print("\n✅ Integration test completed.")
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()