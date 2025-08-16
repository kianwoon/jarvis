"""
Test Performance

Performance benchmarks and tests for the radiating system.
"""

import unittest
import asyncio
import time
import random
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from app.services.radiating.optimization.query_optimizer import QueryOptimizer, QueryType
from app.services.radiating.optimization.cache_strategy import CacheStrategy, CacheTier
from app.services.radiating.optimization.parallel_processor import ParallelProcessor, TaskPriority
from app.services.radiating.optimization.performance_monitor import PerformanceMonitor, MetricType


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        self.query_optimizer = QueryOptimizer()
        self.cache_strategy = CacheStrategy()
        self.parallel_processor = ParallelProcessor(max_workers=4)
        self.performance_monitor = PerformanceMonitor()
    
    def tearDown(self):
        """Clean up after tests"""
        self.parallel_processor.shutdown()
    
    def test_query_optimization_speed(self):
        """Test query optimization performance"""
        queries = [
            ("MATCH (n:Person)-[:KNOWS]->(m:Person) RETURN n, m", QueryType.TRAVERSAL),
            ("MATCH (n:Organization) WHERE n.name CONTAINS 'tech' RETURN n", QueryType.ENTITY),
            ("MATCH p=shortestPath((a)-[*]-(b)) RETURN p", QueryType.PATH),
            ("MATCH (n) RETURN COUNT(n)", QueryType.AGGREGATION)
        ]
        
        start_time = time.time()
        
        for query, query_type in queries:
            plan = asyncio.run(self.query_optimizer.optimize_query(
                query,
                query_type,
                parameters={'limit': 100}
            ))
            
            self.assertIsNotNone(plan)
            self.assertIsNotNone(plan.optimized_query)
            self.assertGreater(plan.estimated_cost, 0)
        
        elapsed = time.time() - start_time
        
        # Should optimize 4 queries in under 100ms
        self.assertLess(elapsed, 0.1)
        print(f"Query optimization: {len(queries)} queries in {elapsed:.3f}s")
    
    def test_cache_performance(self):
        """Test cache operation performance"""
        num_operations = 1000
        
        # Test write performance
        start_time = time.time()
        
        for i in range(num_operations):
            asyncio.run(self.cache_strategy.set(
                f"key_{i}",
                f"value_{i}",
                cache_type="entity",
                tiers=[CacheTier.MEMORY]
            ))
        
        write_time = time.time() - start_time
        writes_per_second = num_operations / write_time
        
        print(f"Cache writes: {writes_per_second:.0f} ops/sec")
        
        # Test read performance
        start_time = time.time()
        hits = 0
        
        for i in range(num_operations):
            value = asyncio.run(self.cache_strategy.get(
                f"key_{i}",
                cache_type="entity",
                check_tiers=[CacheTier.MEMORY]
            ))
            
            if value is not None:
                hits += 1
        
        read_time = time.time() - start_time
        reads_per_second = num_operations / read_time
        hit_rate = hits / num_operations
        
        print(f"Cache reads: {reads_per_second:.0f} ops/sec, hit rate: {hit_rate:.2%}")
        
        # Performance assertions
        self.assertGreater(writes_per_second, 5000)  # At least 5000 writes/sec
        self.assertGreater(reads_per_second, 10000)  # At least 10000 reads/sec
        self.assertEqual(hit_rate, 1.0)  # All should hit since we just wrote them
    
    def test_parallel_processing_performance(self):
        """Test parallel processing performance"""
        # Define a CPU-bound test function
        def cpu_bound_task(n):
            """Simulate CPU-bound work"""
            result = 0
            for i in range(n * 1000):
                result += i ** 2
            return result
        
        # Test data
        test_sizes = [10, 20, 30, 40, 50]
        
        # Sequential processing
        start_time = time.time()
        sequential_results = [cpu_bound_task(n) for n in test_sizes]
        sequential_time = time.time() - start_time
        
        # Parallel processing
        start_time = time.time()
        parallel_results = asyncio.run(
            self.parallel_processor.process(
                cpu_bound_task,
                test_sizes,
                priority=TaskPriority.HIGH
            )
        )
        parallel_time = time.time() - start_time
        
        # Calculate speedup
        speedup = sequential_time / parallel_time
        
        print(f"Sequential: {sequential_time:.3f}s, Parallel: {parallel_time:.3f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        # Should have significant speedup with 4 workers
        self.assertGreater(speedup, 1.5)
        
        # Results should match
        self.assertEqual(sequential_results, parallel_results)
    
    def test_memory_efficiency(self):
        """Test memory efficiency of caching"""
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Get baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Add large amount of cache entries
        large_data = "x" * 1000  # 1KB string
        num_entries = 1000
        
        for i in range(num_entries):
            asyncio.run(self.cache_strategy.set(
                f"large_key_{i}",
                large_data,
                cache_type="entity",
                tiers=[CacheTier.MEMORY]
            ))
        
        # Check memory after caching
        gc.collect()
        after_cache_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = after_cache_memory - baseline_memory
        
        print(f"Memory increase for {num_entries} entries: {memory_increase:.2f} MB")
        
        # Should not use excessive memory (less than 50MB for 1000 1KB entries)
        self.assertLess(memory_increase, 50)
    
    def test_performance_monitoring_overhead(self):
        """Test overhead of performance monitoring"""
        num_operations = 1000
        
        # Test without monitoring
        start_time = time.time()
        
        for i in range(num_operations):
            # Simulate operation
            time.sleep(0.0001)  # 0.1ms operation
        
        without_monitoring = time.time() - start_time
        
        # Test with monitoring
        start_time = time.time()
        
        for i in range(num_operations):
            op_id = f"op_{i}"
            self.performance_monitor.start_operation(op_id)
            
            # Simulate operation
            time.sleep(0.0001)  # 0.1ms operation
            
            self.performance_monitor.end_operation(op_id)
        
        with_monitoring = time.time() - start_time
        
        # Calculate overhead
        overhead = (with_monitoring - without_monitoring) / without_monitoring
        
        print(f"Monitoring overhead: {overhead:.1%}")
        
        # Overhead should be minimal (less than 10%)
        self.assertLess(overhead, 0.1)
    
    def test_batch_query_optimization(self):
        """Test batch query optimization performance"""
        # Create batch of queries
        queries = [
            (
                f"MATCH (n:Entity{i}) WHERE n.id = {i} RETURN n",
                QueryType.ENTITY,
                {'id': i}
            )
            for i in range(100)
        ]
        
        start_time = time.time()
        
        # Optimize batch
        plans = asyncio.run(
            self.query_optimizer.batch_optimize_queries(queries)
        )
        
        elapsed = time.time() - start_time
        
        print(f"Batch optimization: {len(queries)} queries in {elapsed:.3f}s")
        print(f"Average: {elapsed/len(queries)*1000:.2f}ms per query")
        
        # Should optimize 100 queries quickly
        self.assertLess(elapsed, 1.0)
        self.assertEqual(len(plans), len(queries))
    
    def test_cache_eviction_performance(self):
        """Test cache eviction performance"""
        # Fill cache to capacity
        cache_size = 1000
        
        for i in range(cache_size * 2):  # Overfill to trigger eviction
            asyncio.run(self.cache_strategy.set(
                f"evict_key_{i}",
                f"evict_value_{i}",
                cache_type="entity",
                tiers=[CacheTier.MEMORY]
            ))
        
        # Check cache size is maintained
        self.assertLessEqual(
            len(self.cache_strategy.memory_cache),
            cache_size
        )
        
        # Test eviction performance
        start_time = time.time()
        
        # Force eviction by adding more items
        for i in range(100):
            asyncio.run(self.cache_strategy.set(
                f"new_key_{i}",
                f"new_value_{i}",
                cache_type="entity",
                tiers=[CacheTier.MEMORY]
            ))
        
        eviction_time = time.time() - start_time
        
        print(f"Eviction of 100 items: {eviction_time:.3f}s")
        
        # Eviction should be fast
        self.assertLess(eviction_time, 0.1)


class TestPerformanceScaling(unittest.TestCase):
    """Test performance scaling characteristics"""
    
    def test_linear_scaling(self):
        """Test linear scaling of operations"""
        sizes = [100, 200, 400, 800]
        times = []
        
        cache = CacheStrategy()
        
        for size in sizes:
            start_time = time.time()
            
            for i in range(size):
                asyncio.run(cache.set(
                    f"scale_key_{i}",
                    f"scale_value_{i}",
                    cache_type="entity",
                    tiers=[CacheTier.MEMORY]
                ))
            
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        # Calculate scaling factor
        scaling_factors = []
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i-1]
            time_ratio = times[i] / times[i-1]
            scaling_factors.append(time_ratio / size_ratio)
        
        avg_scaling = np.mean(scaling_factors)
        
        print(f"Scaling factors: {scaling_factors}")
        print(f"Average scaling factor: {avg_scaling:.2f}")
        
        # Should scale roughly linearly (factor close to 1)
        self.assertLess(avg_scaling, 1.5)
        self.assertGreater(avg_scaling, 0.8)
    
    def test_concurrent_access_performance(self):
        """Test performance under concurrent access"""
        cache = CacheStrategy()
        num_threads = 10
        operations_per_thread = 100
        
        # Pre-populate cache
        for i in range(100):
            asyncio.run(cache.set(
                f"concurrent_key_{i}",
                f"concurrent_value_{i}",
                cache_type="entity",
                tiers=[CacheTier.MEMORY]
            ))
        
        async def concurrent_operations(thread_id):
            """Perform concurrent cache operations"""
            for i in range(operations_per_thread):
                # Mix of reads and writes
                if random.random() < 0.8:  # 80% reads
                    await cache.get(
                        f"concurrent_key_{random.randint(0, 99)}",
                        cache_type="entity"
                    )
                else:  # 20% writes
                    await cache.set(
                        f"concurrent_key_{random.randint(0, 99)}",
                        f"updated_value_{thread_id}_{i}",
                        cache_type="entity",
                        tiers=[CacheTier.MEMORY]
                    )
        
        # Run concurrent operations
        start_time = time.time()
        
        async def run_all():
            tasks = [
                concurrent_operations(i)
                for i in range(num_threads)
            ]
            await asyncio.gather(*tasks)
        
        asyncio.run(run_all())
        
        elapsed = time.time() - start_time
        total_operations = num_threads * operations_per_thread
        ops_per_second = total_operations / elapsed
        
        print(f"Concurrent access: {ops_per_second:.0f} ops/sec with {num_threads} threads")
        
        # Should handle high concurrent load
        self.assertGreater(ops_per_second, 1000)
    
    def test_bottleneck_detection(self):
        """Test bottleneck detection performance"""
        monitor = PerformanceMonitor()
        
        # Simulate various performance scenarios
        scenarios = [
            ('normal', 0.01, 0.5, 30),  # Normal operation
            ('slow_query', 5.0, 0.5, 30),  # Slow queries
            ('low_cache', 0.01, 0.2, 30),  # Low cache hit rate
            ('high_cpu', 0.01, 0.5, 85),  # High CPU usage
        ]
        
        for scenario_name, query_time, cache_hit_rate, cpu_usage in scenarios:
            # Record metrics
            for _ in range(10):
                monitor.record_query_time(f"query_{scenario_name}", query_time)
                monitor.record_cache_hit(random.random() < cache_hit_rate, "memory")
                
                # Simulate CPU usage
                monitor.collector.record(MetricType.CPU_USAGE, cpu_usage)
            
            # Check if bottlenecks are detected
            bottlenecks = monitor.get_bottlenecks()
            
            if scenario_name != 'normal':
                self.assertGreater(len(bottlenecks), 0, 
                                 f"Should detect bottleneck in {scenario_name}")
                
                print(f"Detected bottlenecks in {scenario_name}:")
                for bottleneck in bottlenecks:
                    print(f"  - {bottleneck.component}: {bottleneck.description}")


class TestPerformanceStress(unittest.TestCase):
    """Stress tests for performance limits"""
    
    def test_cache_stress(self):
        """Stress test cache with high load"""
        cache = CacheStrategy()
        num_operations = 10000
        
        start_time = time.time()
        
        # Rapid fire cache operations
        for i in range(num_operations):
            key = f"stress_key_{i % 1000}"  # Reuse keys for contention
            
            if i % 3 == 0:
                asyncio.run(cache.set(
                    key,
                    f"stress_value_{i}",
                    cache_type="entity",
                    tiers=[CacheTier.MEMORY]
                ))
            else:
                asyncio.run(cache.get(
                    key,
                    cache_type="entity"
                ))
        
        elapsed = time.time() - start_time
        ops_per_second = num_operations / elapsed
        
        print(f"Cache stress test: {ops_per_second:.0f} ops/sec")
        
        # Should handle at least 5000 ops/sec under stress
        self.assertGreater(ops_per_second, 5000)
        
        # Check cache statistics
        stats = cache.get_statistics()
        print(f"Cache statistics after stress:")
        print(f"  Memory hit rate: {stats['memory']['hit_rate']:.2%}")
        print(f"  Evictions: {stats['memory']['evictions']}")
    
    def test_parallel_processor_stress(self):
        """Stress test parallel processor"""
        processor = ParallelProcessor(max_workers=8)
        
        try:
            # Create heavy workload
            def heavy_task(n):
                """Simulate heavy computation"""
                result = 0
                for i in range(n * 10000):
                    result += i ** 2 % 1000
                return result
            
            # Large batch of tasks
            tasks = list(range(1, 101))  # 100 tasks
            
            start_time = time.time()
            
            results = asyncio.run(
                processor.process(
                    heavy_task,
                    tasks,
                    priority=TaskPriority.HIGH,
                    batch_size=10
                )
            )
            
            elapsed = time.time() - start_time
            
            print(f"Parallel stress test: {len(tasks)} tasks in {elapsed:.2f}s")
            print(f"Average: {elapsed/len(tasks):.3f}s per task")
            
            # Check statistics
            stats = processor.get_statistics()
            print(f"Processor statistics:")
            print(f"  Completed: {stats['tasks']['completed']}")
            print(f"  Failed: {stats['tasks']['failed']}")
            print(f"  Queue peak: {stats['queue']['peak_size']}")
            
            # Should complete all tasks
            self.assertEqual(len(results), len(tasks))
            
        finally:
            processor.shutdown()


if __name__ == '__main__':
    unittest.main()