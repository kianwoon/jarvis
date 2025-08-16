#!/usr/bin/env python3
"""
Universal Radiating Coverage System - Health Check Script

This script performs comprehensive health checks on all radiating system components:
- Service connectivity
- Configuration validation
- Performance metrics
- Cache status
- Database integrity
"""

import sys
import os
import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import httpx
import redis
from neo4j import GraphDatabase
from sqlalchemy import create_engine, text
from pymilvus import connections, utility

from app.core.config import get_settings
from app.core.redis_client import get_redis_client
from app.core.radiating_settings_cache import get_radiating_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RadiatingHealthCheck:
    """Comprehensive health check for the Radiating Coverage System"""
    
    def __init__(self):
        self.settings = get_settings()
        self.health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'components': {},
            'metrics': {},
            'warnings': [],
            'errors': []
        }
        
    async def check_postgresql(self) -> Tuple[bool, Dict]:
        """Check PostgreSQL database health"""
        component_status = {
            'status': 'DOWN',
            'details': {},
            'metrics': {}
        }
        
        try:
            database_url = (
                f"postgresql://{self.settings.POSTGRES_USER}:{self.settings.POSTGRES_PASSWORD}"
                f"@{self.settings.POSTGRES_HOST}:{self.settings.POSTGRES_PORT}/{self.settings.POSTGRES_DB}"
            )
            
            engine = create_engine(database_url, pool_pre_ping=True)
            
            with engine.connect() as conn:
                # Check connectivity
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                component_status['details']['version'] = version
                
                # Check radiating settings
                settings_result = conn.execute(
                    text("SELECT COUNT(*) FROM settings WHERE category = 'radiating'")
                )
                settings_count = settings_result.fetchone()[0]
                component_status['details']['radiating_settings'] = settings_count > 0
                
                # Get database size
                size_result = conn.execute(
                    text("SELECT pg_database_size(current_database())")
                )
                db_size = size_result.fetchone()[0]
                component_status['metrics']['database_size_mb'] = db_size / (1024 * 1024)
                
                # Check connection pool
                pool_result = conn.execute(
                    text("""
                    SELECT count(*) as total,
                           count(*) FILTER (WHERE state = 'active') as active,
                           count(*) FILTER (WHERE state = 'idle') as idle
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                    """)
                )
                pool_stats = pool_result.fetchone()
                component_status['metrics']['connections'] = {
                    'total': pool_stats[0],
                    'active': pool_stats[1],
                    'idle': pool_stats[2]
                }
                
                component_status['status'] = 'UP'
                
                # Warnings
                if pool_stats[0] > 50:
                    self.health_status['warnings'].append(
                        f"High PostgreSQL connection count: {pool_stats[0]}"
                    )
                
            engine.dispose()
            return True, component_status
            
        except Exception as e:
            component_status['details']['error'] = str(e)
            self.health_status['errors'].append(f"PostgreSQL: {str(e)}")
            return False, component_status
    
    async def check_neo4j(self) -> Tuple[bool, Dict]:
        """Check Neo4j graph database health"""
        component_status = {
            'status': 'DOWN',
            'details': {},
            'metrics': {}
        }
        
        try:
            uri = f"bolt://{self.settings.NEO4J_HOST}:{self.settings.NEO4J_BOLT_PORT}"
            driver = GraphDatabase.driver(
                uri, 
                auth=(self.settings.NEO4J_USER, self.settings.NEO4J_PASSWORD)
            )
            
            with driver.session() as session:
                # Check connectivity
                result = session.run("CALL dbms.components() YIELD name, versions")
                components = result.data()
                component_status['details']['components'] = components[0] if components else {}
                
                # Get database statistics
                stats_result = session.run("""
                    MATCH (n)
                    OPTIONAL MATCH ()-[r]->()
                    WITH count(DISTINCT n) as nodes, count(r) as relationships
                    RETURN nodes, relationships,
                           CASE WHEN nodes > 0 THEN toFloat(relationships) / nodes ELSE 0 END as avg_degree
                """)
                stats = stats_result.single()
                
                component_status['metrics']['nodes'] = stats['nodes']
                component_status['metrics']['relationships'] = stats['relationships']
                component_status['metrics']['avg_degree'] = round(stats['avg_degree'], 2)
                
                # Check radiating entities
                radiating_result = session.run("""
                    MATCH (n:RadiatingEntity)
                    RETURN count(n) as radiating_count
                """)
                radiating_stats = radiating_result.single()
                component_status['metrics']['radiating_entities'] = radiating_stats['radiating_count']
                
                # Check indexes
                index_result = session.run("SHOW INDEXES")
                indexes = index_result.data()
                component_status['details']['index_count'] = len(indexes)
                
                # Memory usage
                memory_result = session.run("""
                    CALL dbms.queryJmx('org.neo4j:instance=kernel#0,name=Page cache')
                    YIELD attributes
                    RETURN attributes.UsageRatio.value as cache_usage
                """)
                memory = memory_result.single()
                if memory:
                    component_status['metrics']['cache_usage'] = round(memory['cache_usage'], 2)
                
                component_status['status'] = 'UP'
                
                # Warnings
                if stats['nodes'] > 1000000:
                    self.health_status['warnings'].append(
                        f"Large graph size: {stats['nodes']} nodes"
                    )
                
                if component_status['metrics'].get('avg_degree', 0) < 2:
                    self.health_status['warnings'].append(
                        "Low graph connectivity (avg degree < 2)"
                    )
                
            driver.close()
            return True, component_status
            
        except Exception as e:
            component_status['details']['error'] = str(e)
            self.health_status['errors'].append(f"Neo4j: {str(e)}")
            return False, component_status
    
    async def check_redis(self) -> Tuple[bool, Dict]:
        """Check Redis cache health"""
        component_status = {
            'status': 'DOWN',
            'details': {},
            'metrics': {}
        }
        
        try:
            client = redis.Redis(
                host=self.settings.REDIS_HOST,
                port=self.settings.REDIS_PORT,
                password=self.settings.REDIS_PASSWORD,
                decode_responses=True
            )
            
            # Check connectivity
            if not client.ping():
                raise Exception("Redis ping failed")
            
            # Get Redis info
            info = client.info()
            component_status['details']['version'] = info.get('redis_version', 'unknown')
            component_status['details']['uptime_days'] = info.get('uptime_in_days', 0)
            
            # Memory metrics
            memory_info = client.info('memory')
            used_memory_mb = memory_info.get('used_memory', 0) / (1024 * 1024)
            max_memory = memory_info.get('maxmemory', 0)
            max_memory_mb = max_memory / (1024 * 1024) if max_memory > 0 else 0
            
            component_status['metrics']['memory_used_mb'] = round(used_memory_mb, 2)
            component_status['metrics']['memory_max_mb'] = round(max_memory_mb, 2)
            
            if max_memory > 0:
                memory_usage = (memory_info.get('used_memory', 0) / max_memory) * 100
                component_status['metrics']['memory_usage_percent'] = round(memory_usage, 2)
                
                if memory_usage > 80:
                    self.health_status['warnings'].append(
                        f"High Redis memory usage: {memory_usage:.1f}%"
                    )
            
            # Check radiating cache keys
            radiating_keys = client.keys('radiating:*')
            component_status['metrics']['radiating_keys'] = len(radiating_keys)
            
            # Cache hit ratio
            stats = client.info('stats')
            hits = stats.get('keyspace_hits', 0)
            misses = stats.get('keyspace_misses', 0)
            total_ops = hits + misses
            
            if total_ops > 0:
                hit_ratio = (hits / total_ops) * 100
                component_status['metrics']['cache_hit_ratio'] = round(hit_ratio, 2)
                
                if hit_ratio < 50:
                    self.health_status['warnings'].append(
                        f"Low cache hit ratio: {hit_ratio:.1f}%"
                    )
            
            # Check specific radiating namespaces
            namespaces = ['entities', 'paths', 'queries', 'results', 'metrics']
            namespace_stats = {}
            
            for ns in namespaces:
                pattern = f"radiating:{ns}:*"
                count = len(client.keys(pattern))
                namespace_stats[ns] = count
            
            component_status['details']['namespace_stats'] = namespace_stats
            
            component_status['status'] = 'UP'
            client.close()
            return True, component_status
            
        except Exception as e:
            component_status['details']['error'] = str(e)
            self.health_status['errors'].append(f"Redis: {str(e)}")
            return False, component_status
    
    async def check_milvus(self) -> Tuple[bool, Dict]:
        """Check Milvus vector database health"""
        component_status = {
            'status': 'DOWN',
            'details': {},
            'metrics': {}
        }
        
        try:
            # Connect to Milvus
            connections.connect(
                alias="health_check",
                host=self.settings.MILVUS_HOST,
                port=self.settings.MILVUS_PORT
            )
            
            # Check version
            version = utility.get_server_version(using="health_check")
            component_status['details']['version'] = version
            
            # List collections
            collections = utility.list_collections(using="health_check")
            component_status['details']['collection_count'] = len(collections)
            
            # Check primary collection
            primary_collection = self.settings.MILVUS_COLLECTION_NAME
            if primary_collection in collections:
                from pymilvus import Collection
                
                collection = Collection(primary_collection, using="health_check")
                collection.load()
                
                # Get collection stats
                stats = collection.num_entities
                component_status['metrics']['vectors'] = stats
                
                # Check index
                indexes = collection.indexes
                component_status['details']['has_index'] = len(indexes) > 0
                
                collection.release()
            
            component_status['status'] = 'UP'
            connections.disconnect("health_check")
            return True, component_status
            
        except Exception as e:
            component_status['details']['error'] = str(e)
            self.health_status['errors'].append(f"Milvus: {str(e)}")
            try:
                connections.disconnect("health_check")
            except:
                pass
            return False, component_status
    
    async def check_api_endpoints(self) -> Tuple[bool, Dict]:
        """Check API endpoints health"""
        component_status = {
            'status': 'DOWN',
            'details': {},
            'metrics': {}
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Main health endpoint
                health_response = await client.get("http://localhost:8000/health")
                component_status['details']['main_health'] = health_response.status_code == 200
                
                # Radiating specific endpoints
                endpoints = [
                    ("/api/v1/radiating/health", "radiating_health"),
                    ("/api/v1/radiating/metrics", "radiating_metrics"),
                    ("/api/v1/settings/radiating", "radiating_settings")
                ]
                
                endpoint_status = {}
                response_times = []
                
                for endpoint, name in endpoints:
                    try:
                        start_time = time.time()
                        response = await client.get(f"http://localhost:8000{endpoint}")
                        elapsed = (time.time() - start_time) * 1000  # Convert to ms
                        
                        endpoint_status[name] = {
                            'status_code': response.status_code,
                            'response_time_ms': round(elapsed, 2),
                            'healthy': response.status_code == 200
                        }
                        
                        response_times.append(elapsed)
                        
                    except Exception as e:
                        endpoint_status[name] = {
                            'status_code': 0,
                            'error': str(e),
                            'healthy': False
                        }
                
                component_status['details']['endpoints'] = endpoint_status
                
                # Calculate metrics
                if response_times:
                    component_status['metrics']['avg_response_time_ms'] = round(
                        sum(response_times) / len(response_times), 2
                    )
                    component_status['metrics']['max_response_time_ms'] = round(
                        max(response_times), 2
                    )
                    
                    if component_status['metrics']['avg_response_time_ms'] > 1000:
                        self.health_status['warnings'].append(
                            f"Slow API response times: {component_status['metrics']['avg_response_time_ms']}ms avg"
                        )
                
                # Overall API status
                healthy_endpoints = sum(
                    1 for ep in endpoint_status.values() if ep.get('healthy', False)
                )
                
                if healthy_endpoints == len(endpoints):
                    component_status['status'] = 'UP'
                elif healthy_endpoints > 0:
                    component_status['status'] = 'DEGRADED'
                    
            return component_status['status'] != 'DOWN', component_status
            
        except Exception as e:
            component_status['details']['error'] = str(e)
            self.health_status['errors'].append(f"API: {str(e)}")
            return False, component_status
    
    async def check_radiating_settings(self) -> Tuple[bool, Dict]:
        """Validate radiating system configuration"""
        component_status = {
            'status': 'DOWN',
            'details': {},
            'validation': {}
        }
        
        try:
            settings = get_radiating_settings()
            
            if not settings:
                raise Exception("Radiating settings not found")
            
            # Check required settings
            required_keys = [
                'enabled', 'default_depth', 'max_depth',
                'relevance_threshold', 'cache_ttl'
            ]
            
            missing_keys = []
            for key in required_keys:
                if key not in settings:
                    missing_keys.append(key)
            
            component_status['validation']['missing_keys'] = missing_keys
            component_status['validation']['has_all_required'] = len(missing_keys) == 0
            
            # Validate setting values
            validations = {
                'depth_valid': 0 < settings.get('default_depth', 0) <= settings.get('max_depth', 5),
                'threshold_valid': 0 < settings.get('relevance_threshold', 0) <= 1,
                'cache_ttl_valid': settings.get('cache_ttl', 0) > 0,
                'batch_size_valid': 0 < settings.get('performance', {}).get('batch_size', 0) <= 100
            }
            
            component_status['validation'].update(validations)
            
            # Check sub-configurations
            sub_configs = ['query_expansion', 'extraction', 'traversal', 'performance']
            for config in sub_configs:
                component_status['details'][f'has_{config}'] = config in settings
            
            # Performance warnings
            if settings.get('max_depth', 0) > 5:
                self.health_status['warnings'].append(
                    f"High max_depth setting: {settings['max_depth']} (may impact performance)"
                )
            
            if settings.get('performance', {}).get('max_concurrent_queries', 0) > 10:
                self.health_status['warnings'].append(
                    "High concurrent query limit may cause resource exhaustion"
                )
            
            # Overall status
            if all(validations.values()) and len(missing_keys) == 0:
                component_status['status'] = 'UP'
            elif len(missing_keys) == 0:
                component_status['status'] = 'DEGRADED'
                
            return component_status['status'] != 'DOWN', component_status
            
        except Exception as e:
            component_status['details']['error'] = str(e)
            self.health_status['errors'].append(f"Settings: {str(e)}")
            return False, component_status
    
    async def check_performance_metrics(self) -> Dict:
        """Collect and analyze performance metrics"""
        metrics = {}
        
        try:
            # Get Redis performance metrics
            redis_client = get_redis_client()
            
            # Query latency (if tracked)
            latency_key = "radiating:metrics:query_latency"
            if redis_client.exists(latency_key):
                latencies = redis_client.lrange(latency_key, -100, -1)
                if latencies:
                    latency_values = [float(l) for l in latencies]
                    metrics['query_latency'] = {
                        'avg_ms': round(sum(latency_values) / len(latency_values), 2),
                        'min_ms': round(min(latency_values), 2),
                        'max_ms': round(max(latency_values), 2),
                        'samples': len(latency_values)
                    }
            
            # Throughput (if tracked)
            throughput_key = "radiating:metrics:throughput"
            if redis_client.exists(throughput_key):
                throughput = redis_client.get(throughput_key)
                metrics['queries_per_minute'] = float(throughput)
            
            # Cache efficiency
            cache_hits = redis_client.get("radiating:metrics:cache_hits")
            cache_misses = redis_client.get("radiating:metrics:cache_misses")
            
            if cache_hits and cache_misses:
                total = int(cache_hits) + int(cache_misses)
                if total > 0:
                    metrics['cache_hit_ratio'] = round((int(cache_hits) / total) * 100, 2)
            
        except Exception as e:
            logger.warning(f"Could not collect performance metrics: {e}")
        
        return metrics
    
    async def run_health_check(self) -> Dict:
        """Run complete health check"""
        logger.info("Starting Radiating System Health Check...")
        
        # Check all components
        checks = [
            ("PostgreSQL", self.check_postgresql()),
            ("Neo4j", self.check_neo4j()),
            ("Redis", self.check_redis()),
            ("Milvus", self.check_milvus()),
            ("API", self.check_api_endpoints()),
            ("Settings", self.check_radiating_settings())
        ]
        
        all_healthy = True
        
        for name, check_coro in checks:
            try:
                is_healthy, status = await check_coro
                self.health_status['components'][name] = status
                
                if not is_healthy:
                    all_healthy = False
                    
            except Exception as e:
                logger.error(f"Failed to check {name}: {e}")
                self.health_status['components'][name] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                all_healthy = False
        
        # Collect performance metrics
        self.health_status['metrics'] = await self.check_performance_metrics()
        
        # Determine overall status
        component_statuses = [
            c.get('status', 'DOWN') 
            for c in self.health_status['components'].values()
        ]
        
        if all(s == 'UP' for s in component_statuses):
            self.health_status['overall_status'] = 'HEALTHY'
        elif any(s == 'UP' for s in component_statuses):
            self.health_status['overall_status'] = 'DEGRADED'
        else:
            self.health_status['overall_status'] = 'UNHEALTHY'
        
        return self.health_status
    
    def print_report(self):
        """Print formatted health report"""
        status = self.health_status
        
        # Header
        print("\n" + "="*60)
        print("RADIATING SYSTEM HEALTH CHECK REPORT")
        print("="*60)
        print(f"Timestamp: {status['timestamp']}")
        print(f"Overall Status: {self._format_status(status['overall_status'])}")
        print("-"*60)
        
        # Component Status
        print("\nCOMPONENT STATUS:")
        for name, component in status['components'].items():
            status_icon = self._get_status_icon(component['status'])
            print(f"  {status_icon} {name}: {component['status']}")
            
            if component['status'] != 'UP' and 'error' in component.get('details', {}):
                print(f"     Error: {component['details']['error']}")
        
        # Metrics
        if status['metrics']:
            print("\nPERFORMANCE METRICS:")
            for key, value in status['metrics'].items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    - {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")
        
        # Warnings
        if status['warnings']:
            print("\n⚠️ WARNINGS:")
            for warning in status['warnings']:
                print(f"  - {warning}")
        
        # Errors
        if status['errors']:
            print("\n❌ ERRORS:")
            for error in status['errors']:
                print(f"  - {error}")
        
        # Summary
        print("\n" + "-"*60)
        healthy = sum(1 for c in status['components'].values() if c['status'] == 'UP')
        total = len(status['components'])
        print(f"Summary: {healthy}/{total} components healthy")
        
        if status['overall_status'] == 'HEALTHY':
            print("✅ System is fully operational")
        elif status['overall_status'] == 'DEGRADED':
            print("⚠️ System is operational with degraded performance")
        else:
            print("❌ System has critical issues requiring attention")
        
        print("="*60 + "\n")
    
    def _format_status(self, status: str) -> str:
        """Format status with color/icon"""
        icons = {
            'HEALTHY': '✅',
            'DEGRADED': '⚠️',
            'UNHEALTHY': '❌',
            'UNKNOWN': '❓'
        }
        return f"{icons.get(status, '❓')} {status}"
    
    def _get_status_icon(self, status: str) -> str:
        """Get icon for component status"""
        icons = {
            'UP': '✅',
            'DOWN': '❌',
            'DEGRADED': '⚠️',
            'ERROR': '💥',
            'UNKNOWN': '❓'
        }
        return icons.get(status, '❓')
    
    def export_report(self, filepath: str = None):
        """Export health report to JSON file"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"radiating_health_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.health_status, f, indent=2, default=str)
        
        logger.info(f"Health report exported to: {filepath}")
        return filepath

async def main():
    """Main entry point for health check"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Universal Radiating Coverage System - Health Check'
    )
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export report to JSON file'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output'
    )
    parser.add_argument(
        '--watch',
        type=int,
        metavar='SECONDS',
        help='Run health check continuously every N seconds'
    )
    
    args = parser.parse_args()
    
    if args.watch:
        # Continuous monitoring mode
        logger.info(f"Starting continuous health monitoring (interval: {args.watch}s)")
        
        while True:
            try:
                checker = RadiatingHealthCheck()
                status = await checker.run_health_check()
                
                if not args.quiet:
                    checker.print_report()
                
                if status['overall_status'] != 'HEALTHY':
                    logger.warning(f"System status: {status['overall_status']}")
                
                await asyncio.sleep(args.watch)
                
            except KeyboardInterrupt:
                logger.info("Health monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(args.watch)
    else:
        # Single run mode
        checker = RadiatingHealthCheck()
        status = await checker.run_health_check()
        
        if not args.quiet:
            checker.print_report()
        
        if args.export:
            filepath = checker.export_report()
            print(f"Report exported to: {filepath}")
        
        # Exit code based on health
        if status['overall_status'] == 'HEALTHY':
            sys.exit(0)
        elif status['overall_status'] == 'DEGRADED':
            sys.exit(1)
        else:
            sys.exit(2)

if __name__ == "__main__":
    asyncio.run(main())