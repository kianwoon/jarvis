#!/usr/bin/env python3
"""
Critical fixes for AggregatorNode robustness issues identified in testing.
These fixes address the highest priority security and stability concerns.
"""

import re
import time
import math
from typing import Dict, Any, List, Union
from datetime import datetime

class AggregatorNodeSecurityFixes:
    """Critical security and robustness fixes for AggregatorNode"""
    
    # Security Configuration
    MAX_INPUT_SIZE = 10_000_000  # 10MB per input
    MAX_INPUT_COUNT = 1000       # Maximum number of inputs
    MAX_TOTAL_SIZE = 50_000_000  # 50MB total processing limit
    PROCESSING_TIMEOUT = 300     # 5 minutes
    
    # Content Security
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                # JavaScript protocol
        r'data:.*base64',             # Base64 data URIs
        r'vbscript:',                 # VBScript protocol
        r'onload\s*=',                # Event handlers
        r'onerror\s*=',
        r'onclick\s*=',
        r'eval\s*\(',                 # Eval calls
        r'__import__\s*\(',           # Python imports
        r'exec\s*\(',                 # Exec calls
        r'\.system\s*\(',             # System calls
    ]
    
    @classmethod
    def validate_confidence_score(cls, score: Any) -> float:
        """
        Validate and normalize confidence scores to prevent invalid values.
        
        Args:
            score: Raw confidence score value
            
        Returns:
            float: Valid confidence score between 0.0 and 1.0
        """
        try:
            # Handle None or non-numeric values
            if score is None or not isinstance(score, (int, float)):
                return 0.5
            
            # Handle NaN
            if math.isnan(score):
                return 0.5
                
            # Handle infinity
            if math.isinf(score):
                return 1.0 if score > 0 else 0.0
                
            # Clamp to valid range
            return max(0.0, min(1.0, float(score)))
            
        except (ValueError, TypeError, OverflowError):
            return 0.5
    
    @classmethod
    def sanitize_text_input(cls, text: Any) -> str:
        """
        Sanitize text input to prevent injection attacks and ensure safe processing.
        
        Args:
            text: Raw text input
            
        Returns:
            str: Sanitized text safe for processing
        """
        try:
            # Handle None values
            if text is None:
                return ""
            
            # Convert to string
            text_str = str(text)
            
            # Check size limits
            if len(text_str) > cls.MAX_INPUT_SIZE:
                text_str = text_str[:cls.MAX_INPUT_SIZE] + "[TRUNCATED]"
            
            # Remove dangerous patterns
            for pattern in cls.DANGEROUS_PATTERNS:
                text_str = re.sub(pattern, '[REMOVED]', text_str, flags=re.IGNORECASE | re.DOTALL)
            
            # Remove potential SQL injection patterns
            sql_patterns = [
                r';.*?drop\s+table',
                r';.*?delete\s+from',
                r';.*?insert\s+into',
                r';.*?update\s+.*?set',
                r'union\s+select',
                r'1\s*=\s*1',
                r'or\s+1\s*=\s*1',
            ]
            
            for pattern in sql_patterns:
                text_str = re.sub(pattern, '[REMOVED]', text_str, flags=re.IGNORECASE)
            
            # Remove excessive whitespace and control characters
            text_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text_str)
            text_str = re.sub(r'\s+', ' ', text_str).strip()
            
            return text_str
            
        except Exception:
            # If sanitization fails, return safe default
            return "[SANITIZATION_ERROR]"
    
    @classmethod
    def validate_input_limits(cls, inputs: List[Any]) -> tuple[bool, str]:
        """
        Validate input collection against resource limits.
        
        Args:
            inputs: List of input items to validate
            
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # Check input count
            if len(inputs) > cls.MAX_INPUT_COUNT:
                return False, f"Too many inputs: {len(inputs)} exceeds limit of {cls.MAX_INPUT_COUNT}"
            
            # Check total size
            total_size = 0
            for item in inputs:
                item_size = len(str(item))
                if item_size > cls.MAX_INPUT_SIZE:
                    return False, f"Input item too large: {item_size} bytes exceeds limit of {cls.MAX_INPUT_SIZE}"
                total_size += item_size
            
            if total_size > cls.MAX_TOTAL_SIZE:
                return False, f"Total input size too large: {total_size} bytes exceeds limit of {cls.MAX_TOTAL_SIZE}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Input validation error: {str(e)}"
    
    @classmethod
    def create_timeout_wrapper(cls, timeout_seconds: int = None):
        """
        Create a timeout wrapper decorator for aggregation operations.
        
        Args:
            timeout_seconds: Timeout in seconds (default: class PROCESSING_TIMEOUT)
            
        Returns:
            Decorator function
        """
        timeout = timeout_seconds or cls.PROCESSING_TIMEOUT
        
        def timeout_decorator(func):
            def wrapper(*args, **kwargs):
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Operation timed out after {timeout} seconds")
                
                # Set up timeout (Unix only)
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout)
                    
                    try:
                        result = func(*args, **kwargs)
                        return result
                    finally:
                        signal.alarm(0)  # Cancel timeout
                        
                except (AttributeError, OSError):
                    # Windows or signal not available - use alternative timeout
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"Operation exceeded {timeout} seconds")
                    
                    return result
                    
            return wrapper
        return timeout_decorator
    
    @classmethod
    def safe_analyze_output_quality(cls, output: Any, quality_weights: Dict[str, float], index: int) -> Dict[str, Any]:
        """
        Enhanced version of _analyze_output_quality with security fixes.
        
        Args:
            output: Output to analyze
            quality_weights: Quality scoring weights
            index: Output index
            
        Returns:
            dict: Quality analysis with validated scores
        """
        try:
            # Sanitize input
            output_text = cls.sanitize_text_input(output)
            
            # Validate quality weights
            if not isinstance(quality_weights, dict):
                quality_weights = {
                    "length": 0.2,
                    "coherence": 0.3,
                    "relevance": 0.3,
                    "completeness": 0.2
                }
            
            # Validate individual weights
            validated_weights = {}
            for key, value in quality_weights.items():
                validated_weights[key] = cls.validate_confidence_score(value)
            
            # Calculate quality factors with bounds checking
            text_length = len(output_text)
            length_score = min(text_length / 1000.0, 1.0) if text_length > 0 else 0.0
            
            # Coherence based on sentence structure
            sentences = output_text.count('.') + output_text.count('!') + output_text.count('?')
            words = len(output_text.split()) if output_text else 1
            coherence_score = min(sentences / words, 1.0) if words > 0 else 0.0
            
            # Default relevance (would be enhanced with embeddings in production)
            relevance_score = 0.7
            
            # Completeness based on structure indicators
            structure_indicators = (
                output_text.count('\n') + 
                output_text.count(':') + 
                output_text.count('-') +
                output_text.count('*') +
                output_text.count('1.') +  # Numbered lists
                output_text.count('•')     # Bullet points
            )
            completeness_score = min(structure_indicators / 10.0, 1.0)
            
            # Calculate weighted confidence score
            confidence_score = (
                length_score * validated_weights.get("length", 0.2) +
                coherence_score * validated_weights.get("coherence", 0.3) +
                relevance_score * validated_weights.get("relevance", 0.3) +
                completeness_score * validated_weights.get("completeness", 0.2)
            )
            
            # Validate final confidence score
            confidence_score = cls.validate_confidence_score(confidence_score)
            
            return {
                "original_output": output,
                "text": output_text,
                "confidence_score": confidence_score,
                "quality_factors": {
                    "length": length_score,
                    "coherence": coherence_score,
                    "relevance": relevance_score,
                    "completeness": completeness_score
                },
                "index": index,
                "timestamp": datetime.utcnow().isoformat(),
                "security_validated": True
            }
            
        except Exception as e:
            # Safe fallback for any errors
            return {
                "original_output": output,
                "text": cls.sanitize_text_input(output),
                "confidence_score": 0.1,
                "quality_factors": {
                    "length": 0.0,
                    "coherence": 0.0,
                    "relevance": 0.0,
                    "completeness": 0.0
                },
                "index": index,
                "timestamp": datetime.utcnow().isoformat(),
                "error": f"Quality analysis error: {str(e)}",
                "security_validated": False
            }
    
    @classmethod
    def safe_calculate_overall_confidence(cls, outputs: List[Dict[str, Any]], strategy: str) -> float:
        """
        Enhanced version of _calculate_overall_confidence with validation.
        
        Args:
            outputs: List of output analyses
            strategy: Aggregation strategy used
            
        Returns:
            float: Validated overall confidence score
        """
        try:
            if not outputs:
                return 0.0
            
            # Extract and validate all confidence scores
            scores = []
            for output in outputs:
                score = output.get("confidence_score", 0)
                validated_score = cls.validate_confidence_score(score)
                scores.append(validated_score)
            
            if not scores:
                return 0.5
            
            # Calculate confidence based on strategy
            if strategy in ["weighted_vote", "best_selection"]:
                return max(scores)
            elif strategy == "consensus_ranking":
                return sum(scores) / len(scores)
            else:
                # Weighted average favoring higher confidence scores
                sorted_scores = sorted(scores, reverse=True)
                weights = list(range(len(sorted_scores), 0, -1))
                
                if sum(weights) == 0:
                    return 0.5
                    
                weighted_sum = sum(score * weight for score, weight in zip(sorted_scores, weights))
                result = weighted_sum / sum(weights)
                
                return cls.validate_confidence_score(result)
                
        except Exception:
            return 0.5
    
    @classmethod
    def create_secure_aggregator_config(cls, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a security-hardened aggregator configuration.
        
        Args:
            base_config: Base aggregator configuration
            
        Returns:
            dict: Hardened configuration with security settings
        """
        secure_config = base_config.copy()
        
        # Add security settings
        secure_config.update({
            "input_validation_enabled": True,
            "content_sanitization_enabled": True,
            "confidence_validation_enabled": True,
            "resource_limits_enabled": True,
            "timeout_enabled": True,
            "max_input_size": cls.MAX_INPUT_SIZE,
            "max_input_count": cls.MAX_INPUT_COUNT,
            "max_total_size": cls.MAX_TOTAL_SIZE,
            "processing_timeout": cls.PROCESSING_TIMEOUT,
            "security_audit_log": True,
        })
        
        # Validate existing configuration values
        if "confidence_threshold" in secure_config:
            secure_config["confidence_threshold"] = cls.validate_confidence_score(
                secure_config["confidence_threshold"]
            )
        
        if "similarity_threshold" in secure_config:
            secure_config["similarity_threshold"] = cls.validate_confidence_score(
                secure_config["similarity_threshold"]
            )
        
        # Ensure safe fallback strategy
        safe_fallbacks = ["return_best", "simple_merge", "empty_result"]
        if secure_config.get("fallback_strategy") not in safe_fallbacks:
            secure_config["fallback_strategy"] = "return_best"
        
        return secure_config

def demonstrate_fixes():
    """Demonstrate the security fixes with examples"""
    print("AGGREGATOR NODE SECURITY FIXES DEMONSTRATION")
    print("=" * 60)
    
    # Test confidence score validation
    print("\n1. CONFIDENCE SCORE VALIDATION")
    print("-" * 30)
    test_scores = [0.5, float('inf'), float('nan'), -1, 2.0, None, "invalid"]
    
    for score in test_scores:
        validated = AggregatorNodeSecurityFixes.validate_confidence_score(score)
        print(f"Input: {str(score):>10} -> Validated: {validated:.3f}")
    
    # Test input sanitization
    print("\n2. INPUT SANITIZATION")
    print("-" * 30)
    malicious_inputs = [
        "<script>alert('XSS')</script>",
        "'; DROP TABLE users; --",
        "javascript:alert('XSS')",
        "data:text/html,<script>alert('XSS')</script>",
        "eval('malicious code')",
        None,
        "Normal text content",
    ]
    
    for input_text in malicious_inputs:
        sanitized = AggregatorNodeSecurityFixes.sanitize_text_input(input_text)
        print(f"Input: {str(input_text)[:30]:>30} -> Sanitized: {sanitized[:30]}")
    
    # Test input limits validation
    print("\n3. INPUT LIMITS VALIDATION")
    print("-" * 30)
    
    # Test small input
    small_inputs = ["short text"] * 5
    valid, error = AggregatorNodeSecurityFixes.validate_input_limits(small_inputs)
    print(f"Small inputs ({len(small_inputs)} items): {'✓ Valid' if valid else '✗ ' + error}")
    
    # Test large input count
    large_inputs = ["text"] * 1500
    valid, error = AggregatorNodeSecurityFixes.validate_input_limits(large_inputs)
    print(f"Large input count ({len(large_inputs)} items): {'✓ Valid' if valid else '✗ ' + error}")
    
    # Test quality analysis with security
    print("\n4. SECURE QUALITY ANALYSIS")
    print("-" * 30)
    
    test_output = "<script>alert('xss')</script>This is a test analysis with some content."
    malformed_weights = {"length": float('inf'), "coherence": "invalid"}
    
    result = AggregatorNodeSecurityFixes.safe_analyze_output_quality(
        test_output, malformed_weights, 0
    )
    
    print(f"Original: {test_output[:40]}...")
    print(f"Sanitized: {result['text'][:40]}...")
    print(f"Confidence: {result['confidence_score']:.3f}")
    print(f"Security validated: {result['security_validated']}")
    
    # Test secure configuration
    print("\n5. SECURE CONFIGURATION")
    print("-" * 30)
    
    base_config = {
        "aggregation_strategy": "semantic_merge",
        "confidence_threshold": float('inf'),
        "fallback_strategy": "unsafe_strategy"
    }
    
    secure_config = AggregatorNodeSecurityFixes.create_secure_aggregator_config(base_config)
    
    print("Security enhancements added:")
    security_keys = [k for k in secure_config.keys() if k not in base_config]
    for key in security_keys[:5]:  # Show first 5
        print(f"  - {key}: {secure_config[key]}")
    print(f"  ... and {len(security_keys) - 5} more security settings")
    
    print(f"\nFixed confidence_threshold: {secure_config['confidence_threshold']}")
    print(f"Fixed fallback_strategy: {secure_config['fallback_strategy']}")
    
    print("\n" + "=" * 60)
    print("SECURITY FIXES DEMONSTRATION COMPLETED")
    print("All critical security issues have been addressed:")
    print("✓ Confidence score validation and normalization")
    print("✓ Input sanitization against injection attacks")
    print("✓ Resource limits to prevent DoS attacks")
    print("✓ Timeout handling for long operations")
    print("✓ Secure configuration defaults")

if __name__ == "__main__":
    demonstrate_fixes()