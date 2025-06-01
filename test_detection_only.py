#!/usr/bin/env python3
"""
Simple test for large output detection logic only
"""

import re
from datetime import datetime

def detect_large_output_potential(question: str) -> dict:
    """Detect if question will likely produce large output requiring chunked processing"""
    large_output_indicators = [
        "generate", "create", "list", "write", "develop", "design", "build",
        "comprehensive", "detailed", "complete", "full", "extensive", "thorough",
        "step by step", "step-by-step", "all", "many", "multiple", "various",
        "questions", "examples", "ideas", "recommendations", "strategies", "options",
        "points", "items", "factors", "aspects", "benefits", "advantages", "features"
    ]
    
    # Count indicator patterns
    score = 0
    question_lower = question.lower()
    matched_indicators = []
    
    for indicator in large_output_indicators:
        if indicator in question_lower:
            score += 1
            matched_indicators.append(indicator)
    
    # Extract numbers that might indicate quantity
    numbers = re.findall(r'\b(\d+)\b', question)
    max_number = max([int(n) for n in numbers], default=0)
    
    # Additional patterns that suggest large output
    large_patterns = [
        r'\b(\d+)\s+(questions|examples|items|points|ideas|strategies|options|factors|aspects|benefits|features)',
        r'(comprehensive|detailed|complete|full|extensive|thorough)\s+(list|guide|analysis|overview|breakdown)',
        r'(all|many|multiple|various)\s+(ways|methods|approaches|techniques|strategies|options)',
        r'generate.*\b(\d+)',
        r'create.*\b(\d+)',
        r'list.*\b(\d+)'
    ]
    
    pattern_matches = []
    for pattern in large_patterns:
        matches = re.findall(pattern, question_lower)
        if matches:
            pattern_matches.extend(matches)
            score += 2  # Patterns get higher weight
    
    # Calculate confidence and estimated items
    base_confidence = min(1.0, score / 5.0)
    number_confidence = min(1.0, max_number / 100.0) if max_number > 0 else 0
    final_confidence = max(base_confidence, number_confidence)
    
    # Estimate number of items to generate
    if max_number > 10:
        estimated_items = max_number
    elif score >= 3:
        estimated_items = score * 15  # Heuristic: more indicators = more items
    elif any(keyword in question_lower for keyword in ["comprehensive", "detailed", "complete"]):
        estimated_items = 30  # Default for comprehensive requests
    else:
        estimated_items = 10
    
    # More refined logic for determining if it's a large generation request
    is_likely_large = False
    
    # Strong indicators: explicit large numbers
    if max_number >= 30:
        is_likely_large = True
    # Medium indicators: moderate numbers + generation keywords
    elif max_number >= 20 and score >= 2:
        is_likely_large = True
    # Pattern-based indicators: multiple strong keywords suggesting comprehensive content
    elif score >= 3 and any(keyword in question_lower for keyword in ["comprehensive", "detailed", "all", "many"]):
        is_likely_large = True
    # Don't trigger for small numbers even with keywords
    elif max_number > 0 and max_number < 20:
        is_likely_large = False
    
    # Adjust estimated items for small number requests
    if max_number > 0 and max_number < 20:
        estimated_items = max_number
    
    result = {
        "likely_large": is_likely_large,
        "estimated_items": estimated_items,
        "confidence": final_confidence,
        "score": score,
        "max_number": max_number,
        "matched_indicators": matched_indicators,
        "pattern_matches": pattern_matches
    }
    
    return result

def test_large_output_detection():
    """Test the large output detection logic"""
    
    test_cases = [
        # Should trigger large generation
        {
            "question": "Generate 50 interview questions for software engineers",
            "expected_large": True,
            "expected_count_range": (40, 60)
        },
        {
            "question": "Create a comprehensive list of 100 marketing strategies",
            "expected_large": True,
            "expected_count_range": (90, 110)
        },
        {
            "question": "Write detailed explanations for all machine learning algorithms",
            "expected_large": True,
            "expected_count_range": (25, 35)
        },
        {
            "question": "List many examples of successful startups and their strategies",
            "expected_large": True,
            "expected_count_range": (25, 35)
        },
        {
            "question": "Generate 30 creative ideas for improving customer satisfaction",
            "expected_large": True,
            "expected_count_range": (25, 35)
        },
        
        # Should NOT trigger large generation
        {
            "question": "What is machine learning?",
            "expected_large": False,
            "expected_count_range": (5, 15)
        },
        {
            "question": "Explain the benefits of cloud computing",
            "expected_large": False,
            "expected_count_range": (5, 15)
        },
        {
            "question": "Generate 5 ideas for improving our website",
            "expected_large": False,
            "expected_count_range": (5, 10)
        },
        {
            "question": "How does React work?",
            "expected_large": False,
            "expected_count_range": (5, 15)
        }
    ]
    
    print("🧪 Testing Large Output Detection Logic")
    print("=" * 70)
    
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        expected_large = test_case["expected_large"]
        expected_range = test_case["expected_count_range"]
        
        print(f"\nTest {i}: {question}")
        print("-" * 60)
        
        # Test detection
        result = detect_large_output_potential(question)
        
        print(f"Detected large: {result['likely_large']} (expected: {expected_large})")
        print(f"Estimated items: {result['estimated_items']} (expected: {expected_range[0]}-{expected_range[1]})")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Score: {result['score']}")
        print(f"Max number: {result['max_number']}")
        print(f"Matched indicators: {result['matched_indicators']}")
        if result['pattern_matches']:
            print(f"Pattern matches: {result['pattern_matches']}")
        
        # Check if detection is correct
        detection_correct = result['likely_large'] == expected_large
        count_in_range = expected_range[0] <= result['estimated_items'] <= expected_range[1]
        
        if detection_correct and count_in_range:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
        
        print(f"Status: {status}")
        
        if not detection_correct:
            print(f"  ⚠️  Detection mismatch: got {result['likely_large']}, expected {expected_large}")
        if not count_in_range:
            print(f"  ⚠️  Count out of range: got {result['estimated_items']}, expected {expected_range[0]}-{expected_range[1]}")
    
    print(f"\n{'='*70}")
    print(f"🎯 Test Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Large output detection is working correctly.")
    else:
        print("⚠️  Some tests failed. Detection logic may need adjustment.")
    
    return passed == total

if __name__ == "__main__":
    print("🚀 Enhanced Standard Chat - Detection Test")
    print("Testing context-limit-transcending detection capabilities")
    print("=" * 70)
    
    success = test_large_output_detection()
    
    if success:
        print("\n✅ Detection system is ready!")
        print("Standard chat will now automatically handle large output requests")
        print("by routing them to the chunked generation system.")
    else:
        print("\n❌ Detection system needs refinement.")