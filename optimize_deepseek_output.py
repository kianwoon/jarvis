#!/usr/bin/env python3
"""
Optimize DeepSeek R1 for more detailed visible output
"""

print("=" * 70)
print("DeepSeek R1 Output Optimization Analysis")
print("=" * 70)

print("\nCURRENT ISSUE:")
print("- Total output: 5660 characters")
print("- Hidden reasoning: ~3757 characters (66%)")
print("- Visible answer: 1903 characters (34%)")
print("\nThe model is using most tokens for thinking, not answering!")

print("\n" + "=" * 70)
print("SOLUTIONS:")

print("\n1. USE NON-THINKING MODE:")
print("   - Set thinking=False in your requests")
print("   - This prevents <think> tag generation")
print("   - All tokens go to visible output")

print("\n2. ADJUST SYSTEM PROMPT:")
print("   Current: 'provides accurate info and response'")
print("   Better: 'Provide detailed, comprehensive responses with examples and explanations'")

print("\n3. PROMPT ENGINEERING:")
print("   Add to queries:")
print("   - 'Provide a detailed explanation with examples'")
print("   - 'Be comprehensive and thorough'")
print("   - 'Include all relevant information'")

print("\n4. TEMPERATURE ADJUSTMENT:")
print("   - Increase temperature to 0.8-0.9")
print("   - Higher temperature = more elaborate responses")

print("\n5. COMPARISON WITH QWEN3:")
print("   - Qwen3 doesn't use thinking tags by default")
print("   - All its tokens go to visible output")
print("   - This makes it appear more detailed")

print("\n" + "=" * 70)
print("RECOMMENDED SETTINGS:")
print("""
{
    "model": "deepseek-r1:8b",
    "max_tokens": "16384",
    "system_prompt": "You are Jarvis, an AI assistant. Always provide detailed, comprehensive responses with thorough explanations, examples, and step-by-step breakdowns when appropriate. Be verbose and informative. Use clean markdown formatting.",
    "thinking_mode": {
        "temperature": 0.8,
        "top_p": 0.95
    },
    "non_thinking_mode": {
        "temperature": 0.85,
        "top_p": 0.95
    }
}
""")

print("=" * 70)