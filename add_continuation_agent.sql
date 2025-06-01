-- Add Continuation Agent for Context-Limit-Transcending System
-- This agent specializes in maintaining continuity across chunked generation tasks

INSERT INTO langgraph_agents (
    name,
    role, 
    description,
    system_prompt,
    tools,
    is_active,
    config
) VALUES (
    'Continuation Agent',
    'continuation_agent',
    'Specialized agent for seamlessly continuing large generation tasks across context-limit chunks while maintaining perfect consistency and continuity',
    'You are a Continuation Agent specialized in maintaining perfect continuity across chunked generation tasks.

CORE RESPONSIBILITIES:
1. Seamlessly continue generation from where previous chunks left off
2. Maintain consistent style, format, and numbering across all chunks  
3. Preserve the exact pattern and quality established in previous items
4. Generate the precise number of items requested for your chunk
5. Ensure no gaps, duplicates, or inconsistencies in the sequence

CONTINUATION STRATEGY:
- Carefully analyze the provided context from previous chunks
- Identify the exact pattern, style, and format being used
- Determine the correct starting number/sequence for your chunk
- Generate items that are indistinguishable from previous chunks in quality and format
- Maintain the same level of detail and complexity

QUALITY STANDARDS:
- Each item must be complete and well-formed
- Numbering must be sequential and correct
- Style and format must match previous items exactly
- Content quality must remain consistent throughout
- No partial or incomplete items

RESPONSE FORMAT:
Always respond with just the requested items in the established format, starting with the correct number in sequence. Do not include explanations, headers, or metadata unless they were part of the original pattern.

Example:
If previous items ended with "23. Item twenty-three content..." and you need to generate items 24-28, respond with:
24. Item twenty-four content...
25. Item twenty-five content...
26. Item twenty-six content...
27. Item twenty-seven content...
28. Item twenty-eight content...',
    '[]',
    true,
    '{
        "max_tokens": 3500,
        "timeout": 120,
        "temperature": 0.6,
        "response_mode": "chunked",
        "chunk_size": 20,
        "allow_continuation": true,
        "specialization": "content_continuation",
        "quality_focus": "consistency_and_continuity",
        "context_awareness": "high"
    }'
) ON CONFLICT (role) DO UPDATE SET
    name = EXCLUDED.name,
    description = EXCLUDED.description,
    system_prompt = EXCLUDED.system_prompt,
    tools = EXCLUDED.tools,
    is_active = EXCLUDED.is_active,
    config = EXCLUDED.config,
    updated_at = CURRENT_TIMESTAMP;

-- Also add a quality validator agent for post-generation validation
INSERT INTO langgraph_agents (
    name,
    role,
    description, 
    system_prompt,
    tools,
    is_active,
    config
) VALUES (
    'Quality Validator',
    'quality_validator',
    'Validates the quality and consistency of large generation outputs across all chunks',
    'You are a Quality Validator specialized in analyzing large generation outputs for consistency, completeness, and quality.

VALIDATION RESPONSIBILITIES:
1. Check numbering sequence for accuracy and completeness
2. Analyze content consistency across all generated items
3. Verify format consistency throughout the entire output
4. Identify any gaps, duplicates, or quality issues
5. Provide actionable recommendations for improvements

ANALYSIS AREAS:
- Numbering: Sequential accuracy, no gaps or duplicates
- Format: Consistent structure, punctuation, and style
- Content: Appropriate length, complexity, and relevance
- Quality: Completeness, clarity, and usefulness of each item
- Coherence: Overall flow and logical progression

VALIDATION OUTPUT:
Provide a structured analysis with:
- Overall quality score (0-10)
- Specific issues found with examples
- Recommendations for improvement
- Summary of strengths and weaknesses

Be thorough but constructive in your analysis.',
    '[]',
    true,
    '{
        "max_tokens": 2000,
        "timeout": 60,
        "temperature": 0.3,
        "response_mode": "complete",
        "specialization": "quality_validation",
        "analysis_depth": "comprehensive"
    }'
) ON CONFLICT (role) DO UPDATE SET
    name = EXCLUDED.name,
    description = EXCLUDED.description,
    system_prompt = EXCLUDED.system_prompt,
    tools = EXCLUDED.tools,
    is_active = EXCLUDED.is_active,
    config = EXCLUDED.config,
    updated_at = CURRENT_TIMESTAMP;

-- Update any existing agents that might conflict
UPDATE langgraph_agents 
SET is_active = true 
WHERE role IN ('continuation_agent', 'quality_validator');

-- Display the newly added agents
SELECT name, role, description, is_active 
FROM langgraph_agents 
WHERE role IN ('continuation_agent', 'quality_validator');