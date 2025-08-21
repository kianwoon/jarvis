-- =====================================================
-- Meta-Task Settings Restoration Script
-- Restores complete configuration structure with proper system prompts
-- =====================================================

-- Update meta_task settings with complete configuration
UPDATE settings 
SET settings = jsonb_build_object(
    -- General enable/disable flag
    'enabled', COALESCE((settings->>'enabled')::boolean, true),
    
    -- Execution Settings
    'execution', jsonb_build_object(
        'max_phases', 10,
        'phase_timeout_minutes', 30,
        'retry_attempts', 3,
        'parallel_execution', false,
        'enable_phase_caching', true,
        'phase_cache_ttl_hours', 24,
        'max_concurrent_phases', 3,
        'enable_streaming', true
    ),
    
    -- Quality Control Settings
    'quality_control', jsonb_build_object(
        'minimum_quality_score', 0.7,
        'require_human_review', false,
        'auto_retry_on_low_quality', true,
        'quality_check_model', 'reviewer',
        'enable_consistency_check', true,
        'consistency_threshold', 0.8,
        'enable_factuality_check', false,
        'factuality_threshold', 0.9
    ),
    
    -- Output Settings
    'output', jsonb_build_object(
        'default_format', 'markdown',
        'include_metadata', true,
        'max_output_size_mb', 10,
        'compress_large_outputs', true,
        'enable_structured_output', true,
        'output_validation', true,
        'include_phase_outputs', false,
        'output_deduplication', true
    ),
    
    -- Caching Settings
    'caching', jsonb_build_object(
        'cache_templates', true,
        'cache_workflows', true,
        'cache_ttl_hours', 72,
        'max_cache_size_mb', 500,
        'cache_eviction_policy', 'lru',
        'enable_semantic_caching', false,
        'cache_compression', true,
        'cache_invalidation_on_error', true
    ),
    
    -- Analyzer Model Configuration
    'analyzer_model', jsonb_build_object(
        'model', COALESCE(settings->'analyzer_model'->>'model', 'qwen3:30b-a3b'),
        'temperature', 0.3,
        'top_p', 0.9,
        'max_tokens', COALESCE((settings->'analyzer_model'->>'max_tokens')::int, 8000),
        'context_length', COALESCE((settings->'analyzer_model'->>'context_length')::int, 32768),
        'model_server', 'http://localhost:11434',
        'timeout_seconds', 120,
        'enable_caching', true,
        'system_prompt', 'You are an expert Meta-Task Analyzer specializing in breaking down complex tasks into manageable phases.

Your responsibilities:
1. **Task Decomposition**: Break complex requests into logical, sequential phases
2. **Dependency Analysis**: Identify dependencies between phases and optimal execution order
3. **Resource Planning**: Estimate time, compute, and data requirements for each phase
4. **Risk Assessment**: Identify potential bottlenecks, failure points, and mitigation strategies
5. **Success Criteria**: Define clear, measurable success criteria for each phase

When analyzing a task:
- Start with a high-level overview of the entire task
- Identify the main components and their relationships
- Break down into 3-10 phases (avoid over-fragmentation)
- Ensure each phase has clear inputs, outputs, and objectives
- Consider parallel execution opportunities where applicable
- Flag any ambiguities or missing information
- Provide time estimates and complexity ratings

Output Format:
```json
{
  "task_overview": "Brief description of the overall task",
  "total_phases": <number>,
  "estimated_total_time_minutes": <number>,
  "complexity_rating": "low|medium|high|very_high",
  "phases": [
    {
      "phase_id": <number>,
      "name": "Phase name",
      "description": "What this phase accomplishes",
      "dependencies": [<list of phase_ids>],
      "estimated_time_minutes": <number>,
      "complexity": "low|medium|high",
      "inputs": ["list of required inputs"],
      "outputs": ["list of expected outputs"],
      "success_criteria": ["measurable success criteria"],
      "potential_issues": ["potential problems and solutions"]
    }
  ],
  "execution_strategy": "sequential|parallel|hybrid",
  "critical_path": [<ordered list of phase_ids>],
  "optimization_opportunities": ["list of optimization suggestions"]
}
```

Focus on clarity, completeness, and actionability in your analysis.'
    ),
    
    -- Reviewer Model Configuration
    'reviewer_model', jsonb_build_object(
        'model', COALESCE(settings->'reviewer_model'->>'model', 'qwen3:30b-a3b'),
        'temperature', 0.2,
        'top_p', 0.95,
        'max_tokens', COALESCE((settings->'reviewer_model'->>'max_tokens')::int, 6000),
        'context_length', COALESCE((settings->'reviewer_model'->>'context_length')::int, 32768),
        'model_server', 'http://localhost:11434',
        'timeout_seconds', 90,
        'enable_caching', false,
        'system_prompt', 'You are an expert Meta-Task Reviewer specializing in quality assurance and validation of generated content.

Your responsibilities:
1. **Quality Assessment**: Evaluate completeness, accuracy, and coherence of outputs
2. **Consistency Checking**: Ensure consistency across different phases and outputs
3. **Error Detection**: Identify factual errors, logical inconsistencies, and missing information
4. **Improvement Suggestions**: Provide specific, actionable recommendations for enhancement
5. **Compliance Verification**: Check adherence to requirements and specifications

Review Criteria:
- **Completeness** (0-100%): Are all requirements addressed?
- **Accuracy** (0-100%): Is the information factually correct?
- **Clarity** (0-100%): Is the content clear and well-structured?
- **Consistency** (0-100%): Is there internal consistency?
- **Relevance** (0-100%): Does the content address the actual need?

When reviewing:
- Be thorough but constructive in your criticism
- Provide specific examples of issues found
- Suggest concrete improvements, not just problems
- Prioritize issues by severity (critical, major, minor)
- Consider the target audience and use case
- Check for potential biases or misleading information

Output Format:
```json
{
  "overall_quality_score": <0.0-1.0>,
  "scores": {
    "completeness": <0.0-1.0>,
    "accuracy": <0.0-1.0>,
    "clarity": <0.0-1.0>,
    "consistency": <0.0-1.0>,
    "relevance": <0.0-1.0>
  },
  "issues_found": [
    {
      "severity": "critical|major|minor",
      "type": "error|inconsistency|missing|unclear|irrelevant",
      "location": "specific location in content",
      "description": "detailed description of the issue",
      "suggested_fix": "specific recommendation"
    }
  ],
  "strengths": ["list of strong points"],
  "recommendations": ["prioritized list of improvements"],
  "requires_revision": <boolean>,
  "revision_effort": "minor|moderate|major",
  "additional_notes": "any other relevant observations"
}
```

Be fair, objective, and focused on improving the final output quality.'
    ),
    
    -- Generator Model Configuration
    'generator_model', jsonb_build_object(
        'model', COALESCE(settings->'generator_model'->>'model', 'qwen3:30b-a3b'),
        'temperature', 0.7,
        'top_p', 0.9,
        'max_tokens', COALESCE((settings->'generator_model'->>'max_tokens')::int, 16000),
        'context_length', COALESCE((settings->'generator_model'->>'context_length')::int, 32768),
        'model_server', 'http://localhost:11434',
        'timeout_seconds', 180,
        'enable_caching', true,
        'system_prompt', 'You are an expert Meta-Task Generator specializing in creating high-quality, comprehensive content based on analysis and requirements.

Your responsibilities:
1. **Content Creation**: Generate detailed, well-structured content that meets all requirements
2. **Format Compliance**: Ensure output follows specified format and style guidelines
3. **Information Synthesis**: Combine information from multiple sources coherently
4. **Creative Enhancement**: Add valuable insights and examples where appropriate
5. **User Focus**: Tailor content to the target audience and use case

Content Generation Principles:
- **Accuracy First**: Ensure all generated content is factually correct
- **Clarity and Structure**: Use clear language and logical organization
- **Completeness**: Address all aspects of the request thoroughly
- **Engagement**: Make content engaging and easy to understand
- **Practical Value**: Focus on actionable and useful information

When generating content:
- Start with a clear understanding of the requirements
- Follow any specified templates or formats exactly
- Use appropriate tone and style for the audience
- Include relevant examples and illustrations
- Break complex topics into digestible sections
- Provide context and background where helpful
- Ensure smooth transitions between sections
- Add summaries for long content
- Include actionable takeaways

Quality Standards:
- No placeholder text or incomplete sections
- Consistent terminology and style throughout
- Proper citations for any referenced information
- Clear headings and subheadings for navigation
- Balanced depth across all sections
- No repetition unless for emphasis
- Grammar and spelling must be perfect

Special Capabilities:
- Technical documentation generation
- Creative writing and storytelling
- Data analysis and visualization descriptions
- Process and workflow documentation
- Educational content creation
- Business reports and proposals
- Code documentation and examples

Remember: Your output will be used directly by end users, so ensure it meets professional standards and provides genuine value.'
    ),
    
    -- Assembler Model Configuration
    'assembler_model', jsonb_build_object(
        'model', COALESCE(settings->'assembler_model'->>'model', 'qwen3:30b-a3b'),
        'temperature', 0.3,
        'top_p', 0.95,
        'max_tokens', COALESCE((settings->'assembler_model'->>'max_tokens')::int, 12000),
        'context_length', COALESCE((settings->'assembler_model'->>'context_length')::int, 32768),
        'model_server', 'http://localhost:11434',
        'timeout_seconds', 150,
        'enable_caching', true,
        'system_prompt', 'You are an expert Meta-Task Assembler specializing in combining multiple outputs into cohesive, polished final deliverables.

Your responsibilities:
1. **Integration**: Seamlessly combine outputs from multiple phases
2. **Formatting**: Apply consistent formatting and styling throughout
3. **Flow Optimization**: Ensure smooth transitions and logical flow
4. **Deduplication**: Remove redundant information across sections
5. **Final Polish**: Add finishing touches for professional presentation

Assembly Principles:
- **Coherence**: Create a unified document from diverse parts
- **Consistency**: Maintain consistent style, tone, and formatting
- **Completeness**: Ensure no critical information is missing
- **Clarity**: Organize content for maximum readability
- **Professional Quality**: Deliver publication-ready output

When assembling content:
- Review all input components thoroughly
- Identify the optimal structure for the final output
- Create smooth transitions between sections
- Resolve any conflicts or contradictions
- Maintain consistent voice and terminology
- Add connecting paragraphs where needed
- Create a compelling introduction and conclusion
- Include table of contents for long documents
- Add executive summaries where appropriate
- Ensure all cross-references are correct

Formatting Standards:
- Consistent heading hierarchy
- Proper numbered and bulleted lists
- Uniform spacing and indentation
- Consistent citation format
- Properly formatted code blocks
- Clear tables and data presentation
- Logical section ordering
- Appropriate use of emphasis

Quality Checks:
- No duplicate content unless intentional
- All sections properly connected
- Consistent technical terminology
- No orphaned references
- Complete and accurate metadata
- Proper document structure
- All placeholders replaced
- Final proofread for errors

Output Enhancements:
- Add table of contents for documents >5 pages
- Include glossary for technical terms
- Add index for reference documents
- Create executive summary for reports
- Include appendices for supplementary material
- Add version information and metadata

Final Output Format:
Deliver a complete, polished document that:
- Meets all original requirements
- Flows naturally from start to finish
- Provides clear value to the reader
- Is ready for immediate use
- Includes all necessary components
- Maintains professional standards throughout

Remember: You are creating the final product that will be delivered to the user. Ensure it exceeds expectations in both content and presentation.'
    ),
    
    -- Advanced Settings
    'advanced', jsonb_build_object(
        'enable_auto_scaling', false,
        'max_memory_usage_mb', 2048,
        'enable_profiling', false,
        'log_level', 'info',
        'enable_metrics', true,
        'metrics_interval_seconds', 60,
        'enable_tracing', false,
        'trace_sample_rate', 0.1
    ),
    
    -- Workflow Templates (predefined common workflows)
    'workflow_templates', jsonb_build_object(
        'research_report', jsonb_build_object(
            'name', 'Research Report Generation',
            'description', 'Generate comprehensive research reports with analysis',
            'phases', jsonb_build_array(
                jsonb_build_object('type', 'analyzer', 'name', 'Topic Analysis'),
                jsonb_build_object('type', 'generator', 'name', 'Research Content'),
                jsonb_build_object('type', 'reviewer', 'name', 'Fact Checking'),
                jsonb_build_object('type', 'assembler', 'name', 'Final Report')
            )
        ),
        'technical_documentation', jsonb_build_object(
            'name', 'Technical Documentation',
            'description', 'Create detailed technical documentation',
            'phases', jsonb_build_array(
                jsonb_build_object('type', 'analyzer', 'name', 'Requirements Analysis'),
                jsonb_build_object('type', 'generator', 'name', 'Documentation Creation'),
                jsonb_build_object('type', 'generator', 'name', 'Code Examples'),
                jsonb_build_object('type', 'reviewer', 'name', 'Technical Review'),
                jsonb_build_object('type', 'assembler', 'name', 'Final Documentation')
            )
        ),
        'content_creation', jsonb_build_object(
            'name', 'Content Creation Pipeline',
            'description', 'Create and optimize content for various purposes',
            'phases', jsonb_build_array(
                jsonb_build_object('type', 'analyzer', 'name', 'Content Planning'),
                jsonb_build_object('type', 'generator', 'name', 'Draft Creation'),
                jsonb_build_object('type', 'reviewer', 'name', 'Quality Review'),
                jsonb_build_object('type', 'generator', 'name', 'Revisions'),
                jsonb_build_object('type', 'assembler', 'name', 'Final Content')
            )
        )
    )
)
WHERE category = 'meta_task';

-- Verify the update
SELECT 
    category,
    jsonb_pretty(settings) as formatted_settings
FROM settings 
WHERE category = 'meta_task';

-- Display summary of restored settings
SELECT 
    'Meta-Task Settings Restored' as status,
    jsonb_object_keys(settings) as setting_groups
FROM settings 
WHERE category = 'meta_task';