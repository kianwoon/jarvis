-- =====================================================
-- Meta-Task Templates Configuration Script
-- Configures strategic_business_plan and comprehensive_research_report templates
-- with proper prompts, settings, and schemas
-- =====================================================

-- Update the strategic_business_plan template
UPDATE meta_task_templates
SET 
    template_config = '{
        "phases": [
            {
                "name": "market_analysis",
                "type": "analyzer",
                "description": "Comprehensive market analysis including competition, trends, and customer segments",
                "prompt": "Analyze target market, competition, industry trends, and customer segments. Identify market opportunities and threats. Your analysis should include:\n\n1. **Market Size & Growth**: Current market size, growth projections, and key drivers\n2. **Target Customers**: Demographics, psychographics, needs, and pain points\n3. **Competitive Landscape**: Direct and indirect competitors, their strengths/weaknesses, market share\n4. **Industry Trends**: Emerging trends, technological disruptions, regulatory changes\n5. **Market Opportunities**: Untapped segments, gaps in the market, emerging needs\n6. **Threats & Challenges**: Market risks, competitive threats, regulatory challenges\n\nProvide data-driven insights with specific examples and actionable recommendations for market entry and positioning strategies."
            },
            {
                "name": "business_model",
                "type": "generator", 
                "description": "Define comprehensive business model including value proposition and revenue streams",
                "prompt": "Define value proposition, revenue streams, cost structure, key partnerships, and resources. Create detailed business model canvas covering:\n\n1. **Value Proposition**: Unique value offered to each customer segment\n2. **Revenue Streams**: Primary and secondary revenue sources, pricing strategy\n3. **Cost Structure**: Fixed and variable costs, cost drivers, economies of scale\n4. **Key Resources**: Critical assets needed (physical, intellectual, human, financial)\n5. **Key Activities**: Most important activities for value creation and delivery\n6. **Key Partnerships**: Strategic alliances, supplier relationships, joint ventures\n7. **Customer Relationships**: Types of relationships established with customer segments\n8. **Distribution Channels**: How value proposition reaches customers\n9. **Customer Segments**: Distinct groups of customers served\n\nEnsure the business model is sustainable, scalable, and aligned with market opportunities identified in the market analysis."
            },
            {
                "name": "financial_projections",
                "type": "generator",
                "description": "Create detailed 3-5 year financial projections with comprehensive analysis",
                "prompt": "Create 3-5 year financial projections including revenue forecasts, expense budgets, cash flow analysis, and break-even analysis. Your projections should include:\n\n1. **Revenue Projections**: Monthly/quarterly revenue forecasts by product/service line\n2. **Expense Budget**: Detailed breakdown of operating expenses (COGS, SG&A, R&D)\n3. **Cash Flow Analysis**: Operating, investing, and financing cash flows\n4. **Break-even Analysis**: Break-even point calculation and sensitivity analysis\n5. **Profit & Loss Statements**: Projected P&L for each year\n6. **Balance Sheet Projections**: Assets, liabilities, and equity forecasts\n7. **Key Financial Ratios**: Profitability, liquidity, and efficiency ratios\n8. **Scenario Analysis**: Best case, worst case, and most likely scenarios\n9. **Funding Requirements**: Capital needs, funding timeline, and sources\n10. **Return on Investment**: Expected ROI for investors and stakeholders\n\nBase assumptions on market analysis and business model. Include detailed explanations for all assumptions and methodologies used."
            },
            {
                "name": "implementation_plan",
                "type": "generator",
                "description": "Develop detailed implementation roadmap with milestones and resource allocation",
                "prompt": "Develop detailed implementation roadmap with milestones, timelines, resource allocation, and key performance indicators. Your plan should include:\n\n1. **Strategic Roadmap**: High-level implementation phases and major milestones\n2. **Detailed Timeline**: Month-by-month action plan for first year, quarterly for years 2-3\n3. **Resource Allocation**: Human resources, technology, equipment, and capital requirements\n4. **Organizational Structure**: Team structure, roles, responsibilities, and reporting lines\n5. **Operational Processes**: Key business processes, workflows, and standard operating procedures\n6. **Technology Infrastructure**: IT systems, software, hardware, and digital platforms needed\n7. **Marketing & Sales Strategy**: Customer acquisition, retention, and growth strategies\n8. **Key Performance Indicators**: Measurable KPIs for tracking progress and success\n9. **Risk Mitigation**: Contingency plans for identified risks and challenges\n10. **Quality Control**: Processes to ensure consistent delivery and continuous improvement\n\nEnsure the plan is realistic, achievable, and aligned with financial projections and market opportunities."
            },
            {
                "name": "risk_assessment",
                "type": "analyzer",
                "description": "Comprehensive risk identification, assessment, and mitigation strategies",
                "prompt": "Identify potential risks, assess their impact and likelihood, and develop mitigation strategies. Your assessment should cover:\n\n1. **Market Risks**: Changes in market conditions, customer preferences, competition\n2. **Financial Risks**: Cash flow issues, funding challenges, cost overruns\n3. **Operational Risks**: Supply chain disruptions, quality issues, capacity constraints\n4. **Technology Risks**: System failures, cybersecurity threats, obsolescence\n5. **Regulatory Risks**: Compliance issues, regulatory changes, legal challenges\n6. **Strategic Risks**: Strategic misalignment, execution failures, competitive responses\n7. **Human Resources Risks**: Key person dependency, talent acquisition, retention\n8. **Environmental Risks**: Natural disasters, climate change impacts, sustainability\n\nFor each risk category:\n- **Risk Identification**: Specific risks and their sources\n- **Impact Assessment**: Potential consequences (high/medium/low)\n- **Likelihood Assessment**: Probability of occurrence (high/medium/low)\n- **Risk Rating**: Combined impact and likelihood score\n- **Mitigation Strategies**: Specific actions to prevent or minimize risk\n- **Contingency Plans**: Response strategies if risks materialize\n- **Monitoring Methods**: How risks will be tracked and managed\n\nPrioritize risks by severity and provide actionable mitigation recommendations."
            },
            {
                "name": "executive_summary",
                "type": "assembler",
                "description": "Synthesize all sections into compelling executive summary with key insights",
                "prompt": "Synthesize all sections into a compelling executive summary highlighting key insights and recommendations. Your summary should:\n\n1. **Business Concept**: Clear, concise description of the business opportunity\n2. **Market Opportunity**: Key market insights and size of opportunity\n3. **Competitive Advantage**: Unique value proposition and differentiation\n4. **Business Model**: How the business will create and capture value\n5. **Financial Highlights**: Key financial projections and return expectations\n6. **Implementation Strategy**: High-level roadmap and key milestones\n7. **Risk Management**: Major risks and mitigation strategies\n8. **Investment Requirements**: Funding needs and use of funds\n9. **Expected Returns**: Financial projections and ROI for stakeholders\n10. **Call to Action**: Next steps and recommendations\n\nThe executive summary should be:\n- **Compelling**: Capture reader attention and generate interest\n- **Concise**: Maximum 2-3 pages covering all key points\n- **Complete**: Stand-alone document that tells the full story\n- **Credible**: Supported by data and realistic projections\n- **Clear**: Easy to understand for both technical and non-technical audiences\n\nEnsure consistency with all previous sections and highlight the most compelling aspects of the business opportunity."
            }
        ],
        "variables": [
            {
                "name": "business_name",
                "type": "string",
                "description": "Name of the business or venture",
                "required": true
            },
            {
                "name": "industry",
                "type": "string", 
                "description": "Primary industry or sector",
                "required": true
            },
            {
                "name": "target_market",
                "type": "string",
                "description": "Primary target market or customer segment",
                "required": false
            },
            {
                "name": "planning_horizon",
                "type": "integer",
                "description": "Planning period in years (3-5)",
                "default": 5,
                "required": false
            }
        ],
        "output_format": "markdown"
    }'::json,
    
    input_schema = '{
        "type": "object",
        "properties": {
            "business_name": {
                "type": "string",
                "description": "Name of the business or venture"
            },
            "industry": {
                "type": "string",
                "description": "Primary industry or sector"
            },
            "target_market": {
                "type": "string",
                "description": "Primary target market or customer segment"
            },
            "planning_horizon": {
                "type": "integer",
                "minimum": 3,
                "maximum": 10,
                "default": 5,
                "description": "Planning period in years"
            },
            "additional_context": {
                "type": "string",
                "description": "Any additional context or specific requirements"
            }
        },
        "required": ["business_name", "industry"]
    }'::json,
    
    output_schema = '{
        "type": "object",
        "properties": {
            "executive_summary": {
                "type": "string",
                "description": "Comprehensive executive summary"
            },
            "market_analysis": {
                "type": "object",
                "properties": {
                    "market_size": {"type": "string"},
                    "target_customers": {"type": "string"},
                    "competitive_landscape": {"type": "string"},
                    "opportunities": {"type": "array", "items": {"type": "string"}},
                    "threats": {"type": "array", "items": {"type": "string"}}
                }
            },
            "business_model": {
                "type": "object", 
                "properties": {
                    "value_proposition": {"type": "string"},
                    "revenue_streams": {"type": "array", "items": {"type": "string"}},
                    "cost_structure": {"type": "string"},
                    "key_resources": {"type": "array", "items": {"type": "string"}}
                }
            },
            "financial_projections": {
                "type": "object",
                "properties": {
                    "revenue_forecast": {"type": "string"},
                    "expense_budget": {"type": "string"},
                    "cash_flow": {"type": "string"},
                    "break_even": {"type": "string"}
                }
            },
            "implementation_plan": {
                "type": "object",
                "properties": {
                    "roadmap": {"type": "string"},
                    "milestones": {"type": "array", "items": {"type": "string"}},
                    "resources": {"type": "string"},
                    "kpis": {"type": "array", "items": {"type": "string"}}
                }
            },
            "risk_assessment": {
                "type": "object",
                "properties": {
                    "identified_risks": {"type": "array", "items": {"type": "string"}},
                    "mitigation_strategies": {"type": "array", "items": {"type": "string"}}
                }
            }
        }
    }'::json,
    
    default_settings = '{
        "model_config": {
            "analyzer_model": "qwen3:30b-a3b",
            "generator_model": "qwen3:30b-a3b",
            "assembler_model": "qwen3:30b-a3b",
            "temperature": 0.7,
            "max_tokens": 16000
        },
        "execution": {
            "max_phases": 6,
            "phase_timeout_minutes": 45,
            "retry_attempts": 2,
            "parallel_execution": false
        },
        "quality": {
            "minimum_quality_score": 0.8,
            "require_human_review": false,
            "enable_consistency_check": true
        }
    }'::json

WHERE name = 'strategic_business_plan';

-- Update the comprehensive_research_report template
UPDATE meta_task_templates
SET 
    template_config = '{
        "phases": [
            {
                "name": "research_analysis",
                "type": "analyzer",
                "description": "Analyze research topic and create comprehensive outline",
                "prompt": "Analyze the research topic: {{topic}}. Create a comprehensive outline covering all relevant aspects. Consider the target format: {{format}} and desired length: {{target_length}} pages. Your analysis should include:\n\n1. **Topic Scope Definition**: Clear boundaries and focus areas for the research\n2. **Research Questions**: Key questions that need to be answered\n3. **Literature Review Plan**: Relevant sources, databases, and search strategies\n4. **Methodology**: Research approach and analytical frameworks\n5. **Content Structure**: Detailed outline with main sections and subsections\n6. **Data Requirements**: Types of data, statistics, and evidence needed\n7. **Timeline**: Realistic timeline for research and writing phases\n8. **Quality Standards**: Criteria for evaluating sources and information quality\n\nEnsure the outline is:\n- **Comprehensive**: Covers all aspects of the topic thoroughly\n- **Logical**: Follows a clear, logical progression\n- **Balanced**: Appropriate depth and breadth for the target length\n- **Academic**: Meets standards for the specified format\n- **Feasible**: Achievable within reasonable time and resource constraints\n\nThe outline should serve as a detailed roadmap for content generation."
            },
            {
                "name": "content_generation",
                "type": "generator",
                "description": "Generate detailed content for each section with thorough analysis",
                "prompt": "Generate detailed content for each section of the outline. Ensure thorough coverage with examples, data, and analysis. Maintain academic rigor for the {{format}} format. Your content should include:\n\n1. **Introduction**: Clear problem statement, objectives, and scope\n2. **Literature Review**: Comprehensive review of relevant sources and existing knowledge\n3. **Methodology**: Detailed explanation of research approach and methods\n4. **Main Content Sections**: In-depth analysis of each topic area with:\n   - Evidence-based arguments\n   - Data and statistics where relevant\n   - Case studies and examples\n   - Critical analysis and interpretation\n   - Multiple perspectives and viewpoints\n5. **Discussion**: Synthesis of findings and implications\n6. **Conclusion**: Key insights, limitations, and recommendations\n\nContent Quality Standards:\n- **Academic Rigor**: Proper citations, scholarly sources, critical thinking\n- **Depth of Analysis**: Thorough exploration of concepts and issues\n- **Evidence-Based**: Strong support for all claims and arguments\n- **Clarity**: Clear, professional writing appropriate for the audience\n- **Originality**: Fresh insights and original analysis, not just summary\n- **Completeness**: No gaps or missing elements in the coverage\n\nFor {{target_length}} pages, ensure appropriate depth while maintaining engagement and readability."
            },
            {
                "name": "quality_review",
                "type": "reviewer",
                "description": "Review content for accuracy, coherence, and academic standards",
                "prompt": "Review generated content for accuracy, coherence, and completeness. Ensure proper citations, logical flow, and adherence to {{format}} standards. Your review should assess:\n\n1. **Content Quality**:\n   - Factual accuracy and reliability of information\n   - Logical flow and coherent arguments\n   - Completeness of coverage per the outline\n   - Depth and rigor of analysis\n\n2. **Academic Standards**:\n   - Proper citation format and consistency\n   - Quality and credibility of sources\n   - Appropriate academic tone and style\n   - Adherence to {{format}} conventions\n\n3. **Structure and Organization**:\n   - Clear section transitions\n   - Logical progression of ideas\n   - Balanced content distribution\n   - Effective use of headings and subheadings\n\n4. **Writing Quality**:\n   - Grammar, spelling, and punctuation\n   - Sentence structure and readability\n   - Consistent terminology usage\n   - Appropriate length for target audience\n\n5. **Critical Issues to Flag**:\n   - Factual errors or inconsistencies\n   - Missing citations or poor sources\n   - Logical gaps or weak arguments\n   - Sections needing expansion or reduction\n   - Format violations or style issues\n\nProvide specific recommendations for improvement, prioritized by importance. Flag any content that requires fact-checking or additional research."
            },
            {
                "name": "final_assembly",
                "type": "assembler",
                "description": "Assemble final document with professional formatting and structure",
                "prompt": "Assemble the final document with proper formatting, table of contents, references, and appendices. Ensure professional presentation meeting {{format}} standards. Your assembly should include:\n\n1. **Document Structure**:\n   - Title page with proper formatting\n   - Abstract/executive summary (if appropriate)\n   - Table of contents with page numbers\n   - List of figures and tables (if applicable)\n   - Main content with consistent formatting\n   - Conclusion and recommendations\n   - References/bibliography in proper format\n   - Appendices for supplementary material\n\n2. **Formatting Standards**:\n   - Consistent heading hierarchy (H1, H2, H3, etc.)\n   - Proper spacing and margins\n   - Professional font and sizing\n   - Page numbers and headers/footers\n   - Consistent citation format throughout\n   - Proper figure and table formatting\n\n3. **Content Integration**:\n   - Smooth transitions between sections\n   - Consistent terminology and style\n   - Cross-references and internal links\n   - Elimination of redundancy\n   - Logical information flow\n\n4. **Final Quality Checks**:\n   - All placeholders replaced\n   - Complete and accurate metadata\n   - Proper document properties\n   - Error-free formatting\n   - Compliance with {{format}} requirements\n\n5. **Professional Enhancements**:\n   - Executive summary for reports >10 pages\n   - Glossary for technical documents\n   - Index for reference materials\n   - Version control information\n   - Author and publication details\n\nDeliver a publication-ready document that meets professional standards and effectively communicates the research findings to the target audience."
            }
        ],
        "variables": [
            {
                "name": "topic",
                "type": "string",
                "description": "Main research topic or subject area",
                "required": true
            },
            {
                "name": "format",
                "type": "string",
                "description": "Target document format (academic paper, business report, white paper, etc.)",
                "default": "business report",
                "required": false
            },
            {
                "name": "target_length",
                "type": "integer",
                "description": "Desired document length in pages",
                "default": 20,
                "required": false
            },
            {
                "name": "audience",
                "type": "string",
                "description": "Target audience (executives, academics, general public, etc.)",
                "default": "business professionals",
                "required": false
            },
            {
                "name": "focus_areas",
                "type": "array",
                "description": "Specific areas or aspects to emphasize",
                "required": false
            }
        ],
        "output_format": "markdown"
    }'::json,
    
    input_schema = '{
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Main research topic or subject area"
            },
            "format": {
                "type": "string",
                "enum": ["academic paper", "business report", "white paper", "technical document", "market analysis", "industry report"],
                "default": "business report",
                "description": "Target document format"
            },
            "target_length": {
                "type": "integer",
                "minimum": 5,
                "maximum": 100,
                "default": 20,
                "description": "Desired document length in pages"
            },
            "audience": {
                "type": "string",
                "enum": ["executives", "academics", "technical professionals", "general public", "investors", "policymakers"],
                "default": "business professionals",
                "description": "Target audience"
            },
            "focus_areas": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific areas or aspects to emphasize"
            },
            "research_depth": {
                "type": "string",
                "enum": ["overview", "standard", "comprehensive", "deep-dive"],
                "default": "standard",
                "description": "Desired depth of research and analysis"
            },
            "deadline": {
                "type": "string",
                "format": "date",
                "description": "Target completion date"
            }
        },
        "required": ["topic"]
    }'::json,
    
    output_schema = '{
        "type": "object",
        "properties": {
            "final_document": {
                "type": "string",
                "description": "Complete formatted research report"
            },
            "executive_summary": {
                "type": "string",
                "description": "Executive summary of key findings"
            },
            "research_outline": {
                "type": "object",
                "properties": {
                    "main_sections": {"type": "array", "items": {"type": "string"}},
                    "key_questions": {"type": "array", "items": {"type": "string"}},
                    "methodology": {"type": "string"}
                }
            },
            "content_sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                        "key_points": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "references": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of references and sources used"
            },
            "quality_metrics": {
                "type": "object",
                "properties": {
                    "completeness_score": {"type": "number"},
                    "accuracy_score": {"type": "number"},
                    "readability_score": {"type": "number"}
                }
            }
        }
    }'::json,
    
    default_settings = '{
        "model_config": {
            "analyzer_model": "qwen3:30b-a3b",
            "generator_model": "qwen3:30b-a3b", 
            "reviewer_model": "qwen3:30b-a3b",
            "assembler_model": "qwen3:30b-a3b",
            "temperature": 0.6,
            "max_tokens": 20000
        },
        "execution": {
            "max_phases": 4,
            "phase_timeout_minutes": 60,
            "retry_attempts": 2,
            "parallel_execution": false
        },
        "quality": {
            "minimum_quality_score": 0.85,
            "require_human_review": false,
            "enable_consistency_check": true,
            "enable_factuality_check": true
        },
        "research": {
            "max_sources": 50,
            "min_source_quality": 0.7,
            "enable_web_search": true,
            "search_depth": "comprehensive"
        }
    }'::json

WHERE name = 'comprehensive_research_report';

-- Verify the updates
SELECT 
    name,
    template_type,
    description,
    is_active,
    json_extract_path_text(template_config, 'output_format') as output_format,
    json_array_length(json_extract_path(template_config, 'phases')) as phase_count,
    json_array_length(json_extract_path(template_config, 'variables')) as variable_count
FROM meta_task_templates 
WHERE name IN ('strategic_business_plan', 'comprehensive_research_report')
ORDER BY name;

-- Display the phases for verification
SELECT 
    t.name as template_name,
    p.value->>'name' as phase_name,
    p.value->>'type' as phase_type,
    length(p.value->>'prompt') as prompt_length
FROM meta_task_templates t,
     json_array_elements(t.template_config->'phases') p
WHERE t.name IN ('strategic_business_plan', 'comprehensive_research_report')
ORDER BY t.name, (p.ordinality-1);

-- Summary report
SELECT 
    '✅ Meta-Task Templates Configuration Complete' as status,
    COUNT(*) as templates_updated
FROM meta_task_templates 
WHERE name IN ('strategic_business_plan', 'comprehensive_research_report')
    AND template_config IS NOT NULL
    AND input_schema IS NOT NULL
    AND output_schema IS NOT NULL
    AND default_settings IS NOT NULL;