-- Meta-Task System Database Migration
-- Creates 5 tables for comprehensive meta-task workflow management

-- 1. Template definitions for different document types
CREATE TABLE IF NOT EXISTS meta_task_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    template_type VARCHAR(100) NOT NULL,
    template_config JSONB NOT NULL,
    input_schema JSONB,
    output_schema JSONB,
    default_settings JSONB,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 2. Active workflow instances
CREATE TABLE IF NOT EXISTS meta_task_workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_id UUID NOT NULL REFERENCES meta_task_templates(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    workflow_config JSONB NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    input_data JSONB,
    output_data JSONB,
    progress JSONB, -- Track overall workflow progress
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT meta_task_workflows_status_check 
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'paused'))
);

-- 3. Individual workflow nodes/phases
CREATE TABLE IF NOT EXISTS meta_task_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES meta_task_workflows(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    node_type VARCHAR(100) NOT NULL,
    node_config JSONB NOT NULL,
    position_x INTEGER DEFAULT 0,
    position_y INTEGER DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    input_data JSONB,
    output_data JSONB,
    execution_order INTEGER,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT meta_task_nodes_status_check 
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped', 'cancelled'))
);

-- 4. Dependencies between nodes
CREATE TABLE IF NOT EXISTS meta_task_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES meta_task_workflows(id) ON DELETE CASCADE,
    source_node_id UUID NOT NULL REFERENCES meta_task_nodes(id) ON DELETE CASCADE,
    target_node_id UUID NOT NULL REFERENCES meta_task_nodes(id) ON DELETE CASCADE,
    edge_type VARCHAR(50) DEFAULT 'dependency',
    edge_config JSONB, -- Conditions, rules, data mapping
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT meta_task_edges_no_self_loop 
        CHECK (source_node_id != target_node_id)
);

-- 5. Detailed execution tracking and results
CREATE TABLE IF NOT EXISTS meta_task_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id UUID NOT NULL REFERENCES meta_task_nodes(id) ON DELETE CASCADE,
    execution_order INTEGER NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    input_data JSONB,
    output_data JSONB,
    execution_metadata JSONB, -- LLM model used, tokens, etc.
    error_message TEXT,
    error_stack_trace TEXT,
    execution_time_ms INTEGER,
    tokens_used INTEGER,
    cost_estimate DECIMAL(10,4),
    retry_count INTEGER DEFAULT 0,
    parent_execution_id UUID REFERENCES meta_task_executions(id),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT meta_task_executions_status_check 
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'retrying', 'cancelled'))
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_meta_task_workflows_template_id ON meta_task_workflows(template_id);
CREATE INDEX IF NOT EXISTS idx_meta_task_workflows_status ON meta_task_workflows(status);
CREATE INDEX IF NOT EXISTS idx_meta_task_workflows_created_at ON meta_task_workflows(created_at);

CREATE INDEX IF NOT EXISTS idx_meta_task_nodes_workflow_id ON meta_task_nodes(workflow_id);
CREATE INDEX IF NOT EXISTS idx_meta_task_nodes_status ON meta_task_nodes(status);
CREATE INDEX IF NOT EXISTS idx_meta_task_nodes_execution_order ON meta_task_nodes(workflow_id, execution_order);

CREATE INDEX IF NOT EXISTS idx_meta_task_edges_workflow_id ON meta_task_edges(workflow_id);
CREATE INDEX IF NOT EXISTS idx_meta_task_edges_source_node_id ON meta_task_edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_meta_task_edges_target_node_id ON meta_task_edges(target_node_id);

CREATE INDEX IF NOT EXISTS idx_meta_task_executions_node_id ON meta_task_executions(node_id);
CREATE INDEX IF NOT EXISTS idx_meta_task_executions_status ON meta_task_executions(status);
CREATE INDEX IF NOT EXISTS idx_meta_task_executions_started_at ON meta_task_executions(started_at);

-- Insert default meta-task templates
INSERT INTO meta_task_templates (name, description, template_type, template_config, input_schema, default_settings) 
VALUES 
(
    'comprehensive_research_report',
    'Generate comprehensive research reports that exceed token limits through multi-phase execution',
    'document_generation',
    '{
        "phases": [
            {"name": "research_analysis", "type": "analyzer", "description": "Analyze research requirements and create outline"},
            {"name": "content_generation", "type": "generator", "description": "Generate detailed content sections"},
            {"name": "quality_review", "type": "reviewer", "description": "Review and refine generated content"},
            {"name": "final_assembly", "type": "assembler", "description": "Assemble final document with formatting"}
        ],
        "max_tokens_per_phase": 3000,
        "quality_threshold": 0.85,
        "requires_human_review": false
    }',
    '{
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "Research topic or question"},
            "target_length": {"type": "integer", "description": "Target document length in pages"},
            "format": {"type": "string", "enum": ["academic", "business", "technical"], "default": "business"},
            "sources": {"type": "array", "items": {"type": "string"}, "description": "Optional source materials"}
        },
        "required": ["topic", "target_length"]
    }',
    '{
        "thinking_mode": {"model": "qwen3:30b-a3b", "temperature": 0.3},
        "generation_mode": {"model": "qwen3:30b-a3b", "temperature": 0.7},
        "review_mode": {"model": "qwen3:30b-a3b", "temperature": 0.1},
        "max_retries": 2,
        "quality_checks": true
    }'
),
(
    'strategic_business_plan',
    'Create detailed business plans with market analysis, financial projections, and implementation roadmaps',
    'document_generation',
    '{
        "phases": [
            {"name": "market_analysis", "type": "analyzer", "description": "Analyze market conditions and opportunities"},
            {"name": "business_model", "type": "generator", "description": "Develop business model and value proposition"},
            {"name": "financial_projections", "type": "generator", "description": "Create financial models and projections"},
            {"name": "implementation_plan", "type": "generator", "description": "Develop implementation roadmap"},
            {"name": "risk_assessment", "type": "analyzer", "description": "Identify and assess business risks"},
            {"name": "executive_summary", "type": "assembler", "description": "Create executive summary and final assembly"}
        ],
        "max_tokens_per_phase": 4000,
        "quality_threshold": 0.90,
        "requires_human_review": true
    }',
    '{
        "type": "object",
        "properties": {
            "business_concept": {"type": "string", "description": "Core business concept or idea"},
            "industry": {"type": "string", "description": "Target industry or sector"},
            "target_market": {"type": "string", "description": "Primary target market"},
            "funding_required": {"type": "number", "description": "Estimated funding requirements"},
            "timeline": {"type": "string", "description": "Business launch timeline"}
        },
        "required": ["business_concept", "industry", "target_market"]
    }',
    '{
        "thinking_mode": {"model": "qwen3:30b-a3b", "temperature": 0.2},
        "generation_mode": {"model": "qwen3:30b-a3b", "temperature": 0.6},
        "review_mode": {"model": "qwen3:30b-a3b", "temperature": 0.1},
        "max_retries": 3,
        "quality_checks": true,
        "human_review_points": ["financial_projections", "risk_assessment"]
    }'
) ON CONFLICT (name) DO NOTHING;

-- Add update triggers to maintain updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_meta_task_templates_updated_at BEFORE UPDATE ON meta_task_templates FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_meta_task_workflows_updated_at BEFORE UPDATE ON meta_task_workflows FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_meta_task_nodes_updated_at BEFORE UPDATE ON meta_task_nodes FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_meta_task_edges_updated_at BEFORE UPDATE ON meta_task_edges FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_meta_task_executions_updated_at BEFORE UPDATE ON meta_task_executions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();