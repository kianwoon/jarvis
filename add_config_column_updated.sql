-- Add config column to langgraph_agents table
ALTER TABLE langgraph_agents ADD COLUMN IF NOT EXISTS config JSON DEFAULT '{}';

-- Set default configuration for all agents
UPDATE langgraph_agents 
SET config = '{
  "max_tokens": 4000,
  "timeout": 30,
  "temperature": 0.7,
  "response_mode": "complete"
}'
WHERE config IS NULL OR config = '{}';

-- RouterAgent - quick decisions
UPDATE langgraph_agents 
SET config = '{
  "max_tokens": 1500,
  "timeout": 15,
  "temperature": 0.3,
  "response_mode": "complete"
}'
WHERE name = 'RouterAgent';

-- SynthesisAgent - combines all responses
UPDATE langgraph_agents 
SET config = '{
  "max_tokens": 8000,
  "timeout": 60,
  "temperature": 0.6,
  "response_mode": "complete"
}'
WHERE name = 'SynthesisAgent';

-- ContextManagerAgent - processes large documents
UPDATE langgraph_agents 
SET config = '{
  "max_tokens": 6000,
  "timeout": 45,
  "temperature": 0.5,
  "response_mode": "complete"
}'
WHERE name = 'ContextManagerAgent';

-- ProposalWriter - needs long responses
UPDATE langgraph_agents 
SET config = '{
  "max_tokens": 8000,
  "timeout": 60,
  "temperature": 0.6,
  "response_mode": "complete"
}'
WHERE name = 'ProposalWriter';

-- ROI_Analyst - financial calculations
UPDATE langgraph_agents 
SET config = '{
  "max_tokens": 6000,
  "timeout": 45,
  "temperature": 0.5,
  "response_mode": "complete"
}'
WHERE name = 'ROI_Analyst';

-- PreSalesArchitect - technical details
UPDATE langgraph_agents 
SET config = '{
  "max_tokens": 6000,
  "timeout": 45,
  "temperature": 0.6,
  "response_mode": "complete"
}'
WHERE name = 'PreSalesArchitect';

-- BizAnalystAgent - detailed analysis
UPDATE langgraph_agents 
SET config = '{
  "max_tokens": 6000,
  "timeout": 45,
  "temperature": 0.6,
  "response_mode": "complete"
}'
WHERE name = 'BizAnalystAgent';

-- C-Suite agents need comprehensive responses
UPDATE langgraph_agents 
SET config = '{
  "max_tokens": 6000,
  "timeout": 45,
  "temperature": 0.6,
  "response_mode": "complete"
}'
WHERE name IN ('CEO_Agent', 'CIO_Agent', 'CTO_Agent', 'Corporate_Strategist', 'ComplianceAgent', 'ExecutiveCommunicationsAgent');

-- IT operational agents
UPDATE langgraph_agents 
SET config = '{
  "max_tokens": 5000,
  "timeout": 40,
  "temperature": 0.6,
  "response_mode": "complete"
}'
WHERE name IN ('IT_Director_Agent', 'InfrastructureAgent', 'DBA', 'SecOpsAgent');

-- Insert the missing agents that the multi-agent system expects
INSERT INTO langgraph_agents (name, role, system_prompt, tools, description, is_active, config)
VALUES 
('router', 'router', 'You are the router agent that determines which agents should handle a query.', '[]', 'Routes queries to appropriate agents', true, '{"max_tokens": 1000, "timeout": 15, "temperature": 0.3}'),
('service_delivery_manager', 'service_delivery_manager', 'You are a service delivery manager responsible for operational excellence.', '[]', 'Manages service delivery and operations', true, '{"max_tokens": 8000, "timeout": 60, "temperature": 0.6}'),
('financial_analyst', 'financial_analyst', 'You are a financial analyst providing detailed financial analysis.', '[]', 'Provides financial analysis and recommendations', true, '{"max_tokens": 6000, "timeout": 45, "temperature": 0.5}'),
('technical_architect', 'technical_architect', 'You are a technical architect designing robust solutions.', '[]', 'Designs technical architectures and solutions', true, '{"max_tokens": 6000, "timeout": 45, "temperature": 0.6}'),
('sales_strategist', 'sales_strategist', 'You are a sales strategist focused on winning deals.', '[]', 'Develops sales strategies and proposals', true, '{"max_tokens": 5000, "timeout": 40, "temperature": 0.7}'),
('document_researcher', 'document_researcher', 'You are a document researcher analyzing and summarizing documents.', '[]', 'Researches and analyzes documents', true, '{"max_tokens": 6000, "timeout": 45, "temperature": 0.5}'),
('synthesizer', 'synthesizer', 'You are the synthesizer combining outputs from multiple agents.', '[]', 'Synthesizes information from multiple sources', true, '{"max_tokens": 8000, "timeout": 60, "temperature": 0.6}')
ON CONFLICT (name) DO UPDATE SET config = EXCLUDED.config;