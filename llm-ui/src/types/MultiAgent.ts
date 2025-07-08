export interface Agent {
  id: string;
  name: string;
  role: string;
  system_prompt: string;
  tools: string[];
  description: string;
  is_active: boolean;
  config: {
    model?: string;
    temperature?: number;
    max_tokens?: number;
    timeout?: number;
  };
  capabilities?: AgentCapabilities;
  created_at?: string;
  updated_at?: string;
}

export interface AgentCapabilities {
  primary_domain: string;
  skills: string[];
  expertise_areas: string[];
  tools_available: string[];
  interaction_style: string;
  complexity_level: 'basic' | 'intermediate' | 'advanced';
}

export interface MultiAgentMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  status?: 'sending' | 'sent' | 'processing' | 'complete' | 'error';
  agent_contributions?: AgentContribution[];
  execution_summary?: ExecutionSummary;
}

export interface AgentContribution {
  agent_name: string;
  agent_role: string;
  content: string;
  thinking_process?: string;
  tools_used: string[];
  execution_time: number;
  status: 'thinking' | 'executing' | 'communicating' | 'complete' | 'error';
}

export interface ExecutionSummary {
  total_agents: number;
  execution_time: number;
  collaboration_pattern: string;
  agents_involved: string[];
  tools_executed: string[];
  success: boolean;
}

export interface AgentStatus {
  name: string;
  status: 'idle' | 'selected' | 'thinking' | 'executing' | 'communicating' | 'complete' | 'error';
  progress?: number;
  current_task?: string;
  thinking_content?: string;
  tools_executing?: string[];
  error_message?: string;
  start_time?: Date;
  end_time?: Date;
}

export interface AgentCommunication {
  id: string;
  from_agent: string;
  to_agent: string;
  message: string;
  timestamp: Date;
  communication_type: 'handoff' | 'question' | 'insight' | 'tool_result';
}

export interface MultiAgentStreamEvent {
  type: 'agent_selection' | 'agent_start' | 'agent_thinking_start' | 'agent_thinking_complete' |
        'agent_token' | 'agent_tool_start' | 'agent_tool_complete' | 'agent_complete' |
        'agent_communication' | 'synthesis_start' | 'synthesis_progress' | 'final_response' | 'error';
  agent?: string;
  data?: any;
  message?: string;
  timestamp: Date;
}

export interface MultiAgentRequest {
  question: string;
  conversation_id?: string;
  selected_agents?: string[];
  max_iterations?: number;
  conversation_history?: MultiAgentMessage[];
}

export interface MultiAgentResponse {
  final_answer: string;
  agent_responses: AgentContribution[];
  conversation_id: string;
  execution_summary: ExecutionSummary;
}

export interface AgentSelection {
  recommended_agents: Agent[];
  selection_reasoning: string;
  query_analysis: {
    complexity: 'simple' | 'moderate' | 'complex';
    domains: string[];
    required_skills: string[];
    estimated_execution_time: number;
  };
}

export interface AgentPerformanceMetrics {
  agent_name: string;
  success_rate: number;
  average_response_time: number;
  total_executions: number;
  tool_usage_stats: Record<string, number>;
  domain_expertise_scores: Record<string, number>;
  recent_performance_trend: 'improving' | 'stable' | 'declining';
}

export interface CollaborationPhase {
  phase: 'ready' | 'selection' | 'execution' | 'communication' | 'synthesis' | 'complete';
  status: 'pending' | 'active' | 'complete' | 'error';
  progress: number;
  description: string;
  start_time?: Date;
  end_time?: Date;
  agents_involved: string[];
}