import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  Paper,
  Typography,
  CircularProgress,
  Alert
} from '@mui/material';
import {
  Send as SendIcon
} from '@mui/icons-material';
import { 
  MultiAgentMessage, 
  Agent, 
  AgentStatus, 
  CollaborationPhase,
  MultiAgentStreamEvent,
  AgentContribution
} from '../../types/MultiAgent';

interface MultiAgentChatProps {
  messages: MultiAgentMessage[];
  setMessages: React.Dispatch<React.SetStateAction<MultiAgentMessage[]>>;
  sessionId: string;
  loading: boolean;
  setLoading: React.Dispatch<React.SetStateAction<boolean>>;
  setAgentStatuses: React.Dispatch<React.SetStateAction<Record<string, AgentStatus>>>;
  setCollaborationPhase: React.Dispatch<React.SetStateAction<CollaborationPhase>>;
  agentStatuses: Record<string, AgentStatus>;
  setAgentStreamingContent: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  setActiveAgents: React.Dispatch<React.SetStateAction<Agent[]>>;
}

const MultiAgentChat: React.FC<MultiAgentChatProps> = ({
  messages,
  setMessages,
  sessionId,
  loading,
  setLoading,
  setAgentStatuses,
  setCollaborationPhase,
  agentStatuses,
  setAgentStreamingContent,
  setActiveAgents
}) => {
  const [input, setInput] = useState('');
  
  // Debug: Log messages received by component
  //console.log('MultiAgentChat received messages:', messages.length, messages);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage: MultiAgentMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: input,
      timestamp: new Date(),
      status: 'sending'
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    // Initialize collaboration phase
    setCollaborationPhase({
      phase: 'execution',
      status: 'active',
      progress: 0,
      description: 'Starting multi-agent collaboration',
      agents_involved: []
    });

    // Clear previous active agents
    setActiveAgents([]);
    setAgentStatuses({});
    setAgentStreamingContent({});

    try {
      const response = await fetch('/api/v1/langchain/multi-agent', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: input,
          conversation_id: sessionId,
          max_iterations: 10,
          conversation_history: messages.map(msg => ({
            role: msg.role,
            content: msg.content
          }))
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      // Mark user message as sent
      setMessages(prev => 
        prev.map(msg => 
          msg.id === userMessage.id ? { ...msg, status: 'sent' } : msg
        )
      );

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      let assistantMessage: MultiAgentMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        status: 'processing',
        agent_contributions: []
      };

      setMessages(prev => [...prev, assistantMessage]);

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;

          //console.log('Multi-agent stream line:', line);

          try {
            const data = JSON.parse(line);
            await handleStreamEvent(data, assistantMessage);
          } catch (e) {
            //console.warn('Failed to parse multi-agent stream line:', line);
          }
        }
      }

    } catch (error) {
      //console.error('Multi-agent error:', error);
      
      // Mark user message as error
      setMessages(prev => 
        prev.map(msg => 
          msg.id === userMessage.id ? { 
            ...msg, 
            status: 'error'
          } : msg
        )
      );
      
      const errorMessage: MultiAgentMessage = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date(),
        status: 'error'
      };
      setMessages(prev => [...prev, errorMessage]);

      // Reset collaboration phase
      setCollaborationPhase({
        phase: 'ready',
        status: 'pending',
        progress: 0,
        description: 'Error occurred during collaboration',
        agents_involved: []
      });

    } finally {
      setLoading(false);
    }
  };

  const handleStreamEvent = async (data: any, assistantMessage: MultiAgentMessage) => {
    switch (data.type) {
      case 'agent_selection':
        //console.log('Agent selection:', data);
        if (data.selected_agents) {
          // Create agent objects for the selected agents
          const newAgents: Agent[] = data.selected_agents.map((agentName: string) => ({
            id: agentName.toLowerCase().replace(/\s+/g, '-'),
            name: agentName,
            role: 'AI Agent', // Generic role, will be updated if more info comes
            system_prompt: '',
            tools: [],
            description: '',
            is_active: true,
            config: {}
          }));
          setActiveAgents(newAgents);
        }
        setCollaborationPhase({
          phase: 'execution',
          status: 'active',
          progress: 10,
          description: `Selected ${data.selected_agents?.length || 0} agents for collaboration`,
          agents_involved: data.selected_agents || []
        });
        break;

      case 'agent_start':
        //console.log('Agent start:', data);
        setAgentStatuses(prev => ({
          ...prev,
          [data.agent]: {
            name: data.agent,
            status: 'executing',
            current_task: 'Starting execution',
            start_time: new Date(),
            progress: 0
          }
        }));
        // Clear any previous streaming content for this agent
        setAgentStreamingContent(prev => ({
          ...prev,
          [data.agent]: ''
        }));
        break;

      case 'agent_thinking_start':
        //console.log('Agent thinking start:', data);
        setAgentStatuses(prev => ({
          ...prev,
          [data.agent]: {
            ...prev[data.agent],
            status: 'thinking',
            current_task: 'Processing and reasoning',
            progress: 25
          }
        }));
        break;

      case 'agent_thinking_complete':
        //console.log('Agent thinking complete:', data);
        setAgentStatuses(prev => ({
          ...prev,
          [data.agent]: {
            ...prev[data.agent],
            status: 'executing',
            thinking_content: data.thinking,
            progress: 50
          }
        }));
        break;

      case 'agent_token':
        // Handle streaming tokens from agents - route to individual agent windows
        //console.log('Agent token:', data.agent, data.token);
        setAgentStreamingContent(prev => ({
          ...prev,
          [data.agent]: (prev[data.agent] || '') + data.token
        }));
        break;

      case 'agent_tool_start':
        //console.log('Agent tool start:', data);
        setAgentStatuses(prev => ({
          ...prev,
          [data.agent]: {
            ...prev[data.agent],
            status: 'executing',
            current_task: 'Executing tools',
            tools_executing: data.tools || [],
            progress: 75
          }
        }));
        break;

      case 'agent_tool_complete':
        //console.log('Agent tool complete:', data);
        setAgentStatuses(prev => ({
          ...prev,
          [data.agent]: {
            ...prev[data.agent],
            tools_executing: [],
            progress: 90
          }
        }));
        break;

      case 'agent_complete':
        //console.log('Agent complete:', data);
        setAgentStatuses(prev => ({
          ...prev,
          [data.agent]: {
            ...prev[data.agent],
            status: 'complete',
            current_task: 'Completed',
            progress: 100,
            end_time: new Date()
          }
        }));

        // Add agent contribution using streaming content or provided content
        if (assistantMessage.agent_contributions) {
          const streamingContent = data.content || '';
          const contribution: AgentContribution = {
            agent_name: data.agent,
            agent_role: 'AI Agent', // Generic role
            content: streamingContent,
            thinking_process: data.thinking,
            tools_used: data.tools_used || [],
            execution_time: data.execution_time || 0,
            status: 'complete'
          };
          
          assistantMessage.agent_contributions.push(contribution);
        }
        break;

      case 'agent_communication':
        //console.log('Agent communication:', data);
        // Handle inter-agent communication
        break;

      case 'synthesis_start':
        //console.log('Synthesis start:', data);
        
        // Add Synthesizer agent to active agents
        const synthesizerAgent: Agent = {
          id: 'synthesizer',
          name: 'Synthesizer',
          role: 'Final Response Synthesizer',
          system_prompt: '',
          tools: [],
          description: 'Combines and synthesizes all agent responses into a final answer',
          is_active: true,
          config: {}
        };
        
        setActiveAgents(prev => {
          // Only add if not already present
          const exists = prev.some(agent => agent.id === 'synthesizer');
          return exists ? prev : [...prev, synthesizerAgent];
        });
        
        // Set synthesizer status
        setAgentStatuses(prev => ({
          ...prev,
          'Synthesizer': {
            name: 'Synthesizer',
            status: 'executing',
            current_task: 'Synthesizing all agent responses',
            start_time: new Date(),
            progress: 0
          }
        }));
        
        setCollaborationPhase({
          phase: 'synthesis',
          status: 'active',
          progress: 80,
          description: 'Synthesizing agent responses',
          agents_involved: Object.keys(agentStatuses)
        });
        break;

      case 'synthesis_progress':
        //console.log('Synthesis progress:', data);
        if (data.progress !== undefined && !isNaN(data.progress)) {
          setCollaborationPhase(prev => ({
            ...prev,
            progress: 80 + (data.progress * 0.2) // Scale to 80-100%
          }));
        }
        break;

      case 'final_response':
        //console.log('Final response:', data);
        
        // Update synthesizer status to complete
        setAgentStatuses(prev => ({
          ...prev,
          'Synthesizer': {
            ...prev['Synthesizer'],
            status: 'complete',
            current_task: 'Synthesis completed',
            progress: 100,
            end_time: new Date()
          }
        }));
        
        // Add final response as synthesizer contribution
        if (assistantMessage.agent_contributions) {
          const synthesizerContribution: AgentContribution = {
            agent_name: 'Synthesizer',
            agent_role: 'Final Response Synthesizer',
            content: data.response || data.final_answer || '',
            thinking_process: undefined,
            tools_used: [],
            execution_time: 0,
            status: 'complete'
          };
          
          assistantMessage.agent_contributions.push(synthesizerContribution);
        }
        
        assistantMessage.content = data.response || data.final_answer;
        assistantMessage.status = 'complete';
        assistantMessage.execution_summary = data.execution_summary;
        
        setMessages(prev => 
          prev.map(msg => 
            msg.id === assistantMessage.id ? assistantMessage : msg
          )
        );

        setCollaborationPhase({
          phase: 'complete',
          status: 'complete',
          progress: 100,
          description: 'Collaboration completed successfully',
          agents_involved: Object.keys(agentStatuses)
        });
        break;

      case 'error':
        //console.error('Multi-agent error:', data);
        setAgentStatuses(prev => {
          const newStatuses = { ...prev };
          if (data.agent) {
            newStatuses[data.agent] = {
              ...prev[data.agent],
              status: 'error',
              error_message: data.message || 'Unknown error'
            };
          }
          return newStatuses;
        });
        break;

      default:
        //console.log('Unknown multi-agent event:', data.type, data);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const canSendMessage = !loading && input.trim();

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Current Question Display */}
      {messages.length > 0 && (
        <Box sx={{ mb: 1 }}>
          <Typography variant="subtitle2" gutterBottom sx={{ fontSize: '0.875rem' }}>
            Current Question:
          </Typography>
          <Paper sx={{ p: 1.5, backgroundColor: 'primary.main', color: 'white' }}>
            <Typography variant="body1" sx={{ fontSize: '0.875rem' }}>
              {messages[messages.length - 1]?.role === 'user' 
                ? messages[messages.length - 1].content
                : messages.find(m => m.role === 'user')?.content || ''
              }
            </Typography>
          </Paper>
        </Box>
      )}

      {/* Collaboration Status */}
      {loading && (
        <Box sx={{ mb: 1 }}>
          <Alert severity="info" icon={<CircularProgress size={20} />} sx={{ py: 0.5 }}>
            <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>
              Agents are collaborating... Check individual agent windows below for progress.
            </Typography>
          </Alert>
        </Box>
      )}

      {/* Input Area */}
      <Box sx={{ mt: 'auto' }}>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
          <TextField
            fullWidth
            multiline
            maxRows={2}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask your question and agents will be automatically selected..."
            disabled={loading}
            variant="outlined"
            size="small"
          />
          <Button
            variant="contained"
            onClick={sendMessage}
            disabled={!canSendMessage}
            sx={{ minWidth: 60, height: 40 }}
          >
            {loading ? <CircularProgress size={24} color="inherit" /> : <SendIcon />}
          </Button>
        </Box>
      </Box>
    </Box>
  );
};

export default MultiAgentChat;