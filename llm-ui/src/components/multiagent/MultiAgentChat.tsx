import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  TextField,
  Button,
  Paper,
  Typography,
  CircularProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Divider
} from '@mui/material';
import {
  Send as SendIcon,
  ExpandMore as ExpandMoreIcon,
  Group as GroupIcon,
  Psychology as ThinkingIcon,
  Build as ToolIcon,
  Schedule as TimeIcon
} from '@mui/icons-material';
import { MessageContent } from '../shared/MessageContent';
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
  selectedAgents: Agent[];
  sessionId: string;
  loading: boolean;
  setLoading: React.Dispatch<React.SetStateAction<boolean>>;
  setAgentStatuses: React.Dispatch<React.SetStateAction<Record<string, AgentStatus>>>;
  setCollaborationPhase: React.Dispatch<React.SetStateAction<CollaborationPhase>>;
}

const MultiAgentChat: React.FC<MultiAgentChatProps> = ({
  messages,
  setMessages,
  selectedAgents,
  sessionId,
  loading,
  setLoading,
  setAgentStatuses,
  setCollaborationPhase
}) => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading || selectedAgents.length === 0) return;

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
      agents_involved: selectedAgents.map(a => a.name)
    });

    try {
      const response = await fetch('/api/v1/langchain/multi-agent', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: input,
          conversation_id: sessionId,
          selected_agents: selectedAgents.map(agent => agent.name),
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

          console.log('Multi-agent stream line:', line);

          try {
            const data = JSON.parse(line);
            await handleStreamEvent(data, assistantMessage);
          } catch (e) {
            console.warn('Failed to parse multi-agent stream line:', line);
          }
        }
      }

    } catch (error) {
      console.error('Multi-agent error:', error);
      
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
        phase: 'selection',
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
        console.log('Agent selection:', data);
        setCollaborationPhase({
          phase: 'execution',
          status: 'active',
          progress: 10,
          description: `Selected ${data.selected_agents?.length || 0} agents for collaboration`,
          agents_involved: data.selected_agents || []
        });
        break;

      case 'agent_start':
        console.log('Agent start:', data);
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
        break;

      case 'agent_thinking_start':
        console.log('Agent thinking start:', data);
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
        console.log('Agent thinking complete:', data);
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
        // Handle streaming tokens from agents
        if (data.agent && data.token) {
          setMessages(prev => 
            prev.map(msg => 
              msg.id === assistantMessage.id ? {
                ...msg,
                content: msg.content + data.token
              } : msg
            )
          );
        }
        break;

      case 'agent_tool_start':
        console.log('Agent tool start:', data);
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
        console.log('Agent tool complete:', data);
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
        console.log('Agent complete:', data);
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

        // Add agent contribution
        if (data.content && assistantMessage.agent_contributions) {
          const contribution: AgentContribution = {
            agent_name: data.agent,
            agent_role: selectedAgents.find(a => a.name === data.agent)?.role || 'Unknown',
            content: data.content,
            thinking_process: data.thinking,
            tools_used: data.tools_used || [],
            execution_time: data.execution_time || 0,
            status: 'complete'
          };
          
          assistantMessage.agent_contributions.push(contribution);
        }
        break;

      case 'agent_communication':
        console.log('Agent communication:', data);
        // Handle inter-agent communication
        break;

      case 'synthesis_start':
        console.log('Synthesis start:', data);
        setCollaborationPhase({
          phase: 'synthesis',
          status: 'active',
          progress: 80,
          description: 'Synthesizing agent responses',
          agents_involved: selectedAgents.map(a => a.name)
        });
        break;

      case 'synthesis_progress':
        console.log('Synthesis progress:', data);
        if (data.progress) {
          setCollaborationPhase(prev => ({
            ...prev,
            progress: 80 + (data.progress * 0.2) // Scale to 80-100%
          }));
        }
        break;

      case 'final_response':
        console.log('Final response:', data);
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
          agents_involved: selectedAgents.map(a => a.name)
        });
        break;

      case 'error':
        console.error('Multi-agent error:', data);
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
        console.log('Unknown multi-agent event:', data.type, data);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const canSendMessage = !loading && input.trim() && selectedAgents.length > 0;

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Messages Area */}
      <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
        {messages.length === 0 ? (
          <Alert severity="info">
            <Typography variant="body2">
              Select agents and start a multi-agent collaboration!
            </Typography>
          </Alert>
        ) : (
          messages.map((message) => (
            <Box key={message.id} sx={{ mb: 2 }}>
              <Paper
                sx={{
                  p: 2,
                  backgroundColor: message.role === 'user' ? 'primary.main' : 'background.paper',
                  color: message.role === 'user' ? 'white' : 'text.primary',
                  border: message.role === 'assistant' ? 1 : 0,
                  borderColor: 'divider'
                }}
                elevation={2}
              >
                <MessageContent content={message.content} />

                {/* Agent Contributions */}
                {message.agent_contributions && message.agent_contributions.length > 0 && (
                  <Box sx={{ mt: 2 }}>
                    <Divider sx={{ mb: 2 }} />
                    <Typography variant="subtitle2" gutterBottom>
                      <GroupIcon fontSize="small" sx={{ mr: 0.5 }} />
                      Agent Contributions ({message.agent_contributions.length})
                    </Typography>
                    
                    {message.agent_contributions.map((contribution, index) => (
                      <Accordion key={index} sx={{ mb: 1 }}>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="body2" fontWeight="bold">
                              {contribution.agent_name}
                            </Typography>
                            <Chip
                              label={contribution.agent_role}
                              size="small"
                              variant="outlined"
                            />
                            {contribution.tools_used.length > 0 && (
                              <Chip
                                label={`${contribution.tools_used.length} tools`}
                                size="small"
                                icon={<ToolIcon />}
                                variant="outlined"
                              />
                            )}
                            <Chip
                              label={`${contribution.execution_time}s`}
                              size="small"
                              icon={<TimeIcon />}
                              variant="outlined"
                            />
                          </Box>
                        </AccordionSummary>
                        <AccordionDetails>
                          <MessageContent content={contribution.content} />
                          
                          {contribution.thinking_process && (
                            <Accordion sx={{ mt: 1 }}>
                              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                  <ThinkingIcon fontSize="small" color="primary" />
                                  <Typography variant="body2">
                                    Reasoning Process
                                  </Typography>
                                </Box>
                              </AccordionSummary>
                              <AccordionDetails>
                                <MessageContent content={contribution.thinking_process} />
                              </AccordionDetails>
                            </Accordion>
                          )}
                        </AccordionDetails>
                      </Accordion>
                    ))}
                  </Box>
                )}

                {/* Execution Summary */}
                {message.execution_summary && (
                  <Box sx={{ mt: 2 }}>
                    <Divider sx={{ mb: 1 }} />
                    <Typography variant="caption" color="text.secondary">
                      Collaboration completed in {message.execution_summary.execution_time}s using{' '}
                      {message.execution_summary.total_agents} agents
                    </Typography>
                  </Box>
                )}
              </Paper>
              
              <Typography 
                variant="caption" 
                color="text.secondary" 
                sx={{ mt: 0.5, px: 1, display: 'block' }}
              >
                {message.timestamp.toLocaleTimeString()}
              </Typography>
            </Box>
          ))
        )}
        
        {loading && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 2 }}>
            <CircularProgress size={20} />
            <Typography variant="body2" color="text.secondary">
              Agents are collaborating...
            </Typography>
          </Box>
        )}
        
        <div ref={messagesEndRef} />
      </Box>

      {/* Input Area */}
      <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
        {selectedAgents.length === 0 && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            Please select at least one agent to start collaboration
          </Alert>
        )}
        
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
          <TextField
            fullWidth
            multiline
            maxRows={4}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={selectedAgents.length > 0 
              ? `Ask ${selectedAgents.map(a => a.name).join(', ')}...`
              : 'Select agents first...'
            }
            disabled={loading || selectedAgents.length === 0}
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