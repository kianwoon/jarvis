import React from 'react';
import {
  Box,
  Typography,
  Alert,
  Chip,
  CircularProgress,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  Psychology as ThinkingIcon,
  Build as ToolIcon,
  CheckCircle as CompleteIcon,
  Error as ErrorIcon,
  Schedule as PendingIcon,
  ExpandMore as ExpandMoreIcon
} from '@mui/icons-material';
import { MessageContent } from '../shared/MessageContent';
import { Agent, AgentStatus, MultiAgentMessage } from '../../types/MultiAgent';

interface AgentResponseWindowProps {
  agent: Agent;
  agentStatus?: AgentStatus;
  messages: MultiAgentMessage[];
  streamingContent?: string;
}

const AgentResponseWindow: React.FC<AgentResponseWindowProps> = ({
  agent,
  agentStatus,
  messages,
  streamingContent = ''
}) => {
  // Get the latest agent contribution for this agent
  const getAgentResponse = () => {
    // Look through all messages to find contributions from this agent
    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i];
      if (message.agent_contributions) {
        const contribution = message.agent_contributions.find(
          contrib => contrib.agent_name === agent.name
        );
        if (contribution) {
          return contribution;
        }
      }
    }
    return null;
  };

  const agentContribution = getAgentResponse();

  const getStatusIcon = (status?: string) => {
    switch (status) {
      case 'thinking':
        return <ThinkingIcon color="primary" />;
      case 'executing':
        return <CircularProgress size={20} />;
      case 'complete':
        return <CompleteIcon color="success" />;
      case 'error':
        return <ErrorIcon color="error" />;
      default:
        return <PendingIcon color="action" />;
    }
  };

  const getStatusColor = (status?: string) => {
    switch (status) {
      case 'thinking':
        return 'primary';
      case 'executing':
        return 'warning';
      case 'complete':
        return 'success';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Agent Status */}
      <Box sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          {getStatusIcon(agentStatus?.status)}
          <Chip
            label={agentStatus?.status || 'idle'}
            size="small"
            color={getStatusColor(agentStatus?.status) as any}
            variant="outlined"
          />
        </Box>

        {agentStatus?.current_task && (
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            {agentStatus.current_task}
          </Typography>
        )}

        {agentStatus?.progress !== undefined && agentStatus.progress > 0 && (
          <Box sx={{ mb: 1 }}>
            <LinearProgress
              variant="determinate"
              value={agentStatus.progress}
              sx={{ mb: 0.5 }}
            />
            <Typography variant="caption" color="text.secondary">
              {agentStatus.progress}% complete
            </Typography>
          </Box>
        )}

        {/* Tools being executed */}
        {agentStatus?.tools_executing && agentStatus.tools_executing.length > 0 && (
          <Box sx={{ mb: 1 }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Executing tools:
            </Typography>
            <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
              {agentStatus.tools_executing.map((tool) => (
                <Chip
                  key={tool}
                  label={tool}
                  size="small"
                  icon={<ToolIcon />}
                  variant="outlined"
                />
              ))}
            </Box>
          </Box>
        )}

        {/* Error message */}
        {agentStatus?.error_message && (
          <Alert severity="error" sx={{ mb: 1 }}>
            <Typography variant="body2">
              {agentStatus.error_message}
            </Typography>
          </Alert>
        )}
      </Box>

      {/* Agent Response */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {!agentContribution && !streamingContent && agentStatus?.status !== 'complete' && (
          <Alert severity="info">
            <Typography variant="body2">
              Waiting for agent response...
            </Typography>
          </Alert>
        )}

        {/* Show streaming content when agent is active but not complete */}
        {streamingContent && !agentContribution && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Response (streaming...):
            </Typography>
            <MessageContent content={streamingContent} />
          </Box>
        )}

        {agentContribution && (
          <Box>
            {/* Thinking Process */}
            {agentContribution.thinking_process && (
              <Accordion sx={{ mb: 2 }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <ThinkingIcon fontSize="small" color="primary" />
                    <Typography variant="body2" color="primary">
                      Reasoning Process
                    </Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <MessageContent content={agentContribution.thinking_process} />
                </AccordionDetails>
              </Accordion>
            )}

            {/* Main Response */}
            {agentContribution.content && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Response:
                </Typography>
                <MessageContent content={agentContribution.content} />
              </Box>
            )}

            {/* Tools Used */}
            {agentContribution.tools_used && agentContribution.tools_used.length > 0 && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Tools Used:
                </Typography>
                <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                  {agentContribution.tools_used.map((tool) => (
                    <Chip
                      key={tool}
                      label={tool}
                      size="small"
                      icon={<ToolIcon />}
                      variant="outlined"
                      color="primary"
                    />
                  ))}
                </Box>
              </Box>
            )}

            {/* Execution Time */}
            {agentContribution.execution_time > 0 && (
              <Typography variant="caption" color="text.secondary">
                Execution time: {agentContribution.execution_time}s
              </Typography>
            )}
          </Box>
        )}

        {/* Show current thinking content while agent is thinking */}
        {agentStatus?.thinking_content && agentStatus.status === 'thinking' && !agentContribution?.thinking_process && (
          <Box sx={{ mb: 2 }}>
            <Accordion expanded>
              <AccordionSummary>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <ThinkingIcon fontSize="small" color="primary" />
                  <Typography variant="body2" color="primary">
                    Currently Thinking...
                  </Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <MessageContent content={agentStatus.thinking_content} />
              </AccordionDetails>
            </Accordion>
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default AgentResponseWindow;