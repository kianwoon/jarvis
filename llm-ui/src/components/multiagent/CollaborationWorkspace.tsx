import React from 'react';
import {
  Box,
  Typography,
  Paper,
  LinearProgress,
  Chip,
  Card,
  CardContent,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Alert
} from '@mui/material';
import {
  Timeline,
  TimelineItem,
  TimelineSeparator,
  TimelineConnector,
  TimelineContent,
  TimelineDot
} from '@mui/lab';
import {
  Psychology as ThinkingIcon,
  Build as ToolIcon,
  Chat as CommunicationIcon,
  CheckCircle as CompleteIcon,
  Error as ErrorIcon,
  PlayArrow as ExecutingIcon,
  Schedule as PendingIcon,
  ExpandMore as ExpandMoreIcon,
  Group as GroupIcon
} from '@mui/icons-material';
import { Agent, AgentStatus, CollaborationPhase } from '../../types/MultiAgent';

interface CollaborationWorkspaceProps {
  agentStatuses: Record<string, AgentStatus>;
  collaborationPhase: CollaborationPhase;
  selectedAgents: Agent[];
}

const CollaborationWorkspace: React.FC<CollaborationWorkspaceProps> = ({
  agentStatuses,
  collaborationPhase,
  selectedAgents
}) => {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'thinking':
        return <ThinkingIcon color="primary" />;
      case 'executing':
        return <ExecutingIcon color="warning" />;
      case 'communicating':
        return <CommunicationIcon color="info" />;
      case 'complete':
        return <CompleteIcon color="success" />;
      case 'error':
        return <ErrorIcon color="error" />;
      case 'selected':
        return <PendingIcon color="action" />;
      default:
        return <PendingIcon color="action" />;
    }
  };

  const getStatusColor = (status: string): 'primary' | 'secondary' | 'error' | 'warning' | 'info' | 'success' | undefined => {
    switch (status) {
      case 'thinking':
        return 'primary';
      case 'executing':
        return 'warning';
      case 'communicating':
        return 'info';
      case 'complete':
        return 'success';
      case 'error':
        return 'error';
      case 'selected':
        return undefined;
      case 'idle':
        return undefined;
      default:
        return undefined;
    }
  };

  const getPhaseIcon = (phase: string) => {
    switch (phase) {
      case 'selection':
        return <GroupIcon />;
      case 'execution':
        return <ExecutingIcon />;
      case 'communication':
        return <CommunicationIcon />;
      case 'synthesis':
        return <ThinkingIcon />;
      case 'complete':
        return <CompleteIcon />;
      default:
        return <PendingIcon />;
    }
  };

  const formatDuration = (startTime?: Date, endTime?: Date) => {
    if (!startTime) return '';
    const end = endTime || new Date();
    const duration = Math.floor((end.getTime() - startTime.getTime()) / 1000);
    return `${duration}s`;
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Collaboration Phase Status */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          {getPhaseIcon(collaborationPhase.phase)}
          <Typography variant="subtitle1" fontWeight="bold">
            {collaborationPhase.phase.charAt(0).toUpperCase() + collaborationPhase.phase.slice(1)} Phase
          </Typography>
          <Chip
            label={collaborationPhase.status}
            size="small"
            color={getStatusColor(collaborationPhase.status) || 'default'}
            variant="outlined"
          />
        </Box>
        
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          {collaborationPhase.description}
        </Typography>
        
        <LinearProgress
          variant="determinate"
          value={collaborationPhase.progress}
          sx={{ mb: 1 }}
        />
        
        <Typography variant="caption" color="text.secondary">
          Progress: {collaborationPhase.progress}%
        </Typography>
      </Box>

      {/* Agent Timeline */}
      <Box sx={{ flex: 1, overflow: 'auto', p: 1 }}>
        {selectedAgents.length === 0 ? (
          <Alert severity="info" sx={{ m: 2 }}>
            Select agents from the left panel to begin collaboration
          </Alert>
        ) : (
          <Timeline>
            {selectedAgents.map((agent, index) => {
              const status = agentStatuses[agent.name];
              const isLast = index === selectedAgents.length - 1;

              return (
                <TimelineItem key={agent.id}>
                  <TimelineSeparator>
                    <TimelineDot color={getStatusColor(status?.status || 'idle')}>
                      {getStatusIcon(status?.status || 'idle')}
                    </TimelineDot>
                    {!isLast && <TimelineConnector />}
                  </TimelineSeparator>
                  
                  <TimelineContent sx={{ pb: 3 }}>
                    <Card variant="outlined">
                      <CardContent sx={{ pb: 2 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                          <Typography variant="subtitle2" fontWeight="bold">
                            {agent.name}
                          </Typography>
                          <Chip
                            label={status?.status || 'idle'}
                            size="small"
                            color={getStatusColor(status?.status || 'idle') || 'default'}
                            variant="filled"
                          />
                        </Box>
                        
                        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                          {agent.role}
                        </Typography>

                        {status?.current_task && (
                          <Typography variant="body2" sx={{ mb: 1 }}>
                            {status.current_task}
                          </Typography>
                        )}

                        {status?.progress !== undefined && status.progress > 0 && (
                          <Box sx={{ mb: 1 }}>
                            <LinearProgress
                              variant="determinate"
                              value={status.progress}
                              size="small"
                              sx={{ mb: 0.5 }}
                            />
                            <Typography variant="caption" color="text.secondary">
                              {status.progress}% complete
                            </Typography>
                          </Box>
                        )}

                        {/* Tools being executed */}
                        {status?.tools_executing && status.tools_executing.length > 0 && (
                          <Box sx={{ mb: 1 }}>
                            <Typography variant="caption" color="text.secondary" display="block">
                              Executing tools:
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                              {status.tools_executing.map((tool) => (
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

                        {/* Thinking process */}
                        {status?.thinking_content && (
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
                              <Typography variant="body2" color="text.secondary">
                                {status.thinking_content}
                              </Typography>
                            </AccordionDetails>
                          </Accordion>
                        )}

                        {/* Error message */}
                        {status?.error_message && (
                          <Alert severity="error" sx={{ mt: 1 }}>
                            <Typography variant="body2">
                              {status.error_message}
                            </Typography>
                          </Alert>
                        )}

                        {/* Execution time */}
                        {status?.start_time && (
                          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                            Duration: {formatDuration(status.start_time, status.end_time)}
                          </Typography>
                        )}
                      </CardContent>
                    </Card>
                  </TimelineContent>
                </TimelineItem>
              );
            })}
          </Timeline>
        )}
      </Box>

      {/* Agent Tools Summary */}
      {selectedAgents.length > 0 && (
        <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
          <Typography variant="subtitle2" gutterBottom>
            Available Tools
          </Typography>
          <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
            {Array.from(new Set(selectedAgents.flatMap(agent => agent.tools))).map((tool) => (
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
    </Box>
  );
};

export default CollaborationWorkspace;