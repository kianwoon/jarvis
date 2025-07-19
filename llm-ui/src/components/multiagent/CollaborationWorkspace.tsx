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
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 1 }}>
      {getPhaseIcon(collaborationPhase.phase)}
      <Typography variant="subtitle1" fontWeight="bold" sx={{ fontSize: '0.95rem' }}>
        {collaborationPhase.phase.charAt(0).toUpperCase() + collaborationPhase.phase.slice(1)} Phase
      </Typography>
      <Chip
        label={collaborationPhase.status}
        size="small"
        color={getStatusColor(collaborationPhase.status) || 'default'}
        variant="outlined"
      />
      <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.8rem' }}>
        {collaborationPhase.description}
      </Typography>
      <Box sx={{ flex: 1 }} />
      <LinearProgress
        variant="determinate"
        value={collaborationPhase.progress || 0}
        sx={{ width: 100, height: 4 }}
      />
      <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
        Progress: {collaborationPhase.progress || 0}%
      </Typography>
    </Box>
  );
};

export default CollaborationWorkspace;