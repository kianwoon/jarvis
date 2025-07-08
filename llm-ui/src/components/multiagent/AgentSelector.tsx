import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  Checkbox,
  FormControlLabel,
  Divider,
  IconButton,
  Tooltip,
  Alert,
  CircularProgress,
  Switch
} from '@mui/material';
import {
  Psychology as BrainIcon,
  Build as ToolIcon,
  Speed as PerformanceIcon,
  AutoAwesome as SmartIcon,
  Person as PersonIcon,
  CheckCircle as SelectedIcon
} from '@mui/icons-material';
import { Agent, CollaborationPhase, AgentPerformanceMetrics } from '../../types/MultiAgent';

interface AgentSelectorProps {
  agents: Agent[];
  selectedAgents: Agent[];
  onAgentSelection: (agents: Agent[]) => void;
  collaborationPhase: CollaborationPhase;
}

const AgentSelector: React.FC<AgentSelectorProps> = ({
  agents,
  selectedAgents,
  onAgentSelection,
  collaborationPhase
}) => {
  const [smartSelection, setSmartSelection] = useState(true);
  const [performanceMetrics, setPerformanceMetrics] = useState<Record<string, AgentPerformanceMetrics>>({});
  const [loadingMetrics, setLoadingMetrics] = useState(false);
  const [recommendedAgents, setRecommendedAgents] = useState<Agent[]>([]);

  // Load agent performance metrics
  useEffect(() => {
    if (agents.length > 0) {
      loadPerformanceMetrics();
    }
  }, [agents]);

  const loadPerformanceMetrics = async () => {
    setLoadingMetrics(true);
    try {
      // Mock performance metrics - in real implementation, fetch from backend
      const mockMetrics: Record<string, AgentPerformanceMetrics> = {};
      agents.forEach(agent => {
        mockMetrics[agent.name] = {
          agent_name: agent.name,
          success_rate: Math.random() * 0.3 + 0.7, // 70-100%
          average_response_time: Math.random() * 20 + 10, // 10-30 seconds
          total_executions: Math.floor(Math.random() * 100) + 50,
          tool_usage_stats: {},
          domain_expertise_scores: {},
          recent_performance_trend: ['improving', 'stable', 'declining'][Math.floor(Math.random() * 3)] as any
        };
      });
      setPerformanceMetrics(mockMetrics);
    } catch (error) {
      console.error('Error loading performance metrics:', error);
    } finally {
      setLoadingMetrics(false);
    }
  };

  // Get recommended agents based on smart selection
  useEffect(() => {
    if (smartSelection && agents.length > 0) {
      // Mock smart recommendations - in real implementation, use AI analysis
      const recommended = agents
        .filter(agent => agent.is_active)
        .sort((a, b) => {
          const aMetrics = performanceMetrics[a.name];
          const bMetrics = performanceMetrics[b.name];
          if (!aMetrics || !bMetrics) return 0;
          return bMetrics.success_rate - aMetrics.success_rate;
        })
        .slice(0, 3);
      setRecommendedAgents(recommended);
    }
  }, [smartSelection, agents, performanceMetrics]);

  const handleAgentToggle = (agent: Agent, selected: boolean) => {
    let newSelection: Agent[];
    if (selected) {
      newSelection = [...selectedAgents, agent];
    } else {
      newSelection = selectedAgents.filter(a => a.id !== agent.id);
    }
    onAgentSelection(newSelection);
  };

  const handleSmartSelect = () => {
    onAgentSelection(recommendedAgents);
  };

  const handleSelectAll = () => {
    const activeAgents = agents.filter(agent => agent.is_active);
    onAgentSelection(activeAgents);
  };

  const handleClearSelection = () => {
    onAgentSelection([]);
  };

  const isAgentSelected = (agent: Agent) => {
    return selectedAgents.some(selected => selected.id === agent.id);
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'basic': return 'success';
      case 'intermediate': return 'warning';
      case 'advanced': return 'error';
      default: return 'default';
    }
  };

  const getDomainIcon = (domain: string) => {
    switch (domain) {
      case 'technical': return <BrainIcon />;
      case 'business': return <PersonIcon />;
      case 'research': return <SmartIcon />;
      default: return <PersonIcon />;
    }
  };

  const activeAgents = agents.filter(agent => agent.is_active);

  return (
    <Box>
      {/* Selection Controls */}
      <Box sx={{ mb: 2 }}>
        <FormControlLabel
          control={
            <Switch
              checked={smartSelection}
              onChange={(e) => setSmartSelection(e.target.checked)}
              size="small"
            />
          }
          label="Smart Selection"
        />
        
        <Box sx={{ mt: 1, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          {smartSelection && (
            <Button
              size="small"
              variant="outlined"
              onClick={handleSmartSelect}
              startIcon={<SmartIcon />}
              disabled={recommendedAgents.length === 0}
            >
              Use Recommended
            </Button>
          )}
          <Button
            size="small"
            variant="outlined"
            onClick={handleSelectAll}
          >
            Select All
          </Button>
          <Button
            size="small"
            variant="outlined"
            onClick={handleClearSelection}
            disabled={selectedAgents.length === 0}
          >
            Clear
          </Button>
        </Box>
      </Box>

      {/* Selection Summary */}
      {selectedAgents.length > 0 && (
        <Alert severity="info" sx={{ mb: 2 }}>
          <Typography variant="body2">
            Selected {selectedAgents.length} agent{selectedAgents.length > 1 ? 's' : ''}: {' '}
            {selectedAgents.map(agent => agent.name).join(', ')}
          </Typography>
        </Alert>
      )}

      {/* Smart Recommendations */}
      {smartSelection && recommendedAgents.length > 0 && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            <SmartIcon fontSize="small" sx={{ mr: 0.5 }} />
            Recommended Agents
          </Typography>
          <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
            {recommendedAgents.map(agent => (
              <Chip
                key={agent.id}
                label={agent.name}
                size="small"
                color="primary"
                variant={isAgentSelected(agent) ? "filled" : "outlined"}
              />
            ))}
          </Box>
        </Box>
      )}

      <Divider sx={{ my: 2 }} />

      {/* Available Agents */}
      <Typography variant="subtitle2" gutterBottom>
        Available Agents ({activeAgents.length})
      </Typography>

      {loadingMetrics && (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
          <CircularProgress size={20} />
        </Box>
      )}

      <Box sx={{ maxHeight: 'calc(100vh - 400px)', overflow: 'auto' }}>
        {activeAgents.map((agent) => {
          const isSelected = isAgentSelected(agent);
          const metrics = performanceMetrics[agent.name];
          const capabilities = agent.capabilities;

          return (
            <Card
              key={agent.id}
              sx={{
                mb: 1,
                border: isSelected ? 2 : 1,
                borderColor: isSelected ? 'primary.main' : 'divider',
                cursor: 'pointer',
                '&:hover': {
                  borderColor: 'primary.main',
                  boxShadow: 2
                }
              }}
              onClick={() => handleAgentToggle(agent, !isSelected)}
            >
              <CardContent sx={{ pb: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Checkbox
                    checked={isSelected}
                    onChange={(e) => handleAgentToggle(agent, e.target.checked)}
                    onClick={(e) => e.stopPropagation()}
                    size="small"
                  />
                  
                  <Box sx={{ ml: 1, flex: 1 }}>
                    <Typography variant="subtitle2" noWrap>
                      {agent.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" noWrap>
                      {agent.role}
                    </Typography>
                  </Box>

                  {isSelected && (
                    <SelectedIcon color="primary" fontSize="small" />
                  )}
                </Box>

                {/* Agent Capabilities */}
                {capabilities && (
                  <Box sx={{ mb: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
                      {getDomainIcon(capabilities.primary_domain)}
                      <Chip
                        label={capabilities.primary_domain}
                        size="small"
                        variant="outlined"
                      />
                      <Chip
                        label={capabilities.complexity_level}
                        size="small"
                        color={getComplexityColor(capabilities.complexity_level) as any}
                        variant="outlined"
                      />
                    </Box>
                    
                    {capabilities.skills.length > 0 && (
                      <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                        {capabilities.skills.slice(0, 2).map((skill) => (
                          <Chip
                            key={skill}
                            label={skill}
                            size="small"
                            variant="outlined"
                            sx={{ fontSize: '0.7rem', height: 18 }}
                          />
                        ))}
                        {capabilities.skills.length > 2 && (
                          <Typography variant="caption" color="text.secondary">
                            +{capabilities.skills.length - 2} more
                          </Typography>
                        )}
                      </Box>
                    )}
                  </Box>
                )}

                {/* Tools */}
                {agent.tools.length > 0 && (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 1 }}>
                    <ToolIcon fontSize="small" color="action" />
                    <Typography variant="caption" color="text.secondary">
                      {agent.tools.length} tool{agent.tools.length > 1 ? 's' : ''}
                    </Typography>
                  </Box>
                )}

                {/* Performance Metrics */}
                {metrics && (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Tooltip title={`Success Rate: ${(metrics.success_rate * 100).toFixed(1)}%`}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <PerformanceIcon fontSize="small" color="action" />
                        <Typography variant="caption" color="text.secondary">
                          {(metrics.success_rate * 100).toFixed(0)}%
                        </Typography>
                      </Box>
                    </Tooltip>
                    
                    <Typography variant="caption" color="text.secondary">
                      ~{metrics.average_response_time.toFixed(0)}s
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          );
        })}
      </Box>

      {activeAgents.length === 0 && (
        <Alert severity="warning">
          No active agents available. Please check your agent configuration.
        </Alert>
      )}
    </Box>
  );
};

export default AgentSelector;