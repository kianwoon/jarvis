import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Handle, Position } from 'reactflow';
import '../../styles/workflow-animations.css';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box, 
  TextField,
  FormControl,
  InputLabel,
  FormControlLabel,
  Switch,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Alert,
  Tooltip,
  IconButton,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Paper,
  LinearProgress,
  Divider
} from '@mui/material';
import { 
  AccountTree as ParallelIcon,
  ExpandMore as ExpandMoreIcon,
  Settings as SettingsIcon,
  Info as InfoIcon,
  PlayArrow as PlayIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  Schedule as ScheduleIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  Add as AddIcon,
  Delete as DeleteIcon
} from '@mui/icons-material';
import PortalSelect from './PortalSelect';

interface ParallelBranch {
  id: string;
  name: string;
  nodeId?: string;
  status?: 'pending' | 'running' | 'completed' | 'failed';
  result?: any;
  error?: string;
  executionTime?: number;
}

interface ParallelNodeProps {
  data: {
    label?: string;
    // Backend compatible properties
    max_parallel?: number;
    wait_for_all?: boolean;
    combine_strategy?: 'merge' | 'concat' | 'summary' | 'best';
    // Additional frontend properties
    executionMode?: 'all' | 'race' | 'some';
    maxConcurrency?: number;
    timeout?: number;
    continueOnError?: boolean;
    resultStrategy?: 'array' | 'object' | 'first' | 'fastest';
    branches?: ParallelBranch[];
    connectedNodes?: string[]; // From edges
    executionData?: {
      status?: 'idle' | 'running' | 'success' | 'error' | 'partial';
      runningBranches?: string[];
      completedBranches?: string[];
      failedBranches?: string[];
      results?: any;
      totalTime?: number;
      error?: string;
    };
  };
  id: string;
  updateNodeData?: (nodeId: string, newData: any) => void;
  showIO?: boolean;
}

const ParallelNode: React.FC<ParallelNodeProps> = ({ data, id, updateNodeData, showIO = true }) => {
  const [configExpanded, setConfigExpanded] = useState(true);
  const [statusExpanded, setStatusExpanded] = useState(false);
  const [advancedExpanded, setAdvancedExpanded] = useState(false);
  
  const [label, setLabel] = useState(data.label || 'Parallel Execution');
  // Backend compatible states
  const [maxParallel, setMaxParallel] = useState(data.max_parallel || data.maxConcurrency || 3);
  const [waitForAll, setWaitForAll] = useState(data.wait_for_all ?? true);
  const [combineStrategy, setCombineStrategy] = useState(data.combine_strategy || 'merge');
  // Additional frontend states
  const [timeout, setTimeout] = useState(data.timeout || 60000);
  const [continueOnError, setContinueOnError] = useState(data.continueOnError ?? true);
  const [branches, setBranches] = useState<ParallelBranch[]>(data.branches || [
    { id: 'branch_1', name: 'Branch 1', status: 'pending' },
    { id: 'branch_2', name: 'Branch 2', status: 'pending' },
    { id: 'branch_3', name: 'Branch 3', status: 'pending' }
  ]);
  
  // Internal execution data state (like AgentNode)
  const [executionData, setExecutionData] = useState(data.executionData || { status: 'idle' });
  
  // Status info for CSS class mapping (like AgentNode)
  const statusInfo = useMemo(() => {
    const currentStatus = executionData?.status || 'idle';
    switch (currentStatus) {
      case 'running':
        return {
          nodeClass: 'workflow-node--running-control',
          color: '#ff9800'
        };
      case 'success':
        return {
          nodeClass: 'workflow-node--success',
          color: '#4caf50'
        };
      case 'error':
        return {
          nodeClass: 'workflow-node--error',
          color: '#f44336'
        };
      default:
        return {
          nodeClass: 'workflow-node--idle',
          color: '#00bcd4'
        };
    }
  }, [executionData?.status]);
  

  // Sync branches with connected nodes
  useEffect(() => {
    if (data.connectedNodes && data.connectedNodes.length > 0) {
      const existingBranchesMap = new Map(branches.map(branch => [branch.nodeId, branch]));
      
      const newBranches: ParallelBranch[] = data.connectedNodes.map((nodeId, index) => {
        const existing = existingBranchesMap.get(nodeId);
        if (existing) {
          return existing;
        } else {
          return {
            id: `branch_${Date.now()}_${index}`,
            name: `Branch ${index + 1}`,
            nodeId: nodeId,
            status: 'pending'
          };
        }
      });

      // Filter out branches that are no longer connected
      const connectedSet = new Set(data.connectedNodes);
      const filteredBranches = newBranches.filter(branch => 
        branch.nodeId && connectedSet.has(branch.nodeId)
      );
      
      if (JSON.stringify(filteredBranches) !== JSON.stringify(branches)) {
        setBranches(filteredBranches);
      }
    }
  }, [data.connectedNodes]);

  // Sync execution data from props (like AgentNode) 
  useEffect(() => {
    if (data.executionData) {
      setExecutionData(data.executionData);
    }
  }, [data.executionData]);

  // Only update parent when user changes values, not on every render
  const updateParentData = useCallback((updates: any) => {
    if (updateNodeData) {
      updateNodeData(id, { ...data, ...updates });
    }
  }, [updateNodeData, id, data]);

  const getStatusColor = () => {
    switch (executionData?.status) {
      case 'running': return '#ff9800';
      case 'success': return '#4caf50';
      case 'error': return '#f44336';
      case 'partial': return '#ff9800';
      default: return '#00bcd4';
    }
  };

  const hasExecutionData = executionData && executionData.status !== 'idle';
  const isRunning = executionData?.status === 'running';

  const getBranchStatus = (branchId: string) => {
    const branch = branches.find(b => b.id === branchId);
    if (!branch) return 'pending';
    
    if (executionData?.completedBranches?.includes(branchId)) return 'completed';
    if (executionData?.failedBranches?.includes(branchId)) return 'failed';
    if (executionData?.runningBranches?.includes(branchId)) return 'running';
    
    return branch.status || 'pending';
  };

  const getBranchIcon = (status: string) => {
    switch (status) {
      case 'running': return <PlayIcon sx={{ fontSize: 16, color: '#ff9800' }} />;
      case 'completed': return <CheckIcon sx={{ fontSize: 16, color: '#4caf50' }} />;
      case 'failed': return <ErrorIcon sx={{ fontSize: 16, color: '#f44336' }} />;
      default: return <ScheduleIcon sx={{ fontSize: 16, color: '#9e9e9e' }} />;
    }
  };

  const calculateProgress = () => {
    if (!executionData || branches.length === 0) return 0;
    
    const completed = executionData.completedBranches?.length || 0;
    const failed = executionData.failedBranches?.length || 0;
    const total = branches.length;
    
    return ((completed + failed) / total) * 100;
  };

  const updateBranch = (branchId: string, field: keyof ParallelBranch, value: any) => {
    setBranches(branches.map(branch => 
      branch.id === branchId ? { ...branch, [field]: value } : branch
    ));
  };

  const addBranch = () => {
    const newBranch: ParallelBranch = {
      id: `branch_${Date.now()}`,
      name: `Branch ${branches.length + 1}`,
      status: 'pending'
    };
    setBranches([...branches, newBranch]);
  };

  const deleteBranch = (branchId: string) => {
    setBranches(branches.filter(branch => branch.id !== branchId));
  };

  return (
    <Card 
      className={statusInfo.nodeClass}
      sx={{ 
        minWidth: 400,
        maxWidth: 500,
        border: `2px solid ${getStatusColor()}`,
        borderRadius: 2,
        bgcolor: theme => theme.palette.mode === 'dark' ? 'rgba(0, 188, 212, 0.08)' : 'rgba(0, 188, 212, 0.05)',
        position: 'relative',
        overflow: 'visible',
        '&:hover': {
          boxShadow: theme => `0 0 20px ${theme.palette.mode === 'dark' ? 'rgba(0, 188, 212, 0.3)' : 'rgba(0, 188, 212, 0.2)'}`,
        }
      }}
    >
      <CardContent sx={{ pb: 1 }}>
        {/* Header */}
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          <ParallelIcon sx={{ color: getStatusColor() }} />
          <TextField
            value={label}
            onChange={(e) => {
              const newLabel = e.target.value;
              setLabel(newLabel);
              updateParentData({ label: newLabel });
            }}
            variant="standard"
            fullWidth
            sx={{
              '& .MuiInput-root': {
                fontSize: '1.1rem',
                fontWeight: 600,
              }
            }}
          />
          <Tooltip title="Execute multiple branches in parallel">
            <InfoIcon sx={{ fontSize: 18, color: 'text.secondary' }} />
          </Tooltip>
        </Box>

        {/* Execution Status */}
        {hasExecutionData && (
          <Box mb={2}>
            <Alert 
              severity={executionData?.status === 'error' ? 'error' : 
                       executionData?.status === 'partial' ? 'warning' : 'success'} 
              sx={{ mb: 1 }}
            >
              <Typography variant="caption" fontWeight={500}>
                Status: {executionData?.status}
                {executionData?.totalTime && ` (${executionData.totalTime}ms)`}
              </Typography>
            </Alert>
            
            {isRunning && (
              <LinearProgress 
                variant="determinate" 
                value={calculateProgress()} 
                sx={{ mb: 1, height: 6, borderRadius: 1 }}
              />
            )}
          </Box>
        )}

        {/* Combine Strategy */}
        <Box sx={{ mb: 2 }}>
          <PortalSelect
            value={combineStrategy}
            onChange={(value) => {
              const newStrategy = value as 'merge' | 'concat' | 'summary' | 'best';
              setCombineStrategy(newStrategy);
              updateParentData({ combine_strategy: newStrategy });
            }}
            label="Combine Strategy"
            options={[
              { value: 'merge', label: 'Merge Results' },
              { value: 'best', label: 'Best Result (AI Selected)' },
              { value: 'summary', label: 'AI Summary' }
            ]}
            size="small"
            fullWidth
          />
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
            How to combine results from parallel branches
          </Typography>
        </Box>

        {/* Branches Configuration */}
        <Accordion 
          expanded={configExpanded} 
          onChange={(_, isExpanded) => setConfigExpanded(isExpanded)}
          sx={{ mb: 1 }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box display="flex" alignItems="center" gap={1} width="100%">
              <SpeedIcon sx={{ fontSize: 20 }} />
              <Typography variant="body2" fontWeight={500}>Parallel Branches</Typography>
              {data.connectedNodes && data.connectedNodes.length > 0 && (
                <Chip 
                  label="Auto-managed" 
                  size="small" 
                  color="primary"
                  variant="outlined"
                  sx={{ ml: 1 }}
                />
              )}
              <Box flexGrow={1} />
              <Chip 
                label={`${branches.length} branches`} 
                size="small" 
                variant="outlined"
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Box>
              {data.connectedNodes && data.connectedNodes.length > 0 && (
                <Alert severity="info" sx={{ mb: 2 }}>
                  <Typography variant="caption">
                    Branches are automatically created based on connected nodes
                  </Typography>
                </Alert>
              )}
              
              <List dense sx={{ mb: 1 }}>
                {branches.map((branch) => {
                  const status = getBranchStatus(branch.id);
                  return (
                    <ListItem key={branch.id} sx={{ bgcolor: 'action.hover', mb: 0.5, borderRadius: 1 }}>
                      <Box display="flex" alignItems="center" gap={1} flex={1}>
                        {getBranchIcon(status)}
                        <TextField
                          size="small"
                          value={branch.name}
                          onChange={(e) => updateBranch(branch.id, 'name', e.target.value)}
                          variant="standard"
                          sx={{ flex: 1 }}
                        />
                        {branch.nodeId && (
                          <Chip
                            label={branch.nodeId}
                            size="small"
                            variant="outlined"
                            color="primary"
                          />
                        )}
                      </Box>
                      <ListItemSecondaryAction>
                        <IconButton 
                          size="small" 
                          onClick={() => deleteBranch(branch.id)}
                          disabled={!!data.connectedNodes?.length}
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </ListItemSecondaryAction>
                    </ListItem>
                  );
                })}
              </List>

              {!data.connectedNodes?.length && (
                <Button
                  startIcon={<AddIcon />}
                  onClick={addBranch}
                  variant="outlined"
                  size="small"
                  fullWidth
                >
                  Add Branch
                </Button>
              )}
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Advanced Settings */}
        <Accordion 
          expanded={advancedExpanded} 
          onChange={(_, isExpanded) => setAdvancedExpanded(isExpanded)}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box display="flex" alignItems="center" gap={1} width="100%">
              <SettingsIcon sx={{ fontSize: 20 }} />
              <Typography variant="body2" fontWeight={500}>Advanced Settings</Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Box display="flex" flexDirection="column" gap={2}>
              {/* Max Parallel */}
              <TextField
                size="small"
                type="number"
                label="Max Parallel (0 = unlimited)"
                value={maxParallel}
                onChange={(e) => {
                  const newValue = parseInt(e.target.value) || 0;
                  setMaxParallel(newValue);
                  updateParentData({ max_parallel: newValue });
                }}
                inputProps={{ min: 0, max: 100 }}
                fullWidth
                helperText="Maximum number of parallel executions"
              />

              {/* Timeout */}
              <TextField
                size="small"
                type="number"
                label="Timeout (ms)"
                value={timeout}
                onChange={(e) => {
                  const newValue = parseInt(e.target.value) || 60000;
                  setTimeout(newValue);
                  updateParentData({ timeout: newValue });
                }}
                inputProps={{ min: 0, step: 1000 }}
                fullWidth
                helperText="Maximum time to wait for branches"
              />

              {/* Wait for All */}
              <FormControlLabel
                control={
                  <Switch
                    checked={waitForAll}
                    onChange={(e) => {
                      const newValue = e.target.checked;
                      setWaitForAll(newValue);
                      updateParentData({ wait_for_all: newValue });
                    }}
                    size="small"
                  />
                }
                label="Wait for All Branches"
              />
              <Typography variant="caption" color="text.secondary" sx={{ mt: -1 }}>
                If disabled, returns as soon as one branch completes
              </Typography>

              {/* Continue on Error */}
              <FormControlLabel
                control={
                  <Switch
                    checked={continueOnError}
                    onChange={(e) => {
                      const newValue = e.target.checked;
                      setContinueOnError(newValue);
                      updateParentData({ continueOnError: newValue });
                    }}
                    size="small"
                  />
                }
                label="Continue on Error"
              />
              <Typography variant="caption" color="text.secondary" sx={{ mt: -1 }}>
                Continue execution even if some branches fail
              </Typography>
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Execution Status Details */}
        {hasExecutionData && (
          <Accordion 
            expanded={statusExpanded} 
            onChange={(_, isExpanded) => setStatusExpanded(isExpanded)}
            sx={{ mt: 1 }}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box display="flex" alignItems="center" gap={1} width="100%">
                <MemoryIcon sx={{ fontSize: 20 }} />
                <Typography variant="body2" fontWeight={500}>Execution Details</Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Box>
                <Typography variant="caption" component="div">
                  <strong>Running:</strong> {executionData?.runningBranches?.length || 0} branches
                </Typography>
                <Typography variant="caption" component="div">
                  <strong>Completed:</strong> {executionData?.completedBranches?.length || 0} branches
                </Typography>
                <Typography variant="caption" component="div">
                  <strong>Failed:</strong> {executionData?.failedBranches?.length || 0} branches
                </Typography>
                {executionData?.error && (
                  <Alert severity="error" sx={{ mt: 1 }}>
                    <Typography variant="caption">{executionData.error}</Typography>
                  </Alert>
                )}
              </Box>
            </AccordionDetails>
          </Accordion>
        )}

        {/* Input/Output Info */}
        {showIO && (
          <Box mt={2} p={1} bgcolor="action.hover" borderRadius={1}>
            <Typography variant="caption" color="text.secondary" fontWeight={500}>
              Executes connected nodes in parallel based on configuration
            </Typography>
          </Box>
        )}
      </CardContent>

      {/* Input handle */}
      <Handle 
        type="target" 
        position={Position.Top}
        id="input"
        style={{ 
          background: getStatusColor(),
          width: 12,
          height: 12,
          top: -6,
          left: '50%',
          transform: 'translateX(-50%)'
        }} 
      />
      
      {/* Multiple output handles for parallel branches */}
      {branches.map((branch, index) => (
        <Handle
          key={`parallel-${index + 1}`}
          type="source"
          position={Position.Right}
          id={`parallel-${index + 1}`}
          style={{
            background: getStatusColor(),
            width: 10,
            height: 10,
            right: -5,
            top: `${30 + (index * 25)}%`,
            transform: 'translateY(-50%)'
          }}
        />
      ))}
      
      {/* Summary/aggregated output handle */}
      <Handle 
        type="source" 
        position={Position.Bottom}
        id="summary"
        style={{ 
          background: getStatusColor(),
          width: 12,
          height: 12,
          bottom: -6,
          left: '50%',
          transform: 'translateX(-50%)'
        }} 
      />
    </Card>
  );
};

export default ParallelNode;