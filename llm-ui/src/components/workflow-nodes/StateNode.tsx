import React, { useState, useEffect } from 'react';
import { Handle, Position } from 'reactflow';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box, 
  TextField,
  FormControl,
  InputLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Alert,
  Tooltip,
  IconButton,
  Button,
  Paper
} from '@mui/material';
import { 
  Memory as StateIcon,
  ExpandMore as ExpandMoreIcon,
  Key as KeyIcon,
  Settings as SettingsIcon,
  Info as InfoIcon,
  Code as CodeIcon,
  MergeType as MergeIcon,
  ContentCopy as CopyIcon,
  Visibility as ViewIcon
} from '@mui/icons-material';
import ReactJson from 'react-json-view';
import PortalSelect from './PortalSelect';

interface StateNodeProps {
  data: {
    label?: string;
    stateKey?: string;
    operation?: 'read' | 'write' | 'update' | 'delete' | 'merge';
    defaultValue?: any;
    mergeStrategy?: 'shallow' | 'deep' | 'replace' | 'append';
    ttl?: number;
    scope?: 'workflow' | 'global' | 'user';
    persistState?: boolean;
    executionData?: {
      status?: 'idle' | 'running' | 'success' | 'error';
      currentValue?: any;
      previousValue?: any;
      operation?: string;
      error?: string;
    };
  };
  id: string;
  updateNodeData?: (nodeId: string, newData: any) => void;
  showIO?: boolean;
}

const StateNode: React.FC<StateNodeProps> = ({ data, id, updateNodeData, showIO = true }) => {
  const [configExpanded, setConfigExpanded] = useState(false);
  const [valueExpanded, setValueExpanded] = useState(false);
  
  const [label, setLabel] = useState(data.label || 'State Management');
  const [stateKey, setStateKey] = useState(data.stateKey || '');
  const [operation, setOperation] = useState(data.operation || 'read');
  const [defaultValue, setDefaultValue] = useState(data.defaultValue || {});
  const [mergeStrategy, setMergeStrategy] = useState(data.mergeStrategy || 'deep');
  const [ttl, setTtl] = useState(data.ttl || 0);
  const [scope, setScope] = useState(data.scope || 'workflow');
  const [persistState, setPersistState] = useState(data.persistState || false);

  // Update parent node data when local state changes
  useEffect(() => {
    if (updateNodeData) {
      updateNodeData(id, {
        ...data,
        label,
        stateKey,
        operation,
        defaultValue,
        mergeStrategy,
        ttl,
        scope,
        persistState
      });
    }
  }, [label, stateKey, operation, defaultValue, mergeStrategy, ttl, scope, persistState]);

  const getStatusColor = () => {
    switch (data.executionData?.status) {
      case 'running': return '#ff9800';
      case 'success': return '#4caf50';
      case 'error': return '#f44336';
      default: return '#2196f3';
    }
  };

  const getOperationColor = () => {
    switch (operation) {
      case 'read': return '#4caf50';
      case 'write': return '#2196f3';
      case 'update': return '#ff9800';
      case 'delete': return '#f44336';
      case 'merge': return '#9c27b0';
      default: return '#757575';
    }
  };

  const hasExecutionData = data.executionData && data.executionData.status !== 'idle';
  const currentValue = data.executionData?.currentValue;

  const handleDefaultValueEdit = (edit: any) => {
    setDefaultValue(edit.updated_src);
  };

  const operationDescriptions = {
    read: 'Read state value',
    write: 'Write new state value',
    update: 'Update existing state',
    delete: 'Delete state key',
    merge: 'Merge with existing state'
  };

  const getInputOutputInfo = () => {
    switch (operation) {
      case 'read':
        return {
          inputs: ['key (optional)'],
          outputs: ['value', 'exists']
        };
      case 'write':
      case 'update':
        return {
          inputs: ['value', 'key (optional)'],
          outputs: ['success', 'previous_value']
        };
      case 'delete':
        return {
          inputs: ['key (optional)'],
          outputs: ['success', 'deleted_value']
        };
      case 'merge':
        return {
          inputs: ['value', 'key (optional)'],
          outputs: ['merged_value', 'success']
        };
      default:
        return { inputs: [], outputs: [] };
    }
  };

  const ioInfo = getInputOutputInfo();

  return (
    <Card 
      sx={{ 
        minWidth: 350,
        maxWidth: 450,
        border: `2px solid ${getStatusColor()}`,
        borderRadius: 2,
        bgcolor: theme => theme.palette.mode === 'dark' ? 'rgba(33, 150, 243, 0.08)' : 'rgba(33, 150, 243, 0.05)',
        position: 'relative',
        overflow: 'visible',
        '&:hover': {
          boxShadow: theme => `0 0 20px ${theme.palette.mode === 'dark' ? 'rgba(33, 150, 243, 0.3)' : 'rgba(33, 150, 243, 0.2)'}`,
        }
      }}
    >
      <CardContent sx={{ pb: 1 }}>
        {/* Header */}
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          <StateIcon sx={{ color: getStatusColor() }} />
          <TextField
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            variant="standard"
            fullWidth
            sx={{
              '& .MuiInput-root': {
                fontSize: '1.1rem',
                fontWeight: 600,
              }
            }}
          />
          <Tooltip title="Manage workflow state and shared data">
            <InfoIcon sx={{ fontSize: 18, color: 'text.secondary' }} />
          </Tooltip>
        </Box>

        {/* Operation Status */}
        {hasExecutionData && (
          <Alert 
            severity={data.executionData?.status === 'error' ? 'error' : 'info'} 
            sx={{ mb: 2 }}
          >
            <Typography variant="caption" fontWeight={500}>
              {data.executionData?.operation || operation} operation {data.executionData?.status}
            </Typography>
            {data.executionData?.error && (
              <Typography variant="caption" display="block">
                Error: {data.executionData.error}
              </Typography>
            )}
          </Alert>
        )}

        {/* State Key */}
        <TextField
          fullWidth
          size="small"
          label="State Key"
          value={stateKey}
          onChange={(e) => setStateKey(e.target.value)}
          placeholder="user.preferences"
          helperText="Dot notation supported (e.g., user.settings.theme)"
          sx={{ mb: 2 }}
          InputProps={{
            startAdornment: <KeyIcon sx={{ mr: 1, color: 'text.secondary' }} />
          }}
        />

        {/* Operation Selection */}
        <Box sx={{ mb: 2 }}>
          <PortalSelect
            value={operation}
            onChange={(value) => setOperation(value as string)}
            label="Operation"
            options={Object.entries(operationDescriptions).map(([op, desc]) => ({
              value: op,
              label: desc
            }))}
            size="small"
            fullWidth
          />
        </Box>

        {/* Configuration */}
        <Accordion 
          expanded={configExpanded} 
          onChange={(_, isExpanded) => setConfigExpanded(isExpanded)}
          sx={{ mb: 1 }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box display="flex" alignItems="center" gap={1} width="100%">
              <SettingsIcon sx={{ fontSize: 20 }} />
              <Typography variant="body2" fontWeight={500}>State Configuration</Typography>
              <Box flexGrow={1} />
              <Chip 
                label={scope} 
                size="small" 
                variant="outlined"
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Box display="flex" flexDirection="column" gap={2}>
              {/* Scope */}
              <PortalSelect
                value={scope}
                onChange={(value) => setScope(value as string)}
                label="State Scope"
                options={[
                  { value: 'workflow', label: 'Workflow (Current workflow only)' },
                  { value: 'global', label: 'Global (All workflows)' },
                  { value: 'user', label: 'User (Per user state)' }
                ]}
                size="small"
                fullWidth
              />

              {/* Merge Strategy (for merge/update operations) */}
              {(operation === 'merge' || operation === 'update') && (
                <PortalSelect
                  value={mergeStrategy}
                  onChange={(value) => setMergeStrategy(value as string)}
                  label="Merge Strategy"
                  options={[
                    { value: 'shallow', label: 'Shallow Merge' },
                    { value: 'deep', label: 'Deep Merge' },
                    { value: 'replace', label: 'Replace' },
                    { value: 'append', label: 'Append (Arrays)' }
                  ]}
                  size="small"
                  fullWidth
                />
              )}

              {/* TTL */}
              <TextField
                size="small"
                type="number"
                label="TTL (seconds)"
                value={ttl}
                onChange={(e) => setTtl(parseInt(e.target.value) || 0)}
                helperText="0 = no expiration"
                fullWidth
              />

              {/* Default Value */}
              {operation !== 'delete' && (
                <Box>
                  <Typography variant="body2" gutterBottom>
                    Default Value (if key doesn't exist):
                  </Typography>
                  <Paper variant="outlined" sx={{ p: 1, maxHeight: 200, overflow: 'auto' }}>
                    <ReactJson
                      src={defaultValue}
                      theme={localStorage.getItem('theme') === 'dark' ? 'monokai' : 'rjv-default'}
                      onEdit={handleDefaultValueEdit}
                      onAdd={handleDefaultValueEdit}
                      onDelete={handleDefaultValueEdit}
                      displayDataTypes={false}
                      displayObjectSize={false}
                      enableClipboard={true}
                      style={{ fontSize: '0.85rem' }}
                    />
                  </Paper>
                </Box>
              )}
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Current Value Display */}
        {hasExecutionData && currentValue !== undefined && (
          <Accordion 
            expanded={valueExpanded} 
            onChange={(_, isExpanded) => setValueExpanded(isExpanded)}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box display="flex" alignItems="center" gap={1} width="100%">
                <ViewIcon sx={{ fontSize: 20 }} />
                <Typography variant="body2" fontWeight={500}>Current Value</Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Paper variant="outlined" sx={{ p: 1, maxHeight: 200, overflow: 'auto' }}>
                <ReactJson
                  src={currentValue}
                  theme={localStorage.getItem('theme') === 'dark' ? 'monokai' : 'rjv-default'}
                  displayDataTypes={false}
                  displayObjectSize={false}
                  enableClipboard={true}
                  collapsed={false}
                  style={{ fontSize: '0.85rem' }}
                />
              </Paper>
            </AccordionDetails>
          </Accordion>
        )}

        {/* Input/Output Info */}
        {showIO && (
          <Box mt={2} p={1} bgcolor="action.hover" borderRadius={1}>
            <Box mb={1}>
              <Typography variant="caption" color="text.secondary" fontWeight={500}>
                Inputs:
              </Typography>
              <Box mt={0.5}>
                {ioInfo.inputs.map(input => (
                  <Chip key={input} label={input} size="small" sx={{ mr: 0.5, mb: 0.5 }} />
                ))}
              </Box>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary" fontWeight={500}>
                Outputs:
              </Typography>
              <Box mt={0.5}>
                {ioInfo.outputs.map(output => (
                  <Chip key={output} label={output} size="small" sx={{ mr: 0.5, mb: 0.5 }} />
                ))}
              </Box>
            </Box>
          </Box>
        )}
      </CardContent>

      <Handle 
        type="target" 
        position={Position.Top} 
        style={{ 
          background: getStatusColor(),
          width: 12,
          height: 12,
          top: -6
        }} 
      />
      <Handle 
        type="source" 
        position={Position.Bottom} 
        style={{ 
          background: getStatusColor(),
          width: 12,
          height: 12,
          bottom: -6
        }} 
      />
    </Card>
  );
};

export default StateNode;