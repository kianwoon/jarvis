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
  FormControlLabel,
  Switch,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Alert,
  Tooltip,
  IconButton,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction
} from '@mui/material';
import { 
  MergeType as AggregatorIcon,
  ExpandMore as ExpandMoreIcon,
  Settings as SettingsIcon,
  Info as InfoIcon,
  Timer as TimerIcon,
  LayersClear as LayersIcon,
  Analytics as AnalyticsIcon,
  Delete as DeleteIcon
} from '@mui/icons-material';
import ReactJson from 'react-json-view';
import PortalSelect from './PortalSelect';

interface AggregatorNodeProps {
  data: {
    label?: string;
    aggregationType?: 'collect' | 'merge' | 'concat' | 'reduce' | 'group' | 'summary';
    waitStrategy?: 'all' | 'any' | 'timeout' | 'count';
    timeout?: number;
    maxItems?: number;
    mergeStrategy?: 'shallow' | 'deep' | 'custom';
    groupByField?: string;
    outputFormat?: 'array' | 'object' | 'string';
    customReducer?: string;
    executionData?: {
      status?: 'idle' | 'collecting' | 'processing' | 'success' | 'error';
      collectedCount?: number;
      pendingInputs?: string[];
      aggregatedResult?: any;
      error?: string;
    };
  };
  id: string;
  updateNodeData?: (nodeId: string, newData: any) => void;
  showIO?: boolean;
}

const AggregatorNode: React.FC<AggregatorNodeProps> = ({ data, id, updateNodeData, showIO = true }) => {
  const [configExpanded, setConfigExpanded] = useState(false);
  const [statusExpanded, setStatusExpanded] = useState(false);
  
  const [label, setLabel] = useState(data.label || 'Data Aggregator');
  const [aggregationType, setAggregationType] = useState(data.aggregationType || 'collect');
  const [waitStrategy, setWaitStrategy] = useState(data.waitStrategy || 'all');
  const [timeout, setTimeout] = useState(data.timeout || 30000);
  const [maxItems, setMaxItems] = useState(data.maxItems || 100);
  const [mergeStrategy, setMergeStrategy] = useState(data.mergeStrategy || 'deep');
  const [groupByField, setGroupByField] = useState(data.groupByField || '');
  const [outputFormat, setOutputFormat] = useState(data.outputFormat || 'array');
  const [customReducer, setCustomReducer] = useState(data.customReducer || '');

  // Update parent node data when local state changes
  useEffect(() => {
    if (updateNodeData) {
      updateNodeData(id, {
        ...data,
        label,
        aggregationType,
        waitStrategy,
        timeout,
        maxItems,
        mergeStrategy,
        groupByField,
        outputFormat,
        customReducer
      });
    }
  }, [label, aggregationType, waitStrategy, timeout, maxItems, 
      mergeStrategy, groupByField, outputFormat, customReducer]);

  const getStatusColor = () => {
    switch (data.executionData?.status) {
      case 'collecting': return '#ff9800';
      case 'processing': return '#2196f3';
      case 'success': return '#4caf50';
      case 'error': return '#f44336';
      default: return '#8bc34a';
    }
  };

  const hasExecutionData = data.executionData && data.executionData.status !== 'idle';
  const isCollecting = data.executionData?.status === 'collecting';

  const aggregationDescriptions = {
    collect: 'Collect all inputs into an array',
    merge: 'Merge objects into a single object',
    concat: 'Concatenate arrays or strings',
    reduce: 'Apply a reduce function to inputs',
    group: 'Group inputs by a field value',
    summary: 'Generate statistical summary'
  };

  const waitStrategyDescriptions = {
    all: 'Wait for all inputs',
    any: 'Process when any input arrives',
    timeout: 'Wait until timeout',
    count: 'Wait for specific count'
  };

  const getExampleOutput = () => {
    switch (aggregationType) {
      case 'collect':
        return [{ id: 1, data: 'A' }, { id: 2, data: 'B' }];
      case 'merge':
        return { fieldA: 'valueA', fieldB: 'valueB', combined: true };
      case 'concat':
        return ['item1', 'item2', 'item3'];
      case 'group':
        return { 
          groupA: [{ group: 'A', value: 1 }, { group: 'A', value: 2 }],
          groupB: [{ group: 'B', value: 3 }]
        };
      case 'summary':
        return { count: 10, sum: 100, avg: 10, min: 1, max: 20 };
      default:
        return {};
    }
  };

  return (
    <Card 
      sx={{ 
        minWidth: 380,
        maxWidth: 480,
        border: `2px solid ${getStatusColor()}`,
        borderRadius: 2,
        bgcolor: theme => theme.palette.mode === 'dark' ? 'rgba(139, 195, 74, 0.08)' : 'rgba(139, 195, 74, 0.05)',
        position: 'relative',
        overflow: 'visible',
        '&:hover': {
          boxShadow: theme => `0 0 20px ${theme.palette.mode === 'dark' ? 'rgba(139, 195, 74, 0.3)' : 'rgba(139, 195, 74, 0.2)'}`,
        }
      }}
    >
      <CardContent sx={{ pb: 1 }}>
        {/* Header */}
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          <AggregatorIcon sx={{ color: getStatusColor() }} />
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
          <Tooltip title="Aggregate multiple inputs into a single output">
            <InfoIcon sx={{ fontSize: 18, color: 'text.secondary' }} />
          </Tooltip>
        </Box>

        {/* Collection Status */}
        {hasExecutionData && (
          <Alert 
            severity={data.executionData?.status === 'error' ? 'error' : 'info'} 
            sx={{ mb: 2 }}
            icon={isCollecting ? <TimerIcon /> : <AnalyticsIcon />}
          >
            <Typography variant="caption" fontWeight={500}>
              {isCollecting 
                ? `Collecting... (${data.executionData?.collectedCount || 0} items)`
                : `Aggregation ${data.executionData?.status}`}
            </Typography>
            {data.executionData?.error && (
              <Typography variant="caption" display="block">
                Error: {data.executionData.error}
              </Typography>
            )}
          </Alert>
        )}

        {/* Aggregation Type */}
        <Box sx={{ mb: 2 }}>
          <PortalSelect
            value={aggregationType}
            onChange={(value) => setAggregationType(value as string)}
            label="Aggregation Type"
            options={Object.entries(aggregationDescriptions).map(([type, desc]) => ({
              value: type,
              label: `${type} - ${desc}`
            }))}
            size="small"
            fullWidth
          />
        </Box>

        {/* Wait Strategy */}
        <Box sx={{ mb: 2 }}>
          <PortalSelect
            value={waitStrategy}
            onChange={(value) => setWaitStrategy(value as string)}
            label="Wait Strategy"
            options={Object.entries(waitStrategyDescriptions).map(([strategy, desc]) => ({
              value: strategy,
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
              <Typography variant="body2" fontWeight={500}>Aggregation Settings</Typography>
              <Box flexGrow={1} />
              <Chip 
                label={aggregationType} 
                size="small" 
                variant="outlined"
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Box display="flex" flexDirection="column" gap={2}>
              {/* Timeout (for timeout strategy) */}
              {(waitStrategy === 'timeout' || waitStrategy === 'all') && (
                <TextField
                  size="small"
                  type="number"
                  label="Timeout (ms)"
                  value={timeout}
                  onChange={(e) => setTimeout(parseInt(e.target.value) || 30000)}
                  helperText="Maximum wait time before processing"
                  fullWidth
                />
              )}

              {/* Max Items (for count strategy) */}
              {waitStrategy === 'count' && (
                <TextField
                  size="small"
                  type="number"
                  label="Required Count"
                  value={maxItems}
                  onChange={(e) => setMaxItems(parseInt(e.target.value) || 1)}
                  helperText="Number of items to wait for"
                  fullWidth
                />
              )}

              {/* Group By Field (for group aggregation) */}
              {aggregationType === 'group' && (
                <TextField
                  size="small"
                  label="Group By Field"
                  value={groupByField}
                  onChange={(e) => setGroupByField(e.target.value)}
                  placeholder="category"
                  helperText="Field name to group by"
                  fullWidth
                />
              )}

              {/* Merge Strategy (for merge aggregation) */}
              {aggregationType === 'merge' && (
                <PortalSelect
                  value={mergeStrategy}
                  onChange={(value) => setMergeStrategy(value as string)}
                  label="Merge Strategy"
                  options={[
                    { value: 'shallow', label: 'Shallow Merge' },
                    { value: 'deep', label: 'Deep Merge' },
                    { value: 'custom', label: 'Custom Function' }
                  ]}
                  size="small"
                  fullWidth
                />
              )}

              {/* Output Format */}
              <PortalSelect
                value={outputFormat}
                onChange={(value) => setOutputFormat(value as string)}
                label="Output Format"
                options={[
                  { value: 'array', label: 'Array' },
                  { value: 'object', label: 'Object' },
                  { value: 'string', label: 'Concatenated String' }
                ]}
                size="small"
                fullWidth
              />

              {/* Example Output */}
              <Box>
                <Typography variant="body2" gutterBottom>
                  Example Output:
                </Typography>
                <Paper variant="outlined" sx={{ p: 1, maxHeight: 150, overflow: 'auto' }}>
                  <ReactJson
                    src={getExampleOutput()}
                    theme={localStorage.getItem('theme') === 'dark' ? 'monokai' : 'rjv-default'}
                    displayDataTypes={false}
                    displayObjectSize={false}
                    enableClipboard={false}
                    collapsed={false}
                    name={false}
                    style={{ fontSize: '0.85rem' }}
                  />
                </Paper>
              </Box>
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Collection Status */}
        {hasExecutionData && data.executionData?.pendingInputs && (
          <Accordion 
            expanded={statusExpanded} 
            onChange={(_, isExpanded) => setStatusExpanded(isExpanded)}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box display="flex" alignItems="center" gap={1} width="100%">
                <LayersIcon sx={{ fontSize: 20 }} />
                <Typography variant="body2" fontWeight={500}>Collection Status</Typography>
                <Box flexGrow={1} />
                <Chip 
                  label={`${data.executionData.collectedCount || 0} items`} 
                  size="small" 
                  color={isCollecting ? 'warning' : 'success'}
                />
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <List dense>
                {data.executionData.pendingInputs.map((input, index) => (
                  <ListItem key={index}>
                    <ListItemText 
                      primary={`Input ${index + 1}`}
                      secondary={input}
                    />
                  </ListItem>
                ))}
              </List>
            </AccordionDetails>
          </Accordion>
        )}

        {/* Input/Output Info */}
        {showIO && (
          <Box mt={2} p={1} bgcolor="action.hover" borderRadius={1}>
            <Typography variant="caption" color="text.secondary" fontWeight={500}>
              Multiple inputs → Single aggregated output
            </Typography>
            <Box mt={1}>
              <Typography variant="caption" color="text.secondary">
                Collects and combines data from multiple sources
              </Typography>
            </Box>
          </Box>
        )}
      </CardContent>

      {/* Multiple input handles */}
      <Handle 
        type="target" 
        position={Position.Top} 
        id="input-1"
        style={{ 
          background: getStatusColor(),
          width: 12,
          height: 12,
          top: -6,
          left: '30%'
        }} 
      />
      <Handle 
        type="target" 
        position={Position.Top} 
        id="input-2"
        style={{ 
          background: getStatusColor(),
          width: 12,
          height: 12,
          top: -6,
          left: '50%'
        }} 
      />
      <Handle 
        type="target" 
        position={Position.Top} 
        id="input-3"
        style={{ 
          background: getStatusColor(),
          width: 12,
          height: 12,
          top: -6,
          left: '70%'
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

export default AggregatorNode;