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
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import { 
  Route as RouterIcon,
  ExpandMore as ExpandMoreIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Settings as SettingsIcon,
  Info as InfoIcon,
  Rule as RuleIcon,
  Code as CodeIcon,
  ArrowForward as ArrowIcon
} from '@mui/icons-material';
import Editor from '@monaco-editor/react';
import PortalSelect from './PortalSelect';

interface RouteRule {
  id: string;
  condition: string;
  output: string;
  description?: string;
}

interface RouterNodeProps {
  data: {
    label?: string;
    routingType?: 'simple' | 'expression' | 'ai_based';
    routes?: RouteRule[];
    defaultRoute?: string;
    evaluationMode?: 'first_match' | 'all_matches' | 'weighted';
    contextVariables?: string[];
    executionData?: {
      status?: 'idle' | 'running' | 'success' | 'error';
      matchedRoute?: string;
      evaluatedConditions?: { [key: string]: boolean };
      selectedOutput?: string;
      error?: string;
    };
  };
  id: string;
  updateNodeData?: (nodeId: string, newData: any) => void;
  showIO?: boolean;
}

const RouterNode: React.FC<RouterNodeProps> = ({ data, id, updateNodeData, showIO = true }) => {
  const [rulesExpanded, setRulesExpanded] = useState(true);
  const [settingsExpanded, setSettingsExpanded] = useState(false);
  
  const [label, setLabel] = useState(data.label || 'Smart Router');
  const [routingType, setRoutingType] = useState(data.routingType || 'simple');
  const [routes, setRoutes] = useState<RouteRule[]>(data.routes || [
    { id: '1', condition: 'input.type == "question"', output: 'question_handler', description: 'Route questions' },
    { id: '2', condition: 'input.type == "command"', output: 'command_handler', description: 'Route commands' }
  ]);
  const [defaultRoute, setDefaultRoute] = useState(data.defaultRoute || 'default_handler');
  const [evaluationMode, setEvaluationMode] = useState(data.evaluationMode || 'first_match');
  const [editingCondition, setEditingCondition] = useState<string | null>(null);

  // Update parent node data when local state changes
  useEffect(() => {
    if (updateNodeData) {
      updateNodeData(id, {
        ...data,
        label,
        routingType,
        routes,
        defaultRoute,
        evaluationMode
      });
    }
  }, [label, routingType, routes, defaultRoute, evaluationMode]);

  const getStatusColor = () => {
    switch (data.executionData?.status) {
      case 'running': return '#ff9800';
      case 'success': return '#4caf50';
      case 'error': return '#f44336';
      default: return '#3f51b5';
    }
  };

  const hasExecutionData = data.executionData && data.executionData.status !== 'idle';

  const addRoute = () => {
    const newRoute: RouteRule = {
      id: Date.now().toString(),
      condition: '',
      output: '',
      description: ''
    };
    setRoutes([...routes, newRoute]);
  };

  const updateRoute = (id: string, field: keyof RouteRule, value: string) => {
    setRoutes(routes.map(route => 
      route.id === id ? { ...route, [field]: value } : route
    ));
  };

  const deleteRoute = (id: string) => {
    setRoutes(routes.filter(route => route.id !== id));
  };

  const getExampleCondition = () => {
    switch (routingType) {
      case 'simple':
        return 'input.category == "sales"';
      case 'expression':
        return 'input.score > 0.8 && input.confidence == "high"';
      case 'ai_based':
        return 'contains_intent("purchase") && sentiment == "positive"';
      default:
        return '';
    }
  };

  const getOutputOptions = () => {
    // In a real implementation, this would fetch available downstream nodes
    return [
      'question_handler',
      'command_handler',
      'data_processor',
      'error_handler',
      'default_handler'
    ];
  };

  return (
    <Card 
      sx={{ 
        minWidth: 450,
        maxWidth: 600,
        border: `2px solid ${getStatusColor()}`,
        borderRadius: 2,
        bgcolor: theme => theme.palette.mode === 'dark' ? 'rgba(63, 81, 181, 0.08)' : 'rgba(63, 81, 181, 0.05)',
        position: 'relative',
        overflow: 'visible',
        '&:hover': {
          boxShadow: theme => `0 0 20px ${theme.palette.mode === 'dark' ? 'rgba(63, 81, 181, 0.3)' : 'rgba(63, 81, 181, 0.2)'}`,
        }
      }}
    >
      <CardContent sx={{ pb: 1 }}>
        {/* Header */}
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          <RouterIcon sx={{ color: getStatusColor() }} />
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
          <Tooltip title="Route data to different paths based on conditions">
            <InfoIcon sx={{ fontSize: 18, color: 'text.secondary' }} />
          </Tooltip>
        </Box>

        {/* Execution Status */}
        {hasExecutionData && (
          <Alert 
            severity={data.executionData?.status === 'error' ? 'error' : 'success'} 
            sx={{ mb: 2 }}
          >
            <Typography variant="caption" fontWeight={500}>
              {data.executionData?.matchedRoute 
                ? `Routed to: ${data.executionData.selectedOutput}`
                : 'Using default route'}
            </Typography>
            {data.executionData?.error && (
              <Typography variant="caption" display="block">
                Error: {data.executionData.error}
              </Typography>
            )}
          </Alert>
        )}

        {/* Routing Type */}
        <Box sx={{ mb: 2 }}>
          <PortalSelect
            value={routingType}
            onChange={(value) => setRoutingType(value as string)}
            label="Routing Type"
            options={[
              { value: 'simple', label: 'Simple (Field Comparison)' },
              { value: 'expression', label: 'Expression (Complex Logic)' },
              { value: 'ai_based', label: 'AI-Based (Semantic Routing)' }
            ]}
            size="small"
            fullWidth
          />
        </Box>

        {/* Routing Rules */}
        <Accordion 
          expanded={rulesExpanded} 
          onChange={(_, isExpanded) => setRulesExpanded(isExpanded)}
          sx={{ mb: 1 }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box display="flex" alignItems="center" gap={1} width="100%">
              <RuleIcon sx={{ fontSize: 20 }} />
              <Typography variant="body2" fontWeight={500}>Routing Rules</Typography>
              <Box flexGrow={1} />
              <Chip 
                label={`${routes.length} routes`} 
                size="small" 
                variant="outlined"
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Box>
              <TableContainer component={Paper} variant="outlined" sx={{ mb: 2 }}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Condition</TableCell>
                      <TableCell>Output</TableCell>
                      <TableCell>Description</TableCell>
                      <TableCell width={50}></TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {routes.map((route, index) => (
                      <TableRow key={route.id}>
                        <TableCell>
                          {editingCondition === route.id ? (
                            <TextField
                              size="small"
                              fullWidth
                              value={route.condition}
                              onChange={(e) => updateRoute(route.id, 'condition', e.target.value)}
                              onBlur={() => setEditingCondition(null)}
                              autoFocus
                              placeholder={getExampleCondition()}
                            />
                          ) : (
                            <Typography 
                              variant="body2" 
                              sx={{ 
                                fontFamily: 'monospace',
                                cursor: 'pointer',
                                '&:hover': { bgcolor: 'action.hover' }
                              }}
                              onClick={() => setEditingCondition(route.id)}
                            >
                              {route.condition || 'Click to edit condition'}
                            </Typography>
                          )}
                        </TableCell>
                        <TableCell>
                          <PortalSelect
                            value={route.output || ''}
                            onChange={(value) => updateRoute(route.id, 'output', value as string)}
                            label=""
                            options={[
                              { value: '', label: 'Select output' },
                              ...getOutputOptions().map(option => ({ value: option, label: option }))
                            ]}
                            size="small"
                            fullWidth
                          />
                        </TableCell>
                        <TableCell>
                          <TextField
                            size="small"
                            fullWidth
                            value={route.description}
                            onChange={(e) => updateRoute(route.id, 'description', e.target.value)}
                            placeholder="Optional description"
                          />
                        </TableCell>
                        <TableCell>
                          <IconButton 
                            size="small" 
                            onClick={() => deleteRoute(route.id)}
                            color="error"
                          >
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>

              <Button
                startIcon={<AddIcon />}
                onClick={addRoute}
                variant="outlined"
                size="small"
                fullWidth
              >
                Add Route
              </Button>

              {/* Default Route */}
              <Box mt={2}>
                <TextField
                  size="small"
                  fullWidth
                  label="Default Route (No Match)"
                  value={defaultRoute}
                  onChange={(e) => setDefaultRoute(e.target.value)}
                  helperText="Output when no conditions match"
                />
              </Box>
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Advanced Settings */}
        <Accordion 
          expanded={settingsExpanded} 
          onChange={(_, isExpanded) => setSettingsExpanded(isExpanded)}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box display="flex" alignItems="center" gap={1} width="100%">
              <SettingsIcon sx={{ fontSize: 20 }} />
              <Typography variant="body2" fontWeight={500}>Advanced Settings</Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Box display="flex" flexDirection="column" gap={2}>
              {/* Evaluation Mode */}
              <PortalSelect
                value={evaluationMode}
                onChange={(value) => setEvaluationMode(value as string)}
                label="Evaluation Mode"
                options={[
                  { value: 'first_match', label: 'First Match (Stop on first true)' },
                  { value: 'all_matches', label: 'All Matches (Evaluate all)' },
                  { value: 'weighted', label: 'Weighted (Score-based routing)' }
                ]}
                size="small"
                fullWidth
              />

              {/* Condition Syntax Help */}
              <Box bgcolor="action.hover" p={1} borderRadius={1}>
                <Typography variant="caption" color="text.secondary" fontWeight={500}>
                  Condition Syntax:
                </Typography>
                <Typography variant="caption" display="block" sx={{ fontFamily: 'monospace', mt: 0.5 }}>
                  • Field access: input.field_name{'\n'}
                  • Comparisons: ==, !=, {'>'}, {'<'}, {'>='}, {'<='}{'\n'}
                  • Logic: && (AND), || (OR), ! (NOT){'\n'}
                  • Functions: contains(), startsWith(), matches()
                </Typography>
              </Box>
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Input/Output Info */}
        {showIO && (
          <Box mt={2} p={1} bgcolor="action.hover" borderRadius={1}>
            <Typography variant="caption" color="text.secondary" fontWeight={500}>
              Dynamic routing based on conditions - outputs connect to different downstream nodes
            </Typography>
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
      {/* Multiple output handles for different routes */}
      <Handle 
        type="source" 
        position={Position.Bottom} 
        id="default"
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

export default RouterNode;