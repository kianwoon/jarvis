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
  Tabs,
  Tab
} from '@mui/material';
import { 
  Transform as TransformIcon,
  ExpandMore as ExpandMoreIcon,
  Code as CodeIcon,
  PlayArrow as TestIcon,
  Info as InfoIcon,
  Functions as FunctionIcon,
  DataObject as JsonIcon,
  TextFields as TextIcon
} from '@mui/icons-material';
import Editor from '@monaco-editor/react';
import PortalSelect from './PortalSelect';

interface TransformNodeProps {
  data: {
    label?: string;
    transformType?: 'javascript' | 'python' | 'jmespath' | 'jsonata' | 'template';
    transformCode?: string;
    inputSchema?: any;
    outputSchema?: any;
    testInput?: any;
    libraries?: string[];
    executionData?: {
      status?: 'idle' | 'running' | 'success' | 'error';
      input?: any;
      output?: any;
      executionTime?: number;
      error?: string;
    };
  };
  id: string;
  updateNodeData?: (nodeId: string, newData: any) => void;
  showIO?: boolean;
}

const TransformNode: React.FC<TransformNodeProps> = ({ data, id, updateNodeData, showIO = true }) => {
  const [codeExpanded, setCodeExpanded] = useState(true);
  const [testExpanded, setTestExpanded] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  
  const [label, setLabel] = useState(data.label || 'Data Transformer');
  const [transformType, setTransformType] = useState(data.transformType || 'javascript');
  const [transformCode, setTransformCode] = useState(data.transformCode || getDefaultCode(data.transformType || 'javascript'));
  const [testInput, setTestInput] = useState(data.testInput || { example: 'test data' });
  const [testOutput, setTestOutput] = useState<any>(null);
  const [testError, setTestError] = useState<string>('');

  // Update parent node data when local state changes
  useEffect(() => {
    if (updateNodeData) {
      updateNodeData(id, {
        ...data,
        label,
        transformType,
        transformCode,
        testInput
      });
    }
  }, [label, transformType, transformCode, testInput]);

  const getStatusColor = () => {
    switch (data.executionData?.status) {
      case 'running': return '#ff9800';
      case 'success': return '#4caf50';
      case 'error': return '#f44336';
      default: return '#4caf50';
    }
  };

  const hasExecutionData = data.executionData && data.executionData.status !== 'idle';

  function getDefaultCode(type: string): string {
    switch (type) {
      case 'javascript':
        return `// Transform function - receives 'input' and returns transformed data
function transform(input) {
  // Example: Extract and format specific fields
  return {
    processed: true,
    timestamp: new Date().toISOString(),
    data: input
  };
}`;
      case 'python':
        return `# Transform function - receives 'input' and returns transformed data
def transform(input):
    # Example: Process and enrich data
    import json
    from datetime import datetime
    
    return {
        'processed': True,
        'timestamp': datetime.now().isoformat(),
        'data': input
    }`;
      case 'jmespath':
        return `// JMESPath query expression
{
  processed: \`true\`,
  timestamp: now(),
  data: @
}`;
      case 'jsonata':
        return `/* JSONata expression */
{
  "processed": true,
  "timestamp": $now(),
  "data": $
}`;
      case 'template':
        return `<!-- Handlebars/Mustache template -->
{
  "processed": true,
  "timestamp": "{{timestamp}}",
  "data": {{{json data}}}
}`;
      default:
        return '';
    }
  }

  const handleTransformTypeChange = (newType: string) => {
    setTransformType(newType);
    setTransformCode(getDefaultCode(newType));
  };

  const testTransformation = () => {
    // In a real implementation, this would execute the transformation
    // For now, we'll simulate it
    setTestError('');
    setTestOutput(null);
    
    try {
      // Simulate transformation
      setTimeout(() => {
        if (transformCode.includes('error')) {
          setTestError('Transformation error: Simulated error in code');
        } else {
          setTestOutput({
            processed: true,
            timestamp: new Date().toISOString(),
            originalInput: testInput,
            transformType
          });
        }
      }, 500);
    } catch (error) {
      setTestError(`Error: ${error.message}`);
    }
  };

  const getEditorLanguage = () => {
    switch (transformType) {
      case 'javascript': return 'javascript';
      case 'python': return 'python';
      case 'jmespath': return 'json';
      case 'jsonata': return 'json';
      case 'template': return 'handlebars';
      default: return 'javascript';
    }
  };

  return (
    <Card 
      sx={{ 
        minWidth: 450,
        maxWidth: 600,
        border: `2px solid ${getStatusColor()}`,
        borderRadius: 2,
        bgcolor: theme => theme.palette.mode === 'dark' ? 'rgba(76, 175, 80, 0.08)' : 'rgba(76, 175, 80, 0.05)',
        position: 'relative',
        overflow: 'visible',
        '&:hover': {
          boxShadow: theme => `0 0 20px ${theme.palette.mode === 'dark' ? 'rgba(76, 175, 80, 0.3)' : 'rgba(76, 175, 80, 0.2)'}`,
        }
      }}
    >
      <CardContent sx={{ pb: 1 }}>
        {/* Header */}
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          <TransformIcon sx={{ color: getStatusColor() }} />
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
          <Tooltip title="Transform data using code or expressions">
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
              Transformation {data.executionData?.status}
              {data.executionData?.executionTime && ` (${data.executionData.executionTime}ms)`}
            </Typography>
            {data.executionData?.error && (
              <Typography variant="caption" display="block">
                Error: {data.executionData.error}
              </Typography>
            )}
          </Alert>
        )}

        {/* Transform Type */}
        <Box sx={{ mb: 2 }}>
          <PortalSelect
            value={transformType}
            onChange={(value) => handleTransformTypeChange(value as string)}
            label="Transform Type"
            options={[
              { value: 'javascript', label: 'JavaScript Function' },
              { value: 'python', label: 'Python Function' },
              { value: 'jmespath', label: 'JMESPath Query' },
              { value: 'jsonata', label: 'JSONata Expression' },
              { value: 'template', label: 'Template (Handlebars)' }
            ]}
            size="small"
            fullWidth
          />
        </Box>

        {/* Transform Code */}
        <Accordion 
          expanded={codeExpanded} 
          onChange={(_, isExpanded) => setCodeExpanded(isExpanded)}
          sx={{ mb: 1 }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box display="flex" alignItems="center" gap={1} width="100%">
              <CodeIcon sx={{ fontSize: 20 }} />
              <Typography variant="body2" fontWeight={500}>Transform Code</Typography>
              <Box flexGrow={1} />
              <Chip 
                label={transformType} 
                size="small" 
                variant="outlined"
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails sx={{ p: 0 }}>
            <Box sx={{ height: 300, width: '100%' }}>
              <Editor
                height="100%"
                language={getEditorLanguage()}
                value={transformCode}
                onChange={(value) => setTransformCode(value || '')}
                theme={localStorage.getItem('theme') === 'dark' ? 'vs-dark' : 'light'}
                options={{
                  minimap: { enabled: false },
                  fontSize: 12,
                  lineNumbers: 'on',
                  scrollBeyondLastLine: false,
                  automaticLayout: true,
                  tabSize: 2
                }}
              />
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Test Transformation */}
        <Accordion 
          expanded={testExpanded} 
          onChange={(_, isExpanded) => setTestExpanded(isExpanded)}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box display="flex" alignItems="center" gap={1} width="100%">
              <TestIcon sx={{ fontSize: 20 }} />
              <Typography variant="body2" fontWeight={500}>Test Transformation</Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Box>
              <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)} sx={{ mb: 2 }}>
                <Tab label="Test Input" />
                <Tab label="Test Output" />
              </Tabs>

              {activeTab === 0 && (
                <Box>
                  <Typography variant="caption" color="text.secondary" gutterBottom>
                    Provide sample input data to test your transformation:
                  </Typography>
                  <Box sx={{ height: 200, mt: 1 }}>
                    <Editor
                      height="100%"
                      language="json"
                      value={JSON.stringify(testInput, null, 2)}
                      onChange={(value) => {
                        try {
                          setTestInput(JSON.parse(value || '{}'));
                        } catch (e) {
                          // Invalid JSON, ignore
                        }
                      }}
                      theme={localStorage.getItem('theme') === 'dark' ? 'vs-dark' : 'light'}
                      options={{
                        minimap: { enabled: false },
                        fontSize: 12,
                        lineNumbers: 'off',
                        scrollBeyondLastLine: false,
                        automaticLayout: true
                      }}
                    />
                  </Box>
                </Box>
              )}

              {activeTab === 1 && (
                <Box>
                  {testError ? (
                    <Alert severity="error" sx={{ mb: 1 }}>
                      {testError}
                    </Alert>
                  ) : testOutput ? (
                    <Box>
                      <Typography variant="caption" color="text.secondary" gutterBottom>
                        Transformation result:
                      </Typography>
                      <Paper variant="outlined" sx={{ p: 1, mt: 1, maxHeight: 200, overflow: 'auto' }}>
                        <pre style={{ margin: 0, fontSize: '0.85rem' }}>
                          {JSON.stringify(testOutput, null, 2)}
                        </pre>
                      </Paper>
                    </Box>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      Click "Run Test" to see the transformation output
                    </Typography>
                  )}
                </Box>
              )}

              <Button
                startIcon={<TestIcon />}
                onClick={testTransformation}
                variant="contained"
                size="small"
                fullWidth
                sx={{ mt: 2 }}
              >
                Run Test
              </Button>
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Input/Output Info */}
        {showIO && (
          <Box mt={2} p={1} bgcolor="action.hover" borderRadius={1}>
            <Typography variant="caption" color="text.secondary" fontWeight={500}>
              Transforms input data using the specified code/expression
            </Typography>
            <Box mt={1}>
              <Typography variant="caption" color="text.secondary">
                Input: Any data • Output: Transformed result
              </Typography>
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

export default TransformNode;