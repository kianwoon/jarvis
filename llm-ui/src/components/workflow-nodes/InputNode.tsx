import React, { useState, useEffect } from 'react';
import { Handle, Position } from 'reactflow';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box, 
  TextField,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  IconButton,
  Tooltip,
  Chip,
  Alert
} from '@mui/material';
import { 
  Input as InputIcon,
  ExpandMore as ExpandMoreIcon,
  Schema as SchemaIcon,
  DataObject as DefaultsIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import ReactJson from 'react-json-view';

interface InputNodeProps {
  data: {
    label?: string;
    inputSchema?: any;
    defaultValues?: any;
    executionData?: {
      status?: 'idle' | 'running' | 'success' | 'error';
      data?: any;
      message?: string;
    };
  };
  id: string;
  updateNodeData?: (nodeId: string, newData: any) => void;
  showIO?: boolean;
}

const InputNode: React.FC<InputNodeProps> = ({ data, id, updateNodeData, showIO = true }) => {
  const [schemaExpanded, setSchemaExpanded] = useState(false);
  const [defaultsExpanded, setDefaultsExpanded] = useState(false);
  const [label, setLabel] = useState(data.label || 'Workflow Input');
  const [inputSchema, setInputSchema] = useState(data.inputSchema || {
    type: "object",
    properties: {
      query: { type: "string", description: "User query or question" },
      data: { type: "object", description: "Additional data" }
    }
  });
  const [defaultValues, setDefaultValues] = useState(data.defaultValues || {});

  // Update parent node data when local state changes
  useEffect(() => {
    if (updateNodeData) {
      updateNodeData(id, {
        ...data,
        label,
        inputSchema,
        defaultValues
      });
    }
  }, [label, inputSchema, defaultValues]);

  const handleSchemaEdit = (edit: any) => {
    setInputSchema(edit.updated_src);
  };

  const handleDefaultsEdit = (edit: any) => {
    setDefaultValues(edit.updated_src);
  };

  const getStatusColor = () => {
    switch (data.executionData?.status) {
      case 'running': return '#ff9800';
      case 'success': return '#4caf50';
      case 'error': return '#f44336';
      default: return '#059669';
    }
  };

  const hasExecutionData = data.executionData && data.executionData.status !== 'idle';

  return (
    <Card 
      sx={{ 
        minWidth: 350,
        maxWidth: 450,
        border: `2px solid ${getStatusColor()}`,
        borderRadius: 2,
        bgcolor: theme => theme.palette.mode === 'dark' ? 'rgba(5, 150, 105, 0.08)' : 'rgba(5, 150, 105, 0.05)',
        position: 'relative',
        overflow: 'visible',
        '&:hover': {
          boxShadow: theme => `0 0 20px ${theme.palette.mode === 'dark' ? 'rgba(5, 150, 105, 0.3)' : 'rgba(5, 150, 105, 0.2)'}`,
        }
      }}
    >
      <CardContent sx={{ pb: 1 }}>
        {/* Header */}
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          <InputIcon sx={{ color: getStatusColor() }} />
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
          <Tooltip title="This node defines the input data structure and default values for the workflow">
            <InfoIcon sx={{ fontSize: 18, color: 'text.secondary' }} />
          </Tooltip>
        </Box>

        {/* Execution Status */}
        {hasExecutionData && (
          <Alert 
            severity={data.executionData?.status === 'error' ? 'error' : 'success'}
            sx={{ mb: 2, py: 0.5 }}
          >
            {data.executionData?.message || `Input data: ${JSON.stringify(data.executionData?.data)}`}
          </Alert>
        )}

        {/* Schema Configuration */}
        <Accordion 
          expanded={schemaExpanded} 
          onChange={(_, isExpanded) => setSchemaExpanded(isExpanded)}
          sx={{ mb: 1 }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box display="flex" alignItems="center" gap={1} width="100%">
              <SchemaIcon sx={{ fontSize: 20 }} />
              <Typography variant="body2" fontWeight={500}>Input Schema</Typography>
              <Box flexGrow={1} />
              <Chip 
                label={`${Object.keys(inputSchema.properties || {}).length} fields`}
                size="small"
                variant="outlined"
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Box sx={{ 
              maxHeight: 300, 
              overflow: 'auto',
              '& .react-json-view': {
                backgroundColor: 'transparent !important',
              }
            }}>
              <ReactJson
                src={inputSchema}
                theme={localStorage.getItem('theme') === 'dark' ? 'monokai' : 'rjv-default'}
                onEdit={handleSchemaEdit}
                onAdd={handleSchemaEdit}
                onDelete={handleSchemaEdit}
                displayDataTypes={false}
                displayObjectSize={false}
                enableClipboard={true}
                style={{ fontSize: '0.85rem' }}
              />
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Default Values */}
        <Accordion 
          expanded={defaultsExpanded} 
          onChange={(_, isExpanded) => setDefaultsExpanded(isExpanded)}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box display="flex" alignItems="center" gap={1} width="100%">
              <DefaultsIcon sx={{ fontSize: 20 }} />
              <Typography variant="body2" fontWeight={500}>Default Values</Typography>
              <Box flexGrow={1} />
              <Chip 
                label={Object.keys(defaultValues).length === 0 ? 'None' : `${Object.keys(defaultValues).length} values`}
                size="small"
                variant="outlined"
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Box sx={{ 
              maxHeight: 300, 
              overflow: 'auto',
              '& .react-json-view': {
                backgroundColor: 'transparent !important',
              }
            }}>
              <ReactJson
                src={defaultValues}
                theme={localStorage.getItem('theme') === 'dark' ? 'monokai' : 'rjv-default'}
                onEdit={handleDefaultsEdit}
                onAdd={handleDefaultsEdit}
                onDelete={handleDefaultsEdit}
                displayDataTypes={false}
                displayObjectSize={false}
                enableClipboard={true}
                style={{ fontSize: '0.85rem' }}
              />
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Output Info */}
        {showIO && (
          <Box mt={2} p={1} bgcolor="action.hover" borderRadius={1}>
            <Typography variant="caption" color="text.secondary" fontWeight={500}>
              Outputs:
            </Typography>
            <Box mt={0.5}>
              <Chip label="data" size="small" sx={{ mr: 0.5, mb: 0.5 }} />
              <Chip label="message" size="small" sx={{ mb: 0.5 }} />
            </Box>
          </Box>
        )}
      </CardContent>

      {/* No input handle - this is the start of the workflow */}
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

export default InputNode;