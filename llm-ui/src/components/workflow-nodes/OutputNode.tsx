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
  Checkbox,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Alert,
  Button,
  Tooltip,
  IconButton,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import { 
  Output as OutputIcon,
  ExpandMore as ExpandMoreIcon,
  Settings as SettingsIcon,
  Save as SaveIcon,
  Info as InfoIcon,
  FileDownload as DownloadIcon,
  ContentCopy as CopyIcon,
  OpenInNew as OpenInNewIcon
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import PortalSelect from './PortalSelect';

interface OutputNodeProps {
  data: {
    label?: string;
    output_format?: 'text' | 'json' | 'markdown' | 'html' | 'auto';
    include_metadata?: boolean;
    include_tool_calls?: boolean;
    auto_display?: boolean;
    auto_save?: boolean;
    save_path?: string;
    hasReport?: boolean;
    lastReportTimestamp?: string;
    executionData?: {
      status?: 'idle' | 'running' | 'success' | 'error';
      output?: any;
      metadata?: any;
      tool_calls?: any[];
      error?: string;
    };
  };
  id: string;
  updateNodeData?: (nodeId: string, newData: any) => void;
  showIO?: boolean;
}

const OutputNode: React.FC<OutputNodeProps> = ({ data, id, updateNodeData, showIO = true }) => {
  const [settingsExpanded, setSettingsExpanded] = useState(false);
  const [fullReportOpen, setFullReportOpen] = useState(false);
  
  const [label, setLabel] = useState(data.label || 'Workflow Output');
  const [outputFormat, setOutputFormat] = useState(data.output_format || 'auto');
  const [includeMetadata, setIncludeMetadata] = useState(data.include_metadata || false);
  const [includeToolCalls, setIncludeToolCalls] = useState(data.include_tool_calls || false);
  const [autoDisplay, setAutoDisplay] = useState<boolean>(data.auto_display ?? true);
  const [autoSave, setAutoSave] = useState(data.auto_save || false);
  const [savePath, setSavePath] = useState(data.save_path || '');

  // Update parent node data when local state changes
  useEffect(() => {
    if (updateNodeData) {
      updateNodeData(id, {
        ...data,
        label,
        output_format: outputFormat,
        include_metadata: includeMetadata,
        include_tool_calls: includeToolCalls,
        auto_display: autoDisplay,
        auto_save: autoSave,
        save_path: savePath
      });
    }
  }, [label, outputFormat, includeMetadata, includeToolCalls, autoDisplay, autoSave, savePath]);

  const getStatusColor = () => {
    switch (data.executionData?.status) {
      case 'running': return '#ff9800';
      case 'success': return '#4caf50';
      case 'error': return '#f44336';
      default: return '#ec407a';
    }
  };

  const hasExecutionData = data.executionData && data.executionData.status !== 'idle';
  const hasOutput = data.executionData?.output;

  // Helper function to clean output text
  const cleanOutputText = (text: string) => {
    // Remove <think> tags and their content
    const withoutThinkTags = text.replace(/<think>[\s\S]*?<\/think>/gi, '');
    
    // Clean up extra whitespace
    return withoutThinkTags.replace(/\n\s*\n\s*\n/g, '\n\n').trim();
  };

  const renderOutput = () => {
    if (!hasOutput) return null;
    
    let output = data.executionData?.output;
    
    // Clean output if it's a string (remove <think> tags)
    if (typeof output === 'string') {
      output = cleanOutputText(output);
    }
    
    if (outputFormat === 'markdown' || (outputFormat === 'auto' && typeof output === 'string' && output.includes('#'))) {
      return (
        <Box sx={{ 
          maxHeight: 300, 
          overflow: 'auto',
          '& pre': { 
            bgcolor: 'action.hover', 
            p: 1, 
            borderRadius: 1,
            overflow: 'auto'
          }
        }}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {output}
          </ReactMarkdown>
        </Box>
      );
    } else if (outputFormat === 'json' || (outputFormat === 'auto' && typeof output === 'object')) {
      return (
        <Paper variant="outlined" sx={{ p: 1, maxHeight: 300, overflow: 'auto' }}>
          <pre style={{ margin: 0, fontSize: '0.85rem' }}>
            {JSON.stringify(output, null, 2)}
          </pre>
        </Paper>
      );
    } else {
      return (
        <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
          {typeof output === 'string' ? output : JSON.stringify(output)}
        </Typography>
      );
    }
  };

  const handleCopyOutput = () => {
    if (hasOutput) {
      let output = data.executionData?.output;
      if (typeof output === 'string') {
        output = cleanOutputText(output);
      } else {
        output = JSON.stringify(output, null, 2);
      }
      navigator.clipboard.writeText(output);
    }
  };

  const handleDownloadOutput = () => {
    if (hasOutput) {
      let output = data.executionData?.output;
      if (typeof output === 'string') {
        output = cleanOutputText(output);
      } else {
        output = JSON.stringify(output, null, 2);
      }
      const blob = new Blob([output], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `workflow-output-${new Date().toISOString()}.txt`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  return (
    <Card 
      sx={{ 
        minWidth: 350,
        maxWidth: 450,
        border: `2px solid ${getStatusColor()}`,
        borderRadius: 2,
        bgcolor: theme => theme.palette.mode === 'dark' ? 'rgba(236, 64, 122, 0.08)' : 'rgba(236, 64, 122, 0.05)',
        position: 'relative',
        overflow: 'visible',
        '&:hover': {
          boxShadow: theme => `0 0 20px ${theme.palette.mode === 'dark' ? 'rgba(236, 64, 122, 0.3)' : 'rgba(236, 64, 122, 0.2)'}`,
        }
      }}
    >
      <CardContent sx={{ pb: 1 }}>
        {/* Header */}
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          <OutputIcon sx={{ color: getStatusColor() }} />
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
          <Tooltip title="This node displays and saves the workflow output">
            <InfoIcon sx={{ fontSize: 18, color: 'text.secondary' }} />
          </Tooltip>
        </Box>

        {/* Execution Status */}
        <Box mb={2}>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
            <Typography variant="body2" fontWeight={500}>Output:</Typography>
            <Box>
              <IconButton 
                size="small" 
                onClick={() => {
                  // Try to find the workflow result from parent component
                  const workflowEditor = document.querySelector('[data-testid="workflow-editor"]');
                  if (workflowEditor) {
                    const event = new CustomEvent('viewWorkflowReport');
                    workflowEditor.dispatchEvent(event);
                  } else {
                    setFullReportOpen(true);
                  }
                }}
                title="View full report"
              >
                <OpenInNewIcon fontSize="small" />
              </IconButton>
            </Box>
          </Box>
        </Box>

        {/* Settings */}
        <Accordion 
          expanded={settingsExpanded} 
          onChange={(_, isExpanded) => setSettingsExpanded(isExpanded)}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box display="flex" alignItems="center" gap={1} width="100%">
              <SettingsIcon sx={{ fontSize: 20 }} />
              <Typography variant="body2" fontWeight={500}>Output Settings</Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Box display="flex" flexDirection="column" gap={2}>
              {/* Output Format */}
              <PortalSelect
                value={outputFormat}
                onChange={(value) => setOutputFormat(value as 'text' | 'json' | 'markdown' | 'html' | 'auto')}
                label="Output Format"
                options={[
                  { value: 'auto', label: 'Auto-detect' },
                  { value: 'text', label: 'Plain Text' },
                  { value: 'json', label: 'JSON' },
                  { value: 'markdown', label: 'Markdown' },
                  { value: 'html', label: 'HTML' }
                ]}
                size="small"
                fullWidth
              />

              {/* Include Options */}
              <Box>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={includeMetadata}
                      onChange={(e) => setIncludeMetadata(e.target.checked)}
                      size="small"
                    />
                  }
                  label="Include Metadata"
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={includeToolCalls}
                      onChange={(e) => setIncludeToolCalls(e.target.checked)}
                      size="small"
                    />
                  }
                  label="Include Tool Calls"
                />
              </Box>

              {/* Display Options */}
              <Box>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={autoDisplay}
                      onChange={(e) => setAutoDisplay(e.target.checked)}
                      size="small"
                    />
                  }
                  label="Auto Display Output"
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={autoSave}
                      onChange={(e) => setAutoSave(e.target.checked)}
                      size="small"
                    />
                  }
                  label="Auto Save Output"
                />
              </Box>

              {/* Save Path */}
              {autoSave && (
                <TextField
                  size="small"
                  fullWidth
                  label="Save Path"
                  value={savePath}
                  onChange={(e) => setSavePath(e.target.value)}
                  placeholder="output/workflow-result.txt"
                  helperText="Path relative to workspace"
                />
              )}
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Input Info */}
        {showIO && (
          <Box mt={2} p={1} bgcolor="action.hover" borderRadius={1}>
            <Typography variant="caption" color="text.secondary" fontWeight={500}>
              Inputs:
            </Typography>
            <Box mt={0.5}>
              <Chip label="result" size="small" sx={{ mr: 0.5, mb: 0.5 }} />
              <Chip label="context" size="small" sx={{ mr: 0.5, mb: 0.5 }} />
              <Chip label="metadata" size="small" sx={{ mb: 0.5 }} />
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
      {/* No output handle - this is the end of the workflow */}
      
      {/* Full Report Dialog */}
      <Dialog
        open={fullReportOpen}
        onClose={() => setFullReportOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box display="flex" alignItems="center" gap={1}>
            <OutputIcon sx={{ color: getStatusColor() }} />
            {label} - Full Report
          </Box>
        </DialogTitle>
        <DialogContent>
          {hasOutput ? (
            <Box>
              {data.executionData?.status === 'error' ? (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {data.executionData?.error || 'An error occurred'}
                </Alert>
              ) : (
                <Box>
                  {/* Render full output with proper markdown */}
                  {(() => {
                    let output = data.executionData?.output;
                    if (typeof output === 'string') {
                      output = cleanOutputText(output);
                    }
                    
                    if (outputFormat === 'markdown' || (outputFormat === 'auto' && typeof output === 'string' && output.includes('#'))) {
                      return (
                        <Box sx={{ 
                          '& pre': { 
                            bgcolor: 'action.hover', 
                            p: 1, 
                            borderRadius: 1,
                            overflow: 'auto'
                          },
                          '& code': {
                            bgcolor: 'action.hover',
                            px: 0.5,
                            borderRadius: 0.5
                          }
                        }}>
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {output}
                          </ReactMarkdown>
                        </Box>
                      );
                    } else if (outputFormat === 'json' || (outputFormat === 'auto' && typeof output === 'object')) {
                      return (
                        <Paper variant="outlined" sx={{ p: 2 }}>
                          <pre style={{ margin: 0, fontSize: '0.9rem', whiteSpace: 'pre-wrap' }}>
                            {JSON.stringify(output, null, 2)}
                          </pre>
                        </Paper>
                      );
                    } else {
                      return (
                        <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                          {typeof output === 'string' ? output : JSON.stringify(output, null, 2)}
                        </Typography>
                      );
                    }
                  })()}
                </Box>
              )}
              
              {/* Metadata and Tool Calls if enabled */}
              {includeMetadata && data.executionData?.metadata && (
                <Box mt={3}>
                  <Typography variant="h6" gutterBottom>Metadata</Typography>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <pre style={{ margin: 0, fontSize: '0.85rem', whiteSpace: 'pre-wrap' }}>
                      {JSON.stringify(data.executionData.metadata, null, 2)}
                    </pre>
                  </Paper>
                </Box>
              )}
              
              {includeToolCalls && data.executionData?.tool_calls && (
                <Box mt={3}>
                  <Typography variant="h6" gutterBottom>Tool Calls</Typography>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <pre style={{ margin: 0, fontSize: '0.85rem', whiteSpace: 'pre-wrap' }}>
                      {JSON.stringify(data.executionData.tool_calls, null, 2)}
                    </pre>
                  </Paper>
                </Box>
              )}
            </Box>
          ) : (
            <Typography color="text.secondary">
              No output available to display.
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCopyOutput} startIcon={<CopyIcon />}>
            Copy Output
          </Button>
          <Button onClick={handleDownloadOutput} startIcon={<DownloadIcon />}>
            Download
          </Button>
          <Button onClick={() => setFullReportOpen(false)} variant="contained">
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Card>
  );
};

export default OutputNode;