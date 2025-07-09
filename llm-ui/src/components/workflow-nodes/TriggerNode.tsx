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
  IconButton,
  Tooltip,
  FormLabel,
  FormGroup,
  Paper,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tab,
  Tabs
} from '@mui/material';
import { 
  Link as TriggerIcon,
  ExpandMore as ExpandMoreIcon,
  Http as HttpIcon,
  Security as SecurityIcon,
  ContentCopy as CopyIcon,
  Refresh as RefreshIcon,
  Info as InfoIcon,
  Code as CodeIcon,
  Help as HelpIcon,
  PlayArrow as PlayIcon,
  DataObject as DataIcon
} from '@mui/icons-material';
import PortalSelect from './PortalSelect';

interface TriggerNodeProps {
  data: {
    label?: string;
    trigger_name?: string;
    http_methods?: string[];
    authentication_type?: 'none' | 'api_key' | 'bearer_token' | 'basic_auth' | 'custom_header';
    auth_header_name?: string;
    auth_token?: string;
    basic_auth_username?: string;
    basic_auth_password?: string;
    request_body_schema?: any;
    query_params_schema?: any;
    response_mapping?: any;
    rate_limit?: number;
    allowed_origins?: string[];
    executionData?: {
      status?: 'idle' | 'running' | 'success' | 'error';
      trigger_url?: string;
      last_triggered?: string;
      trigger_count?: number;
    };
  };
  id: string;
  updateNodeData?: (nodeId: string, newData: any) => void;
  showIO?: boolean;
}

const TriggerNode: React.FC<TriggerNodeProps> = ({ data, id, updateNodeData, showIO = true }) => {
  const [httpExpanded, setHttpExpanded] = useState(false);
  const [authExpanded, setAuthExpanded] = useState(false);
  const [schemaExpanded, setSchemaExpanded] = useState(false);
  const [helpModalOpen, setHelpModalOpen] = useState(false);
  const [helpTabValue, setHelpTabValue] = useState(0);
  
  const [label, setLabel] = useState(data.label || 'External Trigger');
  const [triggerName, setTriggerName] = useState(data.trigger_name || '');
  const [httpMethods, setHttpMethods] = useState<string[]>(data.http_methods || ['POST']);
  const [authenticationType, setAuthenticationType] = useState(data.authentication_type || 'api_key');
  const [authHeaderName, setAuthHeaderName] = useState(data.auth_header_name || 'X-API-Key');
  const [authToken, setAuthToken] = useState(data.auth_token || '');
  const [basicAuthUsername, setBasicAuthUsername] = useState(data.basic_auth_username || '');
  const [basicAuthPassword, setBasicAuthPassword] = useState(data.basic_auth_password || '');
  const [rateLimit, setRateLimit] = useState(data.rate_limit || 60);
  const [allowedOrigins, setAllowedOrigins] = useState(data.allowed_origins?.join('\n') || '*');

  // Update parent node data when local state changes
  useEffect(() => {
    if (updateNodeData) {
      updateNodeData(id, {
        ...data,
        label,
        trigger_name: triggerName,
        http_methods: httpMethods,
        authentication_type: authenticationType,
        auth_header_name: authHeaderName,
        auth_token: authToken,
        basic_auth_username: basicAuthUsername,
        basic_auth_password: basicAuthPassword,
        rate_limit: rateLimit,
        allowed_origins: allowedOrigins.split('\n').filter(o => o.trim())
      });
    }
  }, [label, triggerName, httpMethods, authenticationType, authHeaderName, authToken, 
      basicAuthUsername, basicAuthPassword, rateLimit, allowedOrigins]);

  const getStatusColor = () => {
    switch (data.executionData?.status) {
      case 'running': return '#ff9800';
      case 'success': return '#4caf50';
      case 'error': return '#f44336';
      default: return '#059669';
    }
  };

  const generateAuthToken = () => {
    const token = `tk_${Math.random().toString(36).substring(2)}${Date.now().toString(36)}`;
    setAuthToken(token);
  };

  const copyTriggerUrl = () => {
    if (data.executionData?.trigger_url) {
      navigator.clipboard.writeText(data.executionData.trigger_url);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const getEndpointUrl = () => {
    // Use backend server URL, not frontend
    const backendUrl = window.location.origin.replace(':5173', ':8000');
    return `${backendUrl}/api/v1/automation/external/trigger/${triggerName || 'your-trigger-name'}`;
  };

  const getCurlExample = () => {
    const url = getEndpointUrl();
    const authHeader = authenticationType !== 'none' ? `\n  -H "${authHeaderName || 'X-API-Key'}: ${authToken || 'your-token'}" \\` : '';
    
    return `curl -X POST \\
  ${url} \\${authHeader}
  -H "Content-Type: application/json" \\
  -d '{
    "message": "Process this data",
    "data": {
      "user_id": "12345",
      "action": "process_order"
    }
  }'`;
  };

  const getJavaScriptExample = () => {
    const url = getEndpointUrl();
    const authHeaders = authenticationType !== 'none' ? `\n    '${authHeaderName || 'X-API-Key'}': '${authToken || 'your-token'}',` : '';
    
    return `const response = await fetch('${url}', {
  method: 'POST',
  headers: {${authHeaders}
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    message: 'Process this data',
    data: {
      user_id: '12345',
      action: 'process_order'
    }
  })
});

const result = await response.json();
console.log(result);`;
  };

  const getPythonExample = () => {
    const url = getEndpointUrl();
    const authHeaders = authenticationType !== 'none' ? `\n    '${authHeaderName || 'X-API-Key'}': '${authToken || 'your-token'}',` : '';
    
    return `import requests

url = '${url}'
headers = {${authHeaders}
    'Content-Type': 'application/json'
}
data = {
    'message': 'Process this data',
    'data': {
        'user_id': '12345',
        'action': 'process_order'
    }
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
print(result)`;
  };

  const hasExecutionData = data.executionData && data.executionData.trigger_url;

  const methodColors: Record<string, string> = {
    GET: '#4caf50',
    POST: '#2196f3',
    PUT: '#ff9800',
    DELETE: '#f44336',
    PATCH: '#9c27b0'
  };

  return (
    <Card 
      sx={{ 
        minWidth: 380,
        maxWidth: 480,
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
          <TriggerIcon sx={{ color: getStatusColor() }} />
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
          <Tooltip title="API Documentation & Examples">
            <IconButton size="small" onClick={() => setHelpModalOpen(true)}>
              <HelpIcon sx={{ fontSize: 18, color: 'text.secondary' }} />
            </IconButton>
          </Tooltip>
        </Box>

        {/* Trigger URL Display */}
        {hasExecutionData && (
          <Alert 
            severity="info" 
            sx={{ mb: 2 }}
            action={
              <IconButton size="small" onClick={copyTriggerUrl}>
                <CopyIcon />
              </IconButton>
            }
          >
            <Typography variant="caption" fontWeight={500}>
              Trigger URL: {data.executionData?.trigger_url}
            </Typography>
            {data.executionData?.last_triggered && (
              <Typography variant="caption" display="block">
                Last triggered: {new Date(data.executionData.last_triggered).toLocaleString()}
              </Typography>
            )}
          </Alert>
        )}

        {/* Trigger Name */}
        <TextField
          fullWidth
          size="small"
          label="Trigger Name"
          value={triggerName}
          onChange={(e) => setTriggerName(e.target.value)}
          placeholder="my-workflow-trigger"
          helperText="Unique identifier for this trigger endpoint"
          sx={{ mb: 2 }}
        />

        {/* HTTP Configuration */}
        <Accordion 
          expanded={httpExpanded} 
          onChange={(_, isExpanded) => setHttpExpanded(isExpanded)}
          sx={{ mb: 1 }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box display="flex" alignItems="center" gap={1} width="100%">
              <HttpIcon sx={{ fontSize: 20 }} />
              <Typography variant="body2" fontWeight={500}>HTTP Configuration</Typography>
              <Box flexGrow={1} />
              {httpMethods.map(method => (
                <Chip 
                  key={method}
                  label={method} 
                  size="small" 
                  sx={{ 
                    bgcolor: methodColors[method], 
                    color: 'white',
                    fontWeight: 500
                  }} 
                />
              ))}
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Box display="flex" flexDirection="column" gap={2}>
              {/* HTTP Methods */}
              <FormControl size="small">
                <FormLabel>Allowed HTTP Methods</FormLabel>
                <FormGroup row>
                  {['GET', 'POST', 'PUT', 'DELETE', 'PATCH'].map(method => (
                    <FormControlLabel
                      key={method}
                      control={
                        <Checkbox
                          checked={httpMethods.includes(method)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setHttpMethods([...httpMethods, method]);
                            } else {
                              setHttpMethods(httpMethods.filter(m => m !== method));
                            }
                          }}
                          size="small"
                        />
                      }
                      label={method}
                    />
                  ))}
                </FormGroup>
              </FormControl>

              {/* Rate Limit */}
              <TextField
                size="small"
                type="number"
                label="Rate Limit (requests/minute)"
                value={rateLimit}
                onChange={(e) => setRateLimit(parseInt(e.target.value) || 60)}
                fullWidth
              />

              {/* Allowed Origins */}
              <TextField
                size="small"
                multiline
                rows={2}
                label="Allowed Origins (CORS)"
                value={allowedOrigins}
                onChange={(e) => setAllowedOrigins(e.target.value)}
                placeholder="*"
                helperText="One origin per line, * for all"
                fullWidth
              />
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Authentication */}
        <Accordion 
          expanded={authExpanded} 
          onChange={(_, isExpanded) => setAuthExpanded(isExpanded)}
          sx={{ mb: 1 }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box display="flex" alignItems="center" gap={1} width="100%">
              <SecurityIcon sx={{ fontSize: 20 }} />
              <Typography variant="body2" fontWeight={500}>Authentication</Typography>
              <Box flexGrow={1} />
              <Chip 
                label={authenticationType === 'none' ? 'No Auth' : authenticationType.replace('_', ' ')} 
                size="small" 
                variant="outlined"
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Box display="flex" flexDirection="column" gap={2}>
              {/* Authentication Type */}
              <PortalSelect
                value={authenticationType}
                onChange={(value) => setAuthenticationType(value as string)}
                label="Authentication Type"
                options={[
                  { value: 'none', label: 'No Authentication' },
                  { value: 'api_key', label: 'API Key (Header)' },
                  { value: 'bearer_token', label: 'Bearer Token' },
                  { value: 'basic_auth', label: 'Basic Authentication' },
                  { value: 'custom_header', label: 'Custom Header' }
                ]}
                size="small"
                fullWidth
              />

              {/* API Key / Custom Header */}
              {(authenticationType === 'api_key' || authenticationType === 'custom_header') && (
                <>
                  <TextField
                    size="small"
                    label="Header Name"
                    value={authHeaderName}
                    onChange={(e) => setAuthHeaderName(e.target.value)}
                    fullWidth
                  />
                  <Box display="flex" gap={1}>
                    <TextField
                      size="small"
                      label="API Key / Token"
                      value={authToken}
                      onChange={(e) => setAuthToken(e.target.value)}
                      placeholder="Auto-generated"
                      fullWidth
                    />
                    <Button
                      variant="outlined"
                      size="small"
                      onClick={generateAuthToken}
                      startIcon={<RefreshIcon />}
                    >
                      Generate
                    </Button>
                  </Box>
                </>
              )}

              {/* Bearer Token */}
              {authenticationType === 'bearer_token' && (
                <Box display="flex" gap={1}>
                  <TextField
                    size="small"
                    label="Bearer Token"
                    value={authToken}
                    onChange={(e) => setAuthToken(e.target.value)}
                    placeholder="Auto-generated"
                    fullWidth
                  />
                  <Button
                    variant="outlined"
                    size="small"
                    onClick={generateAuthToken}
                    startIcon={<RefreshIcon />}
                  >
                    Generate
                  </Button>
                </Box>
              )}

              {/* Basic Auth */}
              {authenticationType === 'basic_auth' && (
                <>
                  <TextField
                    size="small"
                    label="Username"
                    value={basicAuthUsername}
                    onChange={(e) => setBasicAuthUsername(e.target.value)}
                    fullWidth
                  />
                  <TextField
                    size="small"
                    label="Password"
                    type="password"
                    value={basicAuthPassword}
                    onChange={(e) => setBasicAuthPassword(e.target.value)}
                    fullWidth
                  />
                </>
              )}
            </Box>
          </AccordionDetails>
        </Accordion>


        {/* API Help Button */}
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
          <Button
            variant="outlined"
            startIcon={<HelpIcon />}
            onClick={() => setHelpModalOpen(true)}
            size="small"
          >
            API Documentation & Examples
          </Button>
        </Box>

        {/* Output Info */}
        {showIO && (
          <Box mt={2} p={1} bgcolor="action.hover" borderRadius={1}>
            <Typography variant="caption" color="text.secondary" fontWeight={500}>
              Outputs:
            </Typography>
            <Box mt={0.5}>
              <Chip label="trigger_data" size="small" sx={{ mr: 0.5, mb: 0.5 }} />
              <Chip label="query_params" size="small" sx={{ mr: 0.5, mb: 0.5 }} />
              <Chip label="headers" size="small" sx={{ mr: 0.5, mb: 0.5 }} />
              <Chip label="message" size="small" sx={{ mb: 0.5 }} />
            </Box>
          </Box>
        )}
      </CardContent>

      {/* No input handle - triggers are entry points */}
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

      {/* Help Modal */}
      <Dialog
        open={helpModalOpen}
        onClose={() => setHelpModalOpen(false)}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: { maxHeight: '80vh' }
        }}
      >
        <DialogTitle>
          <Box display="flex" alignItems="center" gap={1}>
            <TriggerIcon />
            API Documentation & Examples
          </Box>
        </DialogTitle>
        <DialogContent dividers>
          <Tabs value={helpTabValue} onChange={(_, newValue) => setHelpTabValue(newValue)}>
            <Tab label="Overview" />
            <Tab label="cURL" />
            <Tab label="JavaScript" />
            <Tab label="Python" />
            <Tab label="Outputs" />
          </Tabs>

          {/* Overview Tab */}
          {helpTabValue === 0 && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="h6" gutterBottom>How to Use This Trigger</Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                This node creates an external API endpoint that can be called by other systems to trigger workflow execution.
              </Typography>
              <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>Endpoint Configuration:</Typography>
                <Typography variant="body2"><strong>URL:</strong> <code>{getEndpointUrl()}</code></Typography>
                <Typography variant="body2"><strong>Methods:</strong> {httpMethods.join(', ')}</Typography>
                <Typography variant="body2"><strong>Auth:</strong> {authenticationType === 'none' ? 'None' : authenticationType.replace('_', ' ')}</Typography>
              </Paper>
            </Box>
          )}

          {/* cURL Tab */}
          {helpTabValue === 1 && (
            <Box sx={{ mt: 2 }}>
              <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
                <Typography variant="h6">cURL Example</Typography>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<CopyIcon />}
                  onClick={() => copyToClipboard(getCurlExample())}
                >
                  Copy
                </Button>
              </Box>
              <Paper variant="outlined" sx={{ p: 2, bgcolor: theme => theme.palette.mode === 'dark' ? 'grey.900' : 'grey.50' }}>
                <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', color: theme => theme.palette.mode === 'dark' ? 'grey.100' : 'grey.800' }}>
                  {getCurlExample()}
                </Typography>
              </Paper>
            </Box>
          )}

          {/* JavaScript Tab */}
          {helpTabValue === 2 && (
            <Box sx={{ mt: 2 }}>
              <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
                <Typography variant="h6">JavaScript Example</Typography>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<CopyIcon />}
                  onClick={() => copyToClipboard(getJavaScriptExample())}
                >
                  Copy
                </Button>
              </Box>
              <Paper variant="outlined" sx={{ p: 2, bgcolor: theme => theme.palette.mode === 'dark' ? 'grey.900' : 'grey.50' }}>
                <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', color: theme => theme.palette.mode === 'dark' ? 'grey.100' : 'grey.800' }}>
                  {getJavaScriptExample()}
                </Typography>
              </Paper>
            </Box>
          )}

          {/* Python Tab */}
          {helpTabValue === 3 && (
            <Box sx={{ mt: 2 }}>
              <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
                <Typography variant="h6">Python Example</Typography>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<CopyIcon />}
                  onClick={() => copyToClipboard(getPythonExample())}
                >
                  Copy
                </Button>
              </Box>
              <Paper variant="outlined" sx={{ p: 2, bgcolor: theme => theme.palette.mode === 'dark' ? 'grey.900' : 'grey.50' }}>
                <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', color: theme => theme.palette.mode === 'dark' ? 'grey.100' : 'grey.800' }}>
                  {getPythonExample()}
                </Typography>
              </Paper>
            </Box>
          )}

          {/* Outputs Tab */}
          {helpTabValue === 4 && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="h6" gutterBottom>Available Output Data</Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                This node provides the following outputs that can be used by connected nodes:
              </Typography>
              <Box display="flex" flexDirection="column" gap={2}>
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom color="primary">trigger_data</Typography>
                  <Typography variant="body2" color="text.secondary">Complete request body data sent to the trigger endpoint</Typography>
                </Paper>
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom color="primary">query_params</Typography>
                  <Typography variant="body2" color="text.secondary">URL query parameters (e.g., ?user_id=123&action=process)</Typography>
                </Paper>
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom color="primary">headers</Typography>
                  <Typography variant="body2" color="text.secondary">HTTP request headers (excluding authentication headers)</Typography>
                </Paper>
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom color="primary">message</Typography>
                  <Typography variant="body2" color="text.secondary">Extracted user message/instruction from request data</Typography>
                </Paper>
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom color="primary">formatted_query</Typography>
                  <Typography variant="body2" color="text.secondary">Agent-ready query constructed from request data</Typography>
                </Paper>
              </Box>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setHelpModalOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Card>
  );
};

export default TriggerNode;