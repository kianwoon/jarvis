import React, { useState, useEffect } from 'react';
import { Handle, Position } from 'reactflow';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box, 
  TextField,
  FormControl,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Alert,
  IconButton,
  Tooltip,
  Paper,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tab,
  Tabs,
  Switch,
  FormControlLabel
} from '@mui/material';
import { 
  Language as APIIcon,
  ExpandMore as ExpandMoreIcon,
  Http as HttpIcon,
  Security as SecurityIcon,
  Settings as SettingsIcon,
  Code as CodeIcon,
  PlayArrow as TestIcon,
  ContentCopy as CopyIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import ReactJson from 'react-json-view';
import PortalSelect from './PortalSelect';

interface APINodeProps {
  data: {
    label?: string;
    base_url?: string;
    endpoint_path?: string;
    http_method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
    authentication_type?: 'none' | 'api_key' | 'bearer_token' | 'basic_auth' | 'custom_header';
    auth_header_name?: string;
    auth_token?: string;
    basic_auth_username?: string;
    basic_auth_password?: string;
    request_schema?: any;
    response_schema?: any;
    timeout?: number;
    retry_count?: number;
    rate_limit?: number;
    custom_headers?: any;
    response_transformation?: string;
    error_handling?: 'throw' | 'return_null' | 'return_error' | 'retry';
    enable_mcp_tool?: boolean;
    tool_description?: string;
    executionData?: {
      status?: 'idle' | 'running' | 'success' | 'error';
      response?: any;
      error?: string;
      duration?: number;
      status_code?: number;
    };
  };
  id: string;
  updateNodeData?: (nodeId: string, newData: any) => void;
  showIO?: boolean;
}

const APINode: React.FC<APINodeProps> = ({ data, id, updateNodeData, showIO = true }) => {
  const [endpointExpanded, setEndpointExpanded] = useState(false);
  const [authExpanded, setAuthExpanded] = useState(false);
  const [schemaExpanded, setSchemaExpanded] = useState(false);
  const [advancedExpanded, setAdvancedExpanded] = useState(false);
  const [testModalOpen, setTestModalOpen] = useState(false);
  const [testTabValue, setTestTabValue] = useState(0);
  const [testResult, setTestResult] = useState<any>(null);
  const [testLoading, setTestLoading] = useState(false);
  
  const [label, setLabel] = useState(data.label || 'APINode');
  const [baseUrl, setBaseUrl] = useState(data.base_url || '');
  const [endpointPath, setEndpointPath] = useState(data.endpoint_path || '');
  const [httpMethod, setHttpMethod] = useState(data.http_method || 'GET');
  const [authenticationType, setAuthenticationType] = useState(data.authentication_type || 'none');
  const [authHeaderName, setAuthHeaderName] = useState(data.auth_header_name || 'X-API-Key');
  const [authToken, setAuthToken] = useState(data.auth_token || '');
  const [basicAuthUsername, setBasicAuthUsername] = useState(data.basic_auth_username || '');
  const [basicAuthPassword, setBasicAuthPassword] = useState(data.basic_auth_password || '');
  const [requestSchema, setRequestSchema] = useState(data.request_schema || {
    type: "object",
    properties: {
      query: {
        type: "string",
        description: "Search query parameter"
      }
    },
    required: ["query"]
  });
  const [responseSchema, setResponseSchema] = useState(data.response_schema || {
    type: "object",
    properties: {
      data: {
        type: "object",
        description: "Response data"
      }
    }
  });
  const [timeout, setTimeout] = useState(data.timeout || 30);
  const [retryCount, setRetryCount] = useState(data.retry_count || 3);
  const [rateLimit, setRateLimit] = useState(data.rate_limit || 60);
  const [customHeaders, setCustomHeaders] = useState(data.custom_headers || {});
  const [responseTransformation, setResponseTransformation] = useState(data.response_transformation || '');
  const [errorHandling, setErrorHandling] = useState(data.error_handling || 'throw');
  const [enableMcpTool, setEnableMcpTool] = useState(data.enable_mcp_tool !== false);
  const [toolDescription, setToolDescription] = useState(data.tool_description || '');

  // Update parent node data when local state changes
  useEffect(() => {
    if (updateNodeData) {
      updateNodeData(id, {
        type: 'APINode',
        label,
        base_url: baseUrl,
        endpoint_path: endpointPath,
        http_method: httpMethod,
        authentication_type: authenticationType,
        auth_header_name: authHeaderName,
        auth_token: authToken,
        basic_auth_username: basicAuthUsername,
        basic_auth_password: basicAuthPassword,
        request_schema: requestSchema,
        response_schema: responseSchema,
        timeout,
        retry_count: retryCount,
        rate_limit: rateLimit,
        custom_headers: customHeaders,
        response_transformation: responseTransformation,
        error_handling: errorHandling,
        enable_mcp_tool: enableMcpTool,
        tool_description: toolDescription,
        // Only preserve serializable execution data
        executionData: data.executionData ? {
          status: data.executionData.status,
          response: data.executionData.response,
          error: data.executionData.error,
          duration: data.executionData.duration,
          status_code: data.executionData.status_code
        } : undefined
      });
    }
  }, [label, baseUrl, endpointPath, httpMethod, authenticationType, authHeaderName, authToken, 
      basicAuthUsername, basicAuthPassword, requestSchema, responseSchema, timeout, retryCount, 
      rateLimit, customHeaders, responseTransformation, errorHandling, enableMcpTool, toolDescription, data.executionData]);

  const getStatusColor = () => {
    switch (data.executionData?.status) {
      case 'running': return '#ff9800';
      case 'success': return '#4caf50';
      case 'error': return '#f44336';
      default: return '#10B981';
    }
  };

  const getMethodColor = () => {
    switch (httpMethod) {
      case 'GET': return '#4caf50';
      case 'POST': return '#2196f3';
      case 'PUT': return '#ff9800';
      case 'DELETE': return '#f44336';
      case 'PATCH': return '#9c27b0';
      default: return '#757575';
    }
  };

  const getFullUrl = () => {
    const cleanBaseUrl = baseUrl.replace(/\/$/, '');
    const cleanPath = endpointPath.startsWith('/') ? endpointPath : `/${endpointPath}`;
    return `${cleanBaseUrl}${cleanPath}`;
  };

  const testAPI = async () => {
    setTestLoading(true);
    try {
      // This would be replaced with actual API testing logic
      const testParams = { query: "test" };
      const url = getFullUrl();
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setTestResult({
        success: true,
        status: 200,
        response: { data: "Test response", timestamp: new Date().toISOString() },
        duration: 1000
      });
    } catch (error) {
      setTestResult({
        success: false,
        error: error.message,
        status: 500
      });
    } finally {
      setTestLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const handleRequestSchemaEdit = (edit: any) => {
    setRequestSchema(edit.updated_src);
  };

  const handleResponseSchemaEdit = (edit: any) => {
    setResponseSchema(edit.updated_src);
  };

  const handleCustomHeadersEdit = (edit: any) => {
    setCustomHeaders(edit.updated_src);
  };

  const hasExecutionData = data.executionData && data.executionData.status !== 'idle';

  return (
    <Card 
      sx={{ 
        minWidth: 380,
        maxWidth: 480,
        border: `2px solid ${getStatusColor()}`,
        borderRadius: 2,
        bgcolor: theme => theme.palette.mode === 'dark' ? 'rgba(16, 185, 129, 0.08)' : 'rgba(16, 185, 129, 0.05)',
        position: 'relative',
        overflow: 'visible',
        '&:hover': {
          boxShadow: theme => `0 0 20px ${theme.palette.mode === 'dark' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(16, 185, 129, 0.2)'}`,
        }
      }}
    >
      <CardContent sx={{ pb: 1 }}>
        {/* Header */}
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          <APIIcon sx={{ color: getStatusColor() }} />
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
          <Tooltip title="Universal REST API adapter">
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
              API call {data.executionData?.status} 
              {data.executionData?.status_code && ` (${data.executionData.status_code})`}
              {data.executionData?.duration && ` - ${data.executionData.duration}ms`}
            </Typography>
            {data.executionData?.error && (
              <Typography variant="caption" display="block">
                Error: {data.executionData.error}
              </Typography>
            )}
          </Alert>
        )}

        {/* Quick Summary */}
        <Box sx={{ mb: 2 }}>
          <Box display="flex" alignItems="center" gap={1} mb={1}>
            <Chip 
              label={httpMethod} 
              size="small" 
              sx={{ 
                bgcolor: getMethodColor(), 
                color: 'white',
                fontWeight: 500,
                minWidth: 60
              }} 
            />
            <Typography variant="body2" sx={{ 
              fontFamily: 'monospace',
              fontSize: '0.85rem',
              color: 'text.secondary'
            }}>
              {getFullUrl() || 'Configure endpoint URL'}
            </Typography>
            {baseUrl && endpointPath && (
              <IconButton 
                size="small" 
                onClick={() => copyToClipboard(getFullUrl())}
                sx={{ ml: 'auto' }}
              >
                <CopyIcon sx={{ fontSize: 14 }} />
              </IconButton>
            )}
          </Box>
          <Box display="flex" alignItems="center" gap={1}>
            <Chip 
              label={authenticationType === 'none' ? 'No Auth' : authenticationType.replace('_', ' ')} 
              size="small" 
              variant="outlined"
            />
            {enableMcpTool && (
              <Chip 
                label="MCP Tool" 
                size="small" 
                color="primary"
                variant="outlined"
              />
            )}
          </Box>
        </Box>

        {/* Endpoint Configuration */}
        <Accordion 
          expanded={endpointExpanded} 
          onChange={(_, isExpanded) => setEndpointExpanded(isExpanded)}
          sx={{ mb: 1 }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box display="flex" alignItems="center" gap={1}>
              <HttpIcon sx={{ fontSize: 20 }} />
              <Typography variant="body2" fontWeight={500}>Endpoint Configuration</Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Box display="flex" flexDirection="column" gap={2}>
              <TextField
                fullWidth
                size="small"
                label="Base URL"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="https://api.example.com"
                helperText="Root URL of the API service"
              />
              <TextField
                fullWidth
                size="small"
                label="Endpoint Path"
                value={endpointPath}
                onChange={(e) => setEndpointPath(e.target.value)}
                placeholder="/v1/search"
                helperText="Specific API endpoint path"
              />
              <PortalSelect
                value={httpMethod}
                onChange={(value) => setHttpMethod(value as string)}
                label="HTTP Method"
                options={[
                  { value: 'GET', label: 'GET' },
                  { value: 'POST', label: 'POST' },
                  { value: 'PUT', label: 'PUT' },
                  { value: 'DELETE', label: 'DELETE' },
                  { value: 'PATCH', label: 'PATCH' }
                ]}
                size="small"
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
            <Box display="flex" alignItems="center" gap={1}>
              <SecurityIcon sx={{ fontSize: 20 }} />
              <Typography variant="body2" fontWeight={500}>Authentication</Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Box display="flex" flexDirection="column" gap={2}>
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

              {(authenticationType === 'api_key' || authenticationType === 'custom_header') && (
                <>
                  <TextField
                    size="small"
                    label="Header Name"
                    value={authHeaderName}
                    onChange={(e) => setAuthHeaderName(e.target.value)}
                    fullWidth
                  />
                  <TextField
                    size="small"
                    label="API Key / Token"
                    value={authToken}
                    onChange={(e) => setAuthToken(e.target.value)}
                    type="password"
                    fullWidth
                  />
                </>
              )}

              {authenticationType === 'bearer_token' && (
                <TextField
                  size="small"
                  label="Bearer Token"
                  value={authToken}
                  onChange={(e) => setAuthToken(e.target.value)}
                  type="password"
                  fullWidth
                />
              )}

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
                    value={basicAuthPassword}
                    onChange={(e) => setBasicAuthPassword(e.target.value)}
                    type="password"
                    fullWidth
                  />
                </>
              )}
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Schema Configuration */}
        <Accordion 
          expanded={schemaExpanded} 
          onChange={(_, isExpanded) => setSchemaExpanded(isExpanded)}
          sx={{ mb: 1 }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box display="flex" alignItems="center" gap={1}>
              <CodeIcon sx={{ fontSize: 20 }} />
              <Typography variant="body2" fontWeight={500}>Request/Response Schema</Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Box display="flex" flexDirection="column" gap={2}>
              <Box>
                <Typography variant="body2" gutterBottom fontWeight={500}>
                  Request Schema (for LLM):
                </Typography>
                <Paper variant="outlined" sx={{ p: 1, maxHeight: 200, overflow: 'auto' }}>
                  <ReactJson
                    src={requestSchema}
                    theme={localStorage.getItem('theme') === 'dark' ? 'monokai' : 'rjv-default'}
                    onEdit={handleRequestSchemaEdit}
                    onAdd={handleRequestSchemaEdit}
                    onDelete={handleRequestSchemaEdit}
                    displayDataTypes={false}
                    displayObjectSize={false}
                    enableClipboard={true}
                    style={{ fontSize: '0.85rem' }}
                  />
                </Paper>
              </Box>

              <Box>
                <Typography variant="body2" gutterBottom fontWeight={500}>
                  Response Schema (expected):
                </Typography>
                <Paper variant="outlined" sx={{ p: 1, maxHeight: 200, overflow: 'auto' }}>
                  <ReactJson
                    src={responseSchema}
                    theme={localStorage.getItem('theme') === 'dark' ? 'monokai' : 'rjv-default'}
                    onEdit={handleResponseSchemaEdit}
                    onAdd={handleResponseSchemaEdit}
                    onDelete={handleResponseSchemaEdit}
                    displayDataTypes={false}
                    displayObjectSize={false}
                    enableClipboard={true}
                    style={{ fontSize: '0.85rem' }}
                  />
                </Paper>
              </Box>
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Advanced Settings */}
        <Accordion 
          expanded={advancedExpanded} 
          onChange={(_, isExpanded) => setAdvancedExpanded(isExpanded)}
          sx={{ mb: 1 }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box display="flex" alignItems="center" gap={1}>
              <SettingsIcon sx={{ fontSize: 20 }} />
              <Typography variant="body2" fontWeight={500}>Advanced Settings</Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Box display="flex" flexDirection="column" gap={2}>
              <Box display="flex" gap={2}>
                <TextField
                  size="small"
                  type="number"
                  label="Timeout (seconds)"
                  value={timeout}
                  onChange={(e) => setTimeout(parseInt(e.target.value) || 30)}
                  sx={{ flex: 1 }}
                />
                <TextField
                  size="small"
                  type="number"
                  label="Retry Count"
                  value={retryCount}
                  onChange={(e) => setRetryCount(parseInt(e.target.value) || 3)}
                  sx={{ flex: 1 }}
                />
              </Box>

              <TextField
                size="small"
                type="number"
                label="Rate Limit (requests/minute)"
                value={rateLimit}
                onChange={(e) => setRateLimit(parseInt(e.target.value) || 60)}
                fullWidth
              />

              <PortalSelect
                value={errorHandling}
                onChange={(value) => setErrorHandling(value as string)}
                label="Error Handling"
                options={[
                  { value: 'throw', label: 'Throw Error' },
                  { value: 'return_null', label: 'Return Null' },
                  { value: 'return_error', label: 'Return Error Object' },
                  { value: 'retry', label: 'Retry with Backoff' }
                ]}
                size="small"
                fullWidth
              />

              <Box>
                <Typography variant="body2" gutterBottom fontWeight={500}>
                  Custom Headers:
                </Typography>
                <Paper variant="outlined" sx={{ p: 1, maxHeight: 150, overflow: 'auto' }}>
                  <ReactJson
                    src={customHeaders}
                    theme={localStorage.getItem('theme') === 'dark' ? 'monokai' : 'rjv-default'}
                    onEdit={handleCustomHeadersEdit}
                    onAdd={handleCustomHeadersEdit}
                    onDelete={handleCustomHeadersEdit}
                    displayDataTypes={false}
                    displayObjectSize={false}
                    enableClipboard={true}
                    style={{ fontSize: '0.85rem' }}
                  />
                </Paper>
              </Box>

              <TextField
                size="small"
                multiline
                rows={3}
                label="Response Transformation (JavaScript)"
                value={responseTransformation}
                onChange={(e) => setResponseTransformation(e.target.value)}
                placeholder="// Transform the response&#10;// return response.data.items;"
                fullWidth
              />
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* MCP Tool Settings */}
        <Box sx={{ mt: 2, p: 2, bgcolor: 'action.hover', borderRadius: 1 }}>
          <FormControlLabel
            control={
              <Switch
                checked={enableMcpTool}
                onChange={(e) => setEnableMcpTool(e.target.checked)}
                size="small"
              />
            }
            label="Enable as MCP Tool for connected Agent nodes"
          />
          {enableMcpTool && (
            <TextField
              size="small"
              multiline
              rows={2}
              label="Tool Description for LLM"
              value={toolDescription}
              onChange={(e) => setToolDescription(e.target.value)}
              placeholder="This tool allows you to search for information using the configured API endpoint."
              fullWidth
              sx={{ mt: 1 }}
            />
          )}
        </Box>

        {/* Test API Button */}
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
          <Button
            variant="outlined"
            startIcon={<TestIcon />}
            onClick={() => setTestModalOpen(true)}
            disabled={!baseUrl || !endpointPath}
            size="small"
          >
            Test API
          </Button>
        </Box>

        {/* Output Info */}
        {showIO && (
          <Box mt={2} p={1} bgcolor="action.hover" borderRadius={1}>
            <Typography variant="caption" color="text.secondary" fontWeight={500}>
              Outputs:
            </Typography>
            <Box mt={0.5}>
              <Chip label="response" size="small" sx={{ mr: 0.5, mb: 0.5 }} />
              <Chip label="status" size="small" sx={{ mr: 0.5, mb: 0.5 }} />
              <Chip label="headers" size="small" sx={{ mr: 0.5, mb: 0.5 }} />
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

      {/* Test API Modal */}
      <Dialog
        open={testModalOpen}
        onClose={() => setTestModalOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Test API Endpoint</DialogTitle>
        <DialogContent dividers>
          <Box display="flex" flexDirection="column" gap={2}>
            <Typography variant="body2">
              Test your API configuration with sample parameters:
            </Typography>
            
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>Request:</Typography>
              <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                {httpMethod} {getFullUrl()}
              </Typography>
              {authenticationType !== 'none' && (
                <Typography variant="body2" sx={{ fontFamily: 'monospace', mt: 1 }}>
                  {authHeaderName}: {authToken ? '***' : 'Not configured'}
                </Typography>
              )}
            </Paper>

            <Button
              variant="contained"
              onClick={testAPI}
              disabled={testLoading || !baseUrl || !endpointPath}
              startIcon={testLoading ? null : <TestIcon />}
            >
              {testLoading ? 'Testing...' : 'Test API'}
            </Button>

            {testResult && (
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Test Result:
                </Typography>
                <ReactJson
                  src={testResult}
                  theme={localStorage.getItem('theme') === 'dark' ? 'monokai' : 'rjv-default'}
                  displayDataTypes={false}
                  displayObjectSize={false}
                  enableClipboard={true}
                  collapsed={false}
                  style={{ fontSize: '0.85rem' }}
                />
              </Paper>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTestModalOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Card>
  );
};

export default APINode;