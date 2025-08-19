import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Chip,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Grid,
  Divider,
  Card,
  CardContent,
  CardHeader,
  Tabs,
  Tab,
  Menu,
  MenuItem as MuiMenuItem,
  Snackbar,
  CircularProgress
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  Healing as HealthIcon,
  ExpandMore as ExpandMoreIcon,
  Computer as ServerIcon,
  Code as CommandIcon,
  Web as RemoteIcon,
  Description as ManifestIcon,
  Security as SecurityIcon,
  Cached as CacheIcon,
  MoreVert as MoreVertIcon,
  PlayCircle as PlayAllIcon,
  StopCircle as StopAllIcon,
  RestartAlt as RestartIcon,
  Terminal as TerminalIcon,
  CloudQueue as CloudIcon,
  FiberManualRecord as StatusDotIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  CheckCircle as SuccessIcon,
  NetworkCheck as NetworkCheckIcon,
  Explore as ExploreIcon,
  Cloud as CloudManagedIcon,
  Computer as LocalIcon,
  Link as LinkIcon,
  Info as InfoIcon,
  Help as HelpIcon,
  CheckCircle,
  Visibility,
  VisibilityOff,
  Lock as LockIcon,
  Web
} from '@mui/icons-material';
import MCPLogViewer from './MCPLogViewer';

interface MCPServer {
  id?: number;
  name: string;
  config_type: 'manifest' | 'command' | 'remote_http' | 'http';
  communication_protocol?: 'stdio' | 'http' | 'sse' | 'websocket';
  is_active: boolean;
  
  // Manifest-based config
  manifest_url?: string;
  hostname?: string;
  api_key?: string;
  oauth_credentials?: {
    client_id: string;
    client_secret: string;
    access_token?: string;
    refresh_token?: string;
  };
  
  // Command-based config
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  working_directory?: string;
  restart_policy?: 'on-failure' | 'always' | 'never';
  max_restarts?: number;
  
  // Remote HTTP config
  remote_config?: {
    server_url: string;
    transport_type: 'http' | 'sse';
    auth_headers?: Record<string, string>;
    client_info?: {
      name: string;
      version: string;
    };
    connection_timeout?: number;
  };
  
  // Enhanced error handling
  enhanced_error_handling_config?: {
    enabled: boolean;
    max_tool_retries: number;
    retry_base_delay: number;
    retry_max_delay: number;
    retry_backoff_multiplier: number;
    timeout_seconds: number;
    enable_circuit_breaker: boolean;
    circuit_failure_threshold: number;
    circuit_recovery_timeout: number;
  };
  
  // Auth refresh config
  auth_refresh_config?: {
    enabled: boolean;
    server_type: 'custom' | 'gmail' | 'outlook' | 'jira';
    auth_type: 'oauth2' | 'api_key';
    refresh_endpoint?: string;
    refresh_method: 'POST' | 'GET';
    refresh_headers?: Record<string, string>;
    refresh_data_template?: Record<string, any>;
    token_expiry_buffer_minutes: number;
  };
  
  // Runtime status
  is_running?: boolean;
  health_status?: 'healthy' | 'unhealthy' | 'unknown';
  process_id?: number;
  restart_count?: number;
  last_health_check?: string;
}

interface MCPTool {
  id: number;
  name: string;
  endpoint: string;
  is_active: boolean;
  server_id?: number;
}

interface MCPServerManagerProps {
  data: MCPServer[];
  tools?: MCPTool[];
  onChange: (data: MCPServer[]) => void;
  onRefresh: () => void;
}

const MCPServerManager: React.FC<MCPServerManagerProps> = ({
  data,
  tools,
  onChange,
  onRefresh
}) => {
  const [selectedTab, setSelectedTab] = useState(0);
  const [editDialog, setEditDialog] = useState(false);
  const [editingServer, setEditingServer] = useState<MCPServer | null>(null);
  const [viewDialog, setViewDialog] = useState(false);
  const [viewingServer, setViewingServer] = useState<MCPServer | null>(null);
  const [actionMenuAnchor, setActionMenuAnchor] = useState<null | HTMLElement>(null);
  const [actionMenuServer, setActionMenuServer] = useState<MCPServer | null>(null);
  const [performingBatchAction, setPerformingBatchAction] = useState(false);
  const [logViewerOpen, setLogViewerOpen] = useState(false);
  const [logViewerConfig, setLogViewerConfig] = useState<{serverName: string, serverId?: number | string, isExternal?: boolean}>({ serverName: 'MCP Server' });
  const [batchActionResults, setBatchActionResults] = useState<any>(null);
  const [snackbar, setSnackbar] = useState<{open: boolean, message: string, severity: 'success' | 'error' | 'info' | 'warning'}>({
    open: false,
    message: '',
    severity: 'info'
  });
  

  // State for JSON editing with validation
  const [jsonEditingState, setJsonEditingState] = useState<{
    argsText: string;
    argsError: string | null;
    envText: string;
    envError: string | null;
  }>({
    argsText: '',
    argsError: null,
    envText: '',
    envError: null
  });
  
  // State for managing sensitive environment variables display
  const [showSensitiveEnv, setShowSensitiveEnv] = useState<Record<string, boolean>>({});
  const [loadingEnv, setLoadingEnv] = useState(false);

  // Helper function to check if an environment variable key is sensitive
  const isSensitiveKey = (key: string): boolean => {
    const sensitiveKeywords = [
      'TOKEN', 'KEY', 'SECRET', 'PASSWORD', 'CREDENTIAL', 'AUTH',
      'API_KEY', 'ACCESS_TOKEN', 'REFRESH_TOKEN', 'CLIENT_SECRET',
      'PRIVATE', 'CERT', 'CERTIFICATE', 'JIRA_TOKEN', 'MS_GRAPH_TOKEN'
    ];
    const upperKey = key.toUpperCase();
    return sensitiveKeywords.some(keyword => upperKey.includes(keyword));
  };

  // Helper function to mask a value
  const maskValue = (value: string): string => {
    if (!value) return '';
    if (value.length > 8) {
      return value.substring(0, 4) + '•'.repeat(12);
    }
    return '•'.repeat(12);
  };

  const getServerTypeIcon = (type: string) => {
    switch (type) {
      case 'manifest':
        return <ManifestIcon />;
      case 'command':
        return <CommandIcon />;
      case 'http':
        return <ServerIcon />;
      case 'remote_http':
        return <RemoteIcon />;
      default:
        return <ServerIcon />;
    }
  };

  const getStatusColor = (status?: string) => {
    switch (status) {
      case 'healthy':
      case 'running':
        return 'success';
      case 'unhealthy':
      case 'error':
        return 'error';
      case 'stopped':
      case 'offline':
        return 'default';
      case 'unknown':
        return 'warning';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (status?: string) => {
    switch (status) {
      case 'healthy':
      case 'running':
        return <SuccessIcon fontSize="small" color="success" />;
      case 'unhealthy':
      case 'error':
        return <ErrorIcon fontSize="small" color="error" />;
      case 'stopped':
      case 'offline':
        return <StatusDotIcon fontSize="small" color="disabled" />;
      case 'unknown':
        return <WarningIcon fontSize="small" color="warning" />;
      default:
        return <StatusDotIcon fontSize="small" />;
    }
  };

  const showNotification = (message: string, severity: 'success' | 'error' | 'info' = 'info') => {
    setSnackbar({ open: true, message, severity });
  };

  // Helper function to validate and update JSON fields
  const handleJsonFieldChange = (
    fieldName: 'args' | 'env',
    newValue: string,
    isArray: boolean = false
  ) => {
    const stateKey = fieldName === 'args' ? 'argsText' : 'envText';
    const errorKey = fieldName === 'args' ? 'argsError' : 'envError';
    
    // Update the text state immediately for responsive typing
    setJsonEditingState(prev => ({
      ...prev,
      [stateKey]: newValue,
      [errorKey]: null
    }));
    
    // Try to parse and update the server state
    try {
      const parsed = JSON.parse(newValue);
      
      // Validate the expected type
      if (isArray && !Array.isArray(parsed)) {
        setJsonEditingState(prev => ({
          ...prev,
          [errorKey]: 'Expected a JSON array'
        }));
        return;
      }
      
      if (!isArray && (typeof parsed !== 'object' || Array.isArray(parsed) || parsed === null)) {
        setJsonEditingState(prev => ({
          ...prev,
          [errorKey]: 'Expected a JSON object'
        }));
        return;
      }
      
      // Update the server state with valid JSON
      if (editingServer) {
        setEditingServer({
          ...editingServer,
          [fieldName]: parsed
        });
      }
      
    } catch (error) {
      // Only show error if the field is not empty (allow empty state for deletion)
      if (newValue.trim() !== '') {
        setJsonEditingState(prev => ({
          ...prev,
          [errorKey]: 'Invalid JSON format'
        }));
      }
    }
  };

  const getNewServerTemplate = (type: 'manifest' | 'command' | 'remote_http' | 'http'): MCPServer => {
    const base = {
      name: '',
      config_type: type,
      is_active: true,
      enhanced_error_handling_config: {
        enabled: true,
        max_tool_retries: 3,
        retry_base_delay: 1.0,
        retry_max_delay: 60.0,
        retry_backoff_multiplier: 2.0,
        timeout_seconds: 30,
        enable_circuit_breaker: true,
        circuit_failure_threshold: 5,
        circuit_recovery_timeout: 60
      },
      auth_refresh_config: {
        enabled: false,
        server_type: 'custom' as const,
        auth_type: 'oauth2' as const,
        refresh_method: 'POST' as const,
        token_expiry_buffer_minutes: 5
      }
    };

    switch (type) {
      case 'manifest':
        return {
          ...base,
          communication_protocol: 'http' as const,  // Manifest servers typically use HTTP
          manifest_url: '',
          hostname: '',
          api_key: ''
        };
      case 'command':
        return {
          ...base,
          communication_protocol: 'stdio' as const,  // Default to stdio for command servers
          command: '',
          args: [],
          env: {},
          working_directory: '',
          restart_policy: 'on-failure',
          max_restarts: 3
        };
      case 'http':
        return {
          ...base,
          communication_protocol: 'http' as const,  // HTTP servers use HTTP
          hostname: '',
          api_key: '',
          env: {}  // Initialize empty env for HTTP servers
        };
      case 'remote_http':
        return {
          ...base,
          communication_protocol: 'http' as const,  // Default to HTTP, can be changed to SSE
          remote_config: {
            server_url: '',
            transport_type: 'http',
            auth_headers: {},
            client_info: {
              name: 'Jarvis AI',
              version: '1.0.0'
            },
            connection_timeout: 30
          }
        };
    }
  };

  const handleAdd = (type: 'manifest' | 'command' | 'remote_http' | 'http') => {
    const newServer = getNewServerTemplate(type);
    setEditingServer(newServer);
    
    // Initialize JSON editing state for new server
    setJsonEditingState({
      argsText: JSON.stringify(newServer.args || [], null, 2),
      argsError: null,
      envText: JSON.stringify(newServer.env || {}, null, 2),
      envError: null
    });
    
    setEditDialog(true);
  };

  const handleEdit = async (server: MCPServer) => {
    setEditingServer({ ...server });
    
    // For environment variables, fetch unmasked values if needed
    if (server.id && server.env && Object.keys(server.env).length > 0) {
      // Check if any values are masked
      const hasMaskedValues = Object.values(server.env).some(v => 
        typeof v === 'string' && v.includes('•')
      );
      
      if (hasMaskedValues) {
        setLoadingEnv(true);
        try {
          const response = await fetch(`/api/v1/mcp/servers/${server.id}/env`);
          if (response.ok) {
            const data = await response.json();
            // Update the editing server with unmasked env
            setEditingServer(prev => prev ? { ...prev, env: data.env } : null);
            // Initialize JSON editing state with unmasked values
            setJsonEditingState({
              argsText: JSON.stringify(server.args || [], null, 2),
              argsError: null,
              envText: JSON.stringify(data.env || {}, null, 2),
              envError: null
            });
          }
        } catch (error) {
          console.error('Failed to fetch unmasked environment variables:', error);
          // Fall back to masked values
          setJsonEditingState({
            argsText: JSON.stringify(server.args || [], null, 2),
            argsError: null,
            envText: JSON.stringify(server.env || {}, null, 2),
            envError: null
          });
        } finally {
          setLoadingEnv(false);
        }
      } else {
        // No masked values, use as is
        setJsonEditingState({
          argsText: JSON.stringify(server.args || [], null, 2),
          argsError: null,
          envText: JSON.stringify(server.env || {}, null, 2),
          envError: null
        });
      }
    } else {
      // Initialize JSON editing state with properly formatted JSON
      setJsonEditingState({
        argsText: JSON.stringify(server.args || [], null, 2),
        argsError: null,
        envText: JSON.stringify(server.env || {}, null, 2),
        envError: null
      });
    }
    
    setEditDialog(true);
  };

  const handleDelete = async (id: number) => {
    if (!confirm('Are you sure you want to delete this MCP server?')) return;
    
    try {
      const response = await fetch(`/api/v1/mcp/servers/${id}/`, { method: 'DELETE' });
      if (response.ok) {
        showNotification('Server deleted successfully', 'success');
        onRefresh();
      } else {
        showNotification('Failed to delete server', 'error');
      }
    } catch (error) {
      showNotification('Error deleting server: ' + error, 'error');
    }
  };

  const handleSave = async () => {
    if (!editingServer) return;
    
    try {
      const endpoint = '/api/v1/mcp/servers/';
      const method = editingServer.id ? 'PUT' : 'POST';
      const url = editingServer.id ? `${endpoint}${editingServer.id}/` : endpoint;
      
      const response = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editingServer)
      });
      
      if (response.ok) {
        setEditDialog(false);
        setEditingServer(null);
        showNotification('Server saved successfully', 'success');
        onRefresh();
      } else {
        showNotification('Failed to save server', 'error');
      }
    } catch (error) {
      showNotification('Error saving server: ' + error, 'error');
    }
  };


  const handleBatchAction = async (action: 'start-all' | 'stop-all' | 'refresh-health' | 'discover-tools' | 'test-connections') => {
    try {
      setPerformingBatchAction(true);
      setBatchActionResults(null);
      
      const actionMessages = {
        'start-all': 'Starting local servers...',
        'stop-all': 'Stopping local servers...',
        'refresh-health': 'Checking server health...',
        'discover-tools': 'Discovering tools...',
        'test-connections': 'Testing connections...'
      };
      
      showNotification(actionMessages[action], 'info');
      
      const response = await fetch(`/api/v1/mcp/servers/batch/${action}`, {
        method: 'POST'
      });
      
      if (response.ok) {
        const result = await response.json();
        
        // Store results for detailed display
        setBatchActionResults(result);
        
        // Create appropriate message based on action
        let message = '';
        switch (action) {
          case 'start-all':
            message = `Started: ${result.successful}/${result.total} servers`;
            break;
          case 'stop-all':
            message = `Stopped: ${result.successful}/${result.total} servers`;
            break;
          case 'refresh-health':
            message = `Health checked: ${result.successful}/${result.total} servers`;
            break;
          case 'discover-tools':
            message = `Discovered ${result.total_tools || 0} tools from ${result.successful} servers`;
            break;
          case 'test-connections':
            message = `Tested ${result.total} servers: ${result.successful} connected`;
            break;
        }
        
        showNotification(message, result.failed > 0 ? 'warning' : 'success');
        
        // Show details if there were failures
        if (result.failed > 0) {
          console.log('Batch action results:', result.results);
        }
        
        // Refresh data
        onRefresh();
      } else {
        showNotification(`Failed to ${action}`, 'error');
      }
    } catch (error) {
      showNotification(`Error during ${action}: ` + error, 'error');
    } finally {
      setPerformingBatchAction(false);
    }
  };

  const handleServerAction = async (id: number, action: 'start' | 'stop' | 'restart' | 'health' | 'refresh' | 'discover-tools') => {
    try {
      const response = await fetch(`/api/v1/mcp/servers/${id}/${action}/`, {
        method: 'POST'
      });
      
      if (response.ok) {
        const result = await response.json();
        showNotification(result.message || `${action} completed successfully`, 'success');
        if (action !== 'health') { // Don't refresh for health checks
          onRefresh();
        }
      } else {
        showNotification(`Failed to ${action} server`, 'error');
      }
    } catch (error) {
      showNotification(`Error during ${action}: ` + error, 'error');
    }
  };

  // Helper function to check if a server is local
  const isLocalServer = (server: MCPServer): boolean => {
    if (server.config_type === 'command') {
      return true; // Command servers are always local
    }
    
    if (server.config_type === 'http') {
      // Check if hostname is localhost or Docker internal
      const hostname = server.hostname || '';
      const localPatterns = ['localhost', '127.0.0.1', '0.0.0.0', 'host.docker.internal'];
      if (localPatterns.some(pattern => hostname.toLowerCase().includes(pattern))) {
        return true;
      }
      // Check if it's a Docker service name (no dots in hostname, not a full URL)
      if (!hostname.includes('.') && !hostname.startsWith('http')) {
        return true;
      }
    }
    
    if (server.config_type === 'manifest') {
      // Check manifest URL for local addresses
      const url = server.manifest_url || '';
      const hostname = server.hostname || '';
      const localPatterns = ['localhost', '127.0.0.1', '0.0.0.0', 'host.docker.internal'];
      
      if (localPatterns.some(pattern => url.toLowerCase().includes(pattern))) {
        return true;
      }
      if (localPatterns.some(pattern => hostname.toLowerCase().includes(pattern))) {
        return true;
      }
      // Docker service names
      if (hostname && !hostname.includes('.') && !hostname.startsWith('http')) {
        return true;
      }
    }
    
    // remote_http is always remote
    return false;
  };

  // Helper function to check if a server can be controlled (started/stopped)
  const canControlServer = (server: MCPServer): boolean => {
    // Command servers can always be controlled
    if (server.config_type === 'command') {
      return true;
    }
    // Local HTTP and manifest servers can potentially be controlled
    // if we have a way to start/stop them (e.g., Docker containers)
    if (isLocalServer(server) && (server.config_type === 'http' || server.config_type === 'manifest')) {
      // Check if it's a Docker service that we can control
      const hostname = server.hostname || '';
      // Docker service names or host.docker.internal indicate controllable services
      if (hostname === 'host.docker.internal' || (!hostname.includes('.') && !hostname.startsWith('http'))) {
        return true;
      }
    }
    return false;
  };

  const getServerControlBadge = (server: MCPServer) => {
    const isLocal = isLocalServer(server);
    
    if (server.config_type === 'command' || (isLocal && server.config_type !== 'remote_http')) {
      return (
        <Chip 
          label="Local" 
          size="small" 
          icon={<LocalIcon />}
          color="primary"
          sx={{ ml: 1 }}
        />
      );
    } else if (server.config_type === 'remote_http' || !isLocal) {
      return (
        <Chip 
          label="Cloud-Managed" 
          size="small" 
          icon={<CloudManagedIcon />}
          color="secondary"
          sx={{ ml: 1 }}
        />
      );
    } else {
      return null;
    }
  };

  const renderServerTable = () => (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Name</TableCell>
            <TableCell>Type</TableCell>
            <TableCell>Control</TableCell>
            <TableCell>Status</TableCell>
            <TableCell>Health</TableCell>
            <TableCell>Tools</TableCell>
            <TableCell>Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {data.map((server, index) => (
            <TableRow key={server.id || index}>
              <TableCell>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {getServerTypeIcon(server.config_type)}
                  {server.name}
                </Box>
              </TableCell>
              <TableCell>
                <Chip label={server.config_type} size="small" />
              </TableCell>
              <TableCell>
                {getServerControlBadge(server)}
              </TableCell>
              <TableCell>
                <Chip 
                  label={server.is_active ? 'Active' : 'Inactive'} 
                  color={server.is_active ? 'success' : 'default'}
                  size="small" 
                />
              </TableCell>
              <TableCell>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  {getStatusIcon(server.health_status)}
                  <Chip 
                    label={server.health_status || 'Unknown'} 
                    color={getStatusColor(server.health_status) as any}
                    size="small" 
                  />
                  {server.process_id && (
                    <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                      PID: {server.process_id}
                    </Typography>
                  )}
                </Box>
              </TableCell>
              <TableCell>
                {tools?.filter(t => t.server_id === server.id).length || 0} tools
              </TableCell>
              <TableCell>
                <IconButton size="small" onClick={() => handleEdit(server)}>
                  <EditIcon />
                </IconButton>
                <IconButton 
                  size="small" 
                  onClick={(e) => {
                    setActionMenuAnchor(e.currentTarget);
                    setActionMenuServer(server);
                  }}
                >
                  <MoreVertIcon />
                </IconButton>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );

  const handleCloseActionMenu = () => {
    setActionMenuAnchor(null);
    setActionMenuServer(null);
  };

  const handleMenuAction = async (action: string) => {
    if (!actionMenuServer) return;
    
    if (action === 'view') {
      setViewingServer(actionMenuServer);
      setViewDialog(true);
    } else if (action === 'delete') {
      await handleDelete(actionMenuServer.id!);
    } else if (action === 'logs') {
      // Open log viewer for the server
      setLogViewerConfig({
        serverName: actionMenuServer.name,
        serverId: actionMenuServer.id
      });
      setLogViewerOpen(true);
    } else {
      await handleServerAction(actionMenuServer.id!, action as any);
    }
    
    handleCloseActionMenu();
  };

  const renderEditForm = () => {
    if (!editingServer) return null;

    const updateServer = (updates: Partial<MCPServer>) => {
      setEditingServer({ ...editingServer, ...updates });
    };

    const updateNestedConfig = (key: keyof MCPServer, updates: any) => {
      setEditingServer({
        ...editingServer,
        [key]: { ...(editingServer[key] as any || {}), ...updates }
      });
    };

    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, maxHeight: '70vh', overflow: 'auto' }}>
        {/* Basic Configuration */}
        <Card>
          <CardHeader title="Basic Configuration" />
          <CardContent>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Server Name"
                  value={editingServer.name}
                  onChange={(e) => updateServer({ name: e.target.value })}
                  fullWidth
                  required
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel>Server Type</InputLabel>
                  <Select
                    value={editingServer.config_type}
                    onChange={(e) => updateServer({ config_type: e.target.value as any })}
                  >
                    <MenuItem value="manifest">Manifest-based</MenuItem>
                    <MenuItem value="command">Command-based (stdio)</MenuItem>
                    <MenuItem value="http">HTTP Server</MenuItem>
                    <MenuItem value="remote_http">Remote HTTP/SSE</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel>Communication Protocol</InputLabel>
                  <Select
                    value={editingServer.communication_protocol || 'stdio'}
                    onChange={(e) => updateServer({ communication_protocol: e.target.value as any })}
                  >
                    <MenuItem value="stdio">
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <TerminalIcon fontSize="small" />
                        stdio (Standard I/O)
                      </Box>
                    </MenuItem>
                    <MenuItem value="http">
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Web fontSize="small" />
                        HTTP
                      </Box>
                    </MenuItem>
                    <MenuItem value="sse">
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <CloudIcon fontSize="small" />
                        SSE (Server-Sent Events)
                      </Box>
                    </MenuItem>
                    <MenuItem value="websocket">
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LinkIcon fontSize="small" />
                        WebSocket
                      </Box>
                    </MenuItem>
                  </Select>
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                    {editingServer.communication_protocol === 'stdio' && 'Traditional MCP server using standard I/O streams'}
                    {editingServer.communication_protocol === 'http' && 'Server communicates via HTTP REST API'}
                    {editingServer.communication_protocol === 'sse' && 'Server uses Server-Sent Events for real-time communication'}
                    {editingServer.communication_protocol === 'websocket' && 'Server uses WebSocket for bidirectional communication'}
                  </Typography>
                </FormControl>
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={editingServer.is_active}
                      onChange={(e) => updateServer({ is_active: e.target.checked })}
                    />
                  }
                  label="Active"
                />
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        {/* Type-specific Configuration */}
        {editingServer.config_type === 'http' && (
          <Card>
            <CardHeader title="HTTP Server Configuration" />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <TextField
                    label="Hostname/URL"
                    value={editingServer.hostname || ''}
                    onChange={(e) => updateServer({ hostname: e.target.value })}
                    fullWidth
                    required
                    helperText="e.g., host.docker.internal:3001 or http://localhost:3001"
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    label="API Key (Optional)"
                    value={editingServer.api_key || ''}
                    onChange={(e) => updateServer({ api_key: e.target.value })}
                    fullWidth
                    type="password"
                    helperText="Optional API key for authentication"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        )}

        {editingServer.config_type === 'manifest' && (
          <Card>
            <CardHeader title="Manifest Configuration" />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <TextField
                    label="Manifest URL"
                    value={editingServer.manifest_url || ''}
                    onChange={(e) => updateServer({ manifest_url: e.target.value })}
                    fullWidth
                    required
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="Hostname"
                    value={editingServer.hostname || ''}
                    onChange={(e) => updateServer({ hostname: e.target.value })}
                    fullWidth
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="API Key"
                    value={editingServer.api_key || ''}
                    onChange={(e) => updateServer({ api_key: e.target.value })}
                    fullWidth
                    type="password"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        )}

        {(editingServer.config_type === 'command' || editingServer.config_type === 'http') && (
          <Card>
            <CardHeader title={editingServer.config_type === 'command' ? "Command Configuration" : "HTTP Server Advanced Configuration"} />
            <CardContent>
              <Grid container spacing={2}>
                {editingServer.config_type === 'command' && (
                  <Grid item xs={12}>
                    <TextField
                      label="Working Directory"
                      value={editingServer.working_directory || ''}
                      onChange={(e) => updateServer({ working_directory: e.target.value })}
                      fullWidth
                      helperText="Full path to the directory where the command should be executed"
                    />
                  </Grid>
                )}
                {editingServer.config_type === 'command' && (
                  <>
                    {editingServer.communication_protocol === 'http' && (
                      <Grid item xs={12}>
                        <Alert severity="info">
                          <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                            HTTP Protocol for Command Server
                          </Typography>
                          <Typography variant="body2">
                            This command will start an HTTP server. The system will:
                          </Typography>
                          <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
                            <li>Start the process using the command below</li>
                            <li>Wait for the HTTP server to be ready</li>
                            <li>Connect to it via HTTP to discover tools</li>
                          </ul>
                          <Typography variant="body2">
                            Make sure to specify the hostname/port either in the Hostname field or via environment variables.
                          </Typography>
                        </Alert>
                      </Grid>
                    )}
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Command"
                        value={editingServer.command || ''}
                        onChange={(e) => updateServer({ command: e.target.value })}
                        fullWidth
                        required
                        helperText={
                          editingServer.communication_protocol === 'http' 
                            ? "Command to start the HTTP server (e.g., npm, npx)"
                            : "The executable command (e.g., npm, python, node)"
                        }
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Arguments (comma-separated)"
                        value={editingServer.args?.join(', ') || ''}
                        onChange={(e) => {
                          const argsArray = e.target.value.split(',').map(arg => arg.trim()).filter(arg => arg);
                          updateServer({ args: argsArray });
                          setJsonEditingState(prev => ({
                            ...prev,
                            argsText: JSON.stringify(argsArray, null, 2)
                          }));
                        }}
                        fullWidth
                        helperText="Command arguments separated by commas (e.g., start, --port, 3001)"
                      />
                    </Grid>
                    {editingServer.communication_protocol === 'http' && (
                      <Grid item xs={12}>
                        <TextField
                          label="Hostname/Port (for HTTP server discovery)"
                          value={editingServer.hostname || ''}
                          onChange={(e) => updateServer({ hostname: e.target.value })}
                          fullWidth
                          helperText="Where to connect after starting the server (e.g., localhost:3000). Leave empty to use MCP_PORT env var."
                        />
                      </Grid>
                    )}
                    <Grid item xs={12} md={6}>
                      <FormControl fullWidth>
                        <InputLabel>Restart Policy</InputLabel>
                        <Select
                          value={editingServer.restart_policy || 'on-failure'}
                          onChange={(e) => updateServer({ restart_policy: e.target.value as any })}
                        >
                          <MenuItem value="on-failure">On Failure</MenuItem>
                          <MenuItem value="always">Always</MenuItem>
                          <MenuItem value="never">Never</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Max Restarts"
                        type="number"
                        value={editingServer.max_restarts || 3}
                        onChange={(e) => updateServer({ max_restarts: parseInt(e.target.value) })}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <Divider sx={{ my: 2 }} />
                      <Typography variant="subtitle2" gutterBottom>
                        Advanced Configuration (JSON Format)
                      </Typography>
                    </Grid>
                    <Grid item xs={12}>
                      <TextField
                        label="Arguments (JSON array - for complex args)"
                        value={jsonEditingState.argsText}
                        onChange={(e) => handleJsonFieldChange('args', e.target.value, true)}
                        fullWidth
                        multiline
                        rows={2}
                        error={!!jsonEditingState.argsError}
                        helperText={jsonEditingState.argsError || 'Advanced: Enter a JSON array for complex arguments, e.g., ["--config", "path/to/config.json"]'}
                      />
                    </Grid>
                  </>
                )}
                <Grid item xs={12}>
                  <Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <Typography variant="subtitle2" sx={{ flexGrow: 1 }}>
                        Environment Variables
                      </Typography>
                      {editingServer?.env && Object.keys(editingServer.env).some(isSensitiveKey) && (
                        <Chip
                          icon={<LockIcon />}
                          label="Contains sensitive data"
                          size="small"
                          color="warning"
                          variant="outlined"
                        />
                      )}
                    </Box>
                    {editingServer.config_type === 'http' && (
                      <Alert severity="info" sx={{ mb: 2 }}>
                        <Typography variant="caption">
                          <strong>Note:</strong> Environment variables are used when the HTTP server is started locally via scripts (e.g., start_mcp_server.sh).
                          Remote HTTP servers manage their own environment variables.
                        </Typography>
                      </Alert>
                    )}
                    <TextField
                      value={jsonEditingState.envText}
                      onChange={(e) => handleJsonFieldChange('env', e.target.value, false)}
                      fullWidth
                      multiline
                      rows={5}
                      error={!!jsonEditingState.envError}
                      helperText={jsonEditingState.envError || 'Enter a valid JSON object. Sensitive values (tokens, keys, passwords) will be masked in the UI for security.'}
                      sx={{
                        '& .MuiInputBase-input': {
                          fontFamily: 'monospace',
                          fontSize: '0.875rem'
                        }
                      }}
                      InputProps={{
                        endAdornment: loadingEnv && <CircularProgress size={20} />
                      }}
                    />
                    {editingServer?.env && Object.keys(editingServer.env).length > 0 && (
                      <Alert severity="info" sx={{ mt: 1 }}>
                        <Typography variant="caption">
                          💡 Tip: Sensitive environment variables (containing TOKEN, KEY, SECRET, etc.) are automatically masked when displayed in the list view.
                          The actual values are securely stored in the database and used when running the MCP server.
                        </Typography>
                      </Alert>
                    )}
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        )}

        {editingServer.config_type === 'remote_http' && (
          <>
            <Card>
              <CardHeader title="Remote HTTP Configuration" />
              <CardContent>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={8}>
                    <TextField
                      label="Server URL"
                      value={editingServer.remote_config?.server_url || ''}
                      onChange={(e) => updateNestedConfig('remote_config', { server_url: e.target.value })}
                      fullWidth
                      required
                      helperText="The URL of your cloud-hosted MCP server (e.g., https://my-mcp-server.azurewebsites.net)"
                    />
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <FormControl fullWidth>
                      <InputLabel>Transport Type</InputLabel>
                      <Select
                        value={editingServer.remote_config?.transport_type || 'http'}
                        onChange={(e) => updateNestedConfig('remote_config', { transport_type: e.target.value })}
                      >
                        <MenuItem value="http">HTTP</MenuItem>
                        <MenuItem value="sse">Server-Sent Events</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                      label="Connection Timeout (seconds)"
                      type="number"
                      value={editingServer.remote_config?.connection_timeout || 30}
                      onChange={(e) => updateNestedConfig('remote_config', { connection_timeout: parseInt(e.target.value) })}
                      fullWidth
                    />
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader title="Cloud Provider Information (Optional)" />
              <CardContent>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <FormControl fullWidth>
                      <InputLabel>Cloud Provider</InputLabel>
                      <Select
                        value={editingServer.remote_config?.cloud_provider || 'custom'}
                        onChange={(e) => updateNestedConfig('remote_config', { cloud_provider: e.target.value })}
                      >
                        <MenuItem value="custom">Custom/Self-Hosted</MenuItem>
                        <MenuItem value="aws">Amazon Web Services (AWS)</MenuItem>
                        <MenuItem value="azure">Microsoft Azure</MenuItem>
                        <MenuItem value="gcp">Google Cloud Platform</MenuItem>
                        <MenuItem value="vercel">Vercel</MenuItem>
                        <MenuItem value="netlify">Netlify</MenuItem>
                        <MenuItem value="heroku">Heroku</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                      label="Management Console URL"
                      value={editingServer.remote_config?.management_url || ''}
                      onChange={(e) => updateNestedConfig('remote_config', { management_url: e.target.value })}
                      fullWidth
                      helperText="Link to your cloud provider's management console for this server"
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <TextField
                      label="Notes / Cost Information"
                      value={editingServer.remote_config?.notes || ''}
                      onChange={(e) => updateNestedConfig('remote_config', { notes: e.target.value })}
                      fullWidth
                      multiline
                      rows={2}
                      helperText="Any notes about this server (e.g., 'Costs $5/month on AWS Lambda')"
                    />
                  </Grid>
                </Grid>
                <Alert severity="info" sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    Cloud-managed servers cannot be started or stopped from this interface. 
                    Use your cloud provider's console to manage the server lifecycle.
                  </Typography>
                </Alert>
              </CardContent>
            </Card>
          </>
        )}

        {/* Enhanced Error Handling */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">Enhanced Error Handling</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={editingServer.enhanced_error_handling_config?.enabled || false}
                      onChange={(e) => updateNestedConfig('enhanced_error_handling_config', { enabled: e.target.checked })}
                    />
                  }
                  label="Enable Enhanced Error Handling"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Max Tool Retries"
                  type="number"
                  value={editingServer.enhanced_error_handling_config?.max_tool_retries || 3}
                  onChange={(e) => updateNestedConfig('enhanced_error_handling_config', { max_tool_retries: parseInt(e.target.value) })}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Timeout (seconds)"
                  type="number"
                  value={editingServer.enhanced_error_handling_config?.timeout_seconds || 30}
                  onChange={(e) => updateNestedConfig('enhanced_error_handling_config', { timeout_seconds: parseInt(e.target.value) })}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={editingServer.enhanced_error_handling_config?.enable_circuit_breaker || false}
                      onChange={(e) => updateNestedConfig('enhanced_error_handling_config', { enable_circuit_breaker: e.target.checked })}
                    />
                  }
                  label="Enable Circuit Breaker"
                />
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>

        {/* Authentication Refresh */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">Authentication Refresh</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={editingServer.auth_refresh_config?.enabled || false}
                      onChange={(e) => updateNestedConfig('auth_refresh_config', { enabled: e.target.checked })}
                    />
                  }
                  label="Enable Auto Token Refresh"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel>Server Type</InputLabel>
                  <Select
                    value={editingServer.auth_refresh_config?.server_type || 'custom'}
                    onChange={(e) => updateNestedConfig('auth_refresh_config', { server_type: e.target.value })}
                  >
                    <MenuItem value="custom">Custom</MenuItem>
                    <MenuItem value="gmail">Gmail</MenuItem>
                    <MenuItem value="outlook">Outlook</MenuItem>
                    <MenuItem value="jira">Jira</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel>Auth Type</InputLabel>
                  <Select
                    value={editingServer.auth_refresh_config?.auth_type || 'oauth2'}
                    onChange={(e) => updateNestedConfig('auth_refresh_config', { auth_type: e.target.value })}
                  >
                    <MenuItem value="oauth2">OAuth2</MenuItem>
                    <MenuItem value="api_key">API Key</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>
      </Box>
    );
  };

  return (
    <Box>
      {/* Server Type Information Banner */}
      <Alert severity="info" sx={{ mb: 2 }}>
        <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
          Understanding MCP Server Types
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, mt: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip label="Local" size="small" icon={<LocalIcon />} color="primary" />
            <Typography variant="body2">
              Servers running on your machine or local network (includes localhost, Docker containers, and local HTTP servers). 
              Command-based servers can be started/stopped from this interface.
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip label="Cloud-Managed" size="small" icon={<CloudManagedIcon />} color="secondary" />
            <Typography variant="body2">
              Remote servers hosted in the cloud or external networks. These are managed externally and cannot be controlled from this interface.
            </Typography>
          </Box>
        </Box>
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
          Note: Servers are categorized by their location (local vs remote), not their protocol. 
          HTTP servers on localhost are considered local, while servers with external domains are cloud-managed.
        </Typography>
      </Alert>

      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="h6">MCP Server Management</Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Button 
            onClick={() => handleBatchAction('refresh-health')} 
            startIcon={<HealthIcon />}
            variant="outlined"
            disabled={performingBatchAction}
          >
            Refresh Health
          </Button>
          <Button 
            onClick={() => handleBatchAction('discover-tools')} 
            startIcon={<ExploreIcon />}
            variant="outlined"
            disabled={performingBatchAction}
          >
            Discover Tools
          </Button>
          <Button 
            onClick={() => handleBatchAction('test-connections')} 
            startIcon={<NetworkCheckIcon />}
            variant="outlined"
            disabled={performingBatchAction}
          >
            Test Connections
          </Button>
          <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />
          <Button onClick={onRefresh} startIcon={<RefreshIcon />}>
            Refresh
          </Button>
          <Button 
            onClick={() => handleAdd('manifest')} 
            variant="contained" 
            startIcon={<AddIcon />}
            size="small"
          >
            Add Manifest
          </Button>
          <Button 
            onClick={() => handleAdd('command')} 
            variant="contained" 
            startIcon={<AddIcon />}
            size="small"
          >
            Add Command
          </Button>
          <Button 
            onClick={() => handleAdd('http')} 
            variant="contained" 
            startIcon={<AddIcon />}
            size="small"
          >
            Add HTTP
          </Button>
          <Button 
            onClick={() => handleAdd('remote_http')} 
            variant="contained" 
            startIcon={<AddIcon />}
            size="small"
          >
            Add Remote
          </Button>
        </Box>
      </Box>

      {data.length === 0 ? (
        <Alert severity="info">No MCP servers configured</Alert>
      ) : (
        renderServerTable()
      )}
      
      {/* Internal MCP Tools Section */}
      {tools && tools.filter(t => t.endpoint.startsWith('internal://')).length > 0 && (
        <Box sx={{ mt: 4 }}>
          <Typography variant="h6" sx={{ mb: 2 }}>Internal MCP Services</Typography>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Name</TableCell>
                  <TableCell>Endpoint</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Type</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {tools
                  .filter(tool => tool.endpoint.startsWith('internal://'))
                  .map((tool) => (
                    <TableRow key={tool.id}>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <ServerIcon fontSize="small" />
                          <Typography variant="body2" fontWeight="bold">
                            {tool.name}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                          {tool.endpoint}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip 
                          label={tool.is_active ? 'Active' : 'Inactive'} 
                          color={tool.is_active ? 'success' : 'default'}
                          size="small" 
                        />
                      </TableCell>
                      <TableCell>
                        <Chip 
                          label="Internal Service" 
                          color="primary"
                          size="small" 
                          icon={<SecurityIcon fontSize="small" />}
                        />
                      </TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
          </TableContainer>
          <Alert severity="info" sx={{ mt: 2 }}>
            Internal MCP services are built-in system tools that cannot be modified or removed. They provide core functionality like RAG search.
          </Alert>
        </Box>
      )}

      {/* Edit Dialog */}
      <Dialog open={editDialog} onClose={() => setEditDialog(false)} maxWidth="lg" fullWidth>
        <DialogTitle>
          {editingServer?.id ? 'Edit MCP Server' : 'Add MCP Server'}
        </DialogTitle>
        <DialogContent>
          {renderEditForm()}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialog(false)}>Cancel</Button>
          <Button onClick={handleSave} variant="contained">Save</Button>
        </DialogActions>
      </Dialog>

      {/* View Dialog */}
      <Dialog open={viewDialog} onClose={() => setViewDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>MCP Server Details</DialogTitle>
        <DialogContent>
          <pre style={{ fontSize: '12px', overflow: 'auto' }}>
            {JSON.stringify(viewingServer, null, 2)}
          </pre>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setViewDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Action Menu */}
      <Menu
        anchorEl={actionMenuAnchor}
        open={Boolean(actionMenuAnchor)}
        onClose={handleCloseActionMenu}
      >
        <MuiMenuItem onClick={() => handleMenuAction('view')}>
          <SettingsIcon sx={{ mr: 1 }} fontSize="small" />
          View Details
        </MuiMenuItem>
        {actionMenuServer && canControlServer(actionMenuServer) && (
          <>
            <Divider />
            {actionMenuServer?.is_running ? (
              <>
                <MuiMenuItem onClick={() => handleMenuAction('stop')}>
                  <StopIcon sx={{ mr: 1 }} fontSize="small" color="error" />
                  Stop Server
                </MuiMenuItem>
                <MuiMenuItem onClick={() => handleMenuAction('restart')}>
                  <RestartIcon sx={{ mr: 1 }} fontSize="small" color="primary" />
                  Restart Server
                </MuiMenuItem>
              </>
            ) : (
              <MuiMenuItem onClick={() => handleMenuAction('start')}>
                <StartIcon sx={{ mr: 1 }} fontSize="small" color="success" />
                Start Server
              </MuiMenuItem>
            )}
            {actionMenuServer?.config_type === 'command' && (
              <MuiMenuItem onClick={() => handleMenuAction('logs')}>
                <TerminalIcon sx={{ mr: 1 }} fontSize="small" />
                View Logs
              </MuiMenuItem>
            )}
            <Divider />
          </>
        )}
        <MuiMenuItem onClick={() => handleMenuAction('health')}>
          <HealthIcon sx={{ mr: 1 }} fontSize="small" />
          Check Health
        </MuiMenuItem>
        <MuiMenuItem onClick={() => handleMenuAction('refresh')}>
          <RefreshIcon sx={{ mr: 1 }} fontSize="small" />
          Refresh
        </MuiMenuItem>
        <MuiMenuItem onClick={() => handleMenuAction('discover-tools')}>
          <CacheIcon sx={{ mr: 1 }} fontSize="small" />
          Discover Tools
        </MuiMenuItem>
        <Divider />
        <MuiMenuItem onClick={() => handleMenuAction('delete')}>
          <DeleteIcon sx={{ mr: 1 }} fontSize="small" color="error" />
          <Typography color="error">Delete Server</Typography>
        </MuiMenuItem>
      </Menu>

      {/* Loading Overlay */}
      {performingBatchAction && (
        <Box 
          sx={{ 
            position: 'fixed', 
            top: 0, 
            left: 0, 
            right: 0, 
            bottom: 0, 
            bgcolor: 'rgba(0,0,0,0.5)', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            zIndex: 9999
          }}
        >
          <CircularProgress color="primary" size={60} />
        </Box>
      )}

      {/* Batch Action Results Dialog */}
      {batchActionResults && (
        <Dialog 
          open={true} 
          onClose={() => setBatchActionResults(null)} 
          maxWidth="md" 
          fullWidth
        >
          <DialogTitle>
            Batch Operation Results
            <Typography variant="body2" color="text.secondary">
              Successful: {batchActionResults.successful} / Failed: {batchActionResults.failed}
            </Typography>
          </DialogTitle>
          <DialogContent>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Server</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Message</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {batchActionResults.results?.map((result: any, index: number) => (
                    <TableRow key={index}>
                      <TableCell>{result.server_name}</TableCell>
                      <TableCell>
                        <Chip 
                          label={result.server_type || 'unknown'} 
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        {result.success ? (
                          <CheckCircle color="success" fontSize="small" />
                        ) : (
                          <ErrorIcon color="error" fontSize="small" />
                        )}
                      </TableCell>
                      <TableCell>
                        {result.message}
                        {result.latency_ms && (
                          <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                            ({result.latency_ms}ms)
                          </Typography>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setBatchActionResults(null)}>Close</Button>
          </DialogActions>
        </Dialog>
      )}

      {/* Notification Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <Alert 
          onClose={() => setSnackbar({ ...snackbar, open: false })} 
          severity={snackbar.severity as 'success' | 'error' | 'info' | 'warning'}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>

      {/* Log Viewer Dialog */}
      <MCPLogViewer
        open={logViewerOpen}
        onClose={() => setLogViewerOpen(false)}
        serverName={logViewerConfig.serverName}
        serverId={logViewerConfig.serverId}
        isExternal={logViewerConfig.isExternal}
      />
    </Box>
  );
};

export default MCPServerManager;