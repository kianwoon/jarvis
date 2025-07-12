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
  Snackbar
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
  MoreVert as MoreVertIcon
} from '@mui/icons-material';

interface MCPServer {
  id?: number;
  name: string;
  config_type: 'manifest' | 'command' | 'remote_http';
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
  const [snackbar, setSnackbar] = useState<{open: boolean, message: string, severity: 'success' | 'error' | 'info'}>({
    open: false,
    message: '',
    severity: 'info'
  });

  const getServerTypeIcon = (type: string) => {
    switch (type) {
      case 'manifest':
        return <ManifestIcon />;
      case 'command':
        return <CommandIcon />;
      case 'remote_http':
        return <RemoteIcon />;
      default:
        return <ServerIcon />;
    }
  };

  const getStatusColor = (status?: string) => {
    switch (status) {
      case 'healthy':
        return 'success';
      case 'unhealthy':
        return 'error';
      default:
        return 'default';
    }
  };

  const showNotification = (message: string, severity: 'success' | 'error' | 'info' = 'info') => {
    setSnackbar({ open: true, message, severity });
  };

  const getNewServerTemplate = (type: 'manifest' | 'command' | 'remote_http'): MCPServer => {
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
          manifest_url: '',
          hostname: '',
          api_key: ''
        };
      case 'command':
        return {
          ...base,
          command: '',
          args: [],
          env: {},
          working_directory: '',
          restart_policy: 'on-failure',
          max_restarts: 3
        };
      case 'remote_http':
        return {
          ...base,
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

  const handleAdd = (type: 'manifest' | 'command' | 'remote_http') => {
    setEditingServer(getNewServerTemplate(type));
    setEditDialog(true);
  };

  const handleEdit = (server: MCPServer) => {
    setEditingServer({ ...server });
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

  const renderServerTable = () => (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Name</TableCell>
            <TableCell>Type</TableCell>
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
                <Chip 
                  label={server.is_active ? 'Active' : 'Inactive'} 
                  color={server.is_active ? 'success' : 'default'}
                  size="small" 
                />
              </TableCell>
              <TableCell>
                <Chip 
                  label={server.health_status || 'Unknown'} 
                  color={getStatusColor(server.health_status) as any}
                  size="small" 
                />
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
        [key]: { ...editingServer[key], ...updates }
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
                    <MenuItem value="command">Command-based</MenuItem>
                    <MenuItem value="remote_http">Remote HTTP/SSE</MenuItem>
                  </Select>
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

        {editingServer.config_type === 'command' && (
          <Card>
            <CardHeader title="Command Configuration" />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="Command"
                    value={editingServer.command || ''}
                    onChange={(e) => updateServer({ command: e.target.value })}
                    fullWidth
                    required
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="Working Directory"
                    value={editingServer.working_directory || ''}
                    onChange={(e) => updateServer({ working_directory: e.target.value })}
                    fullWidth
                  />
                </Grid>
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
                  <TextField
                    label="Arguments (JSON array)"
                    value={JSON.stringify(editingServer.args || [])}
                    onChange={(e) => {
                      try {
                        const args = JSON.parse(e.target.value);
                        updateServer({ args });
                      } catch {}
                    }}
                    fullWidth
                    multiline
                    rows={2}
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    label="Environment Variables (JSON object)"
                    value={JSON.stringify(editingServer.env || {})}
                    onChange={(e) => {
                      try {
                        const env = JSON.parse(e.target.value);
                        updateServer({ env });
                      } catch {}
                    }}
                    fullWidth
                    multiline
                    rows={3}
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        )}

        {editingServer.config_type === 'remote_http' && (
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
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">MCP Server Management</Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button onClick={onRefresh} startIcon={<RefreshIcon />}>
            Refresh Data
          </Button>
          <Button 
            onClick={() => handleAdd('manifest')} 
            variant="contained" 
            startIcon={<AddIcon />}
          >
            Add Manifest Server
          </Button>
          <Button 
            onClick={() => handleAdd('command')} 
            variant="contained" 
            startIcon={<AddIcon />}
          >
            Add Command Server
          </Button>
          <Button 
            onClick={() => handleAdd('remote_http')} 
            variant="contained" 
            startIcon={<AddIcon />}
          >
            Add Remote Server
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
        <MuiMenuItem onClick={() => handleMenuAction('health')}>
          <HealthIcon sx={{ mr: 1 }} fontSize="small" />
          Check Health
        </MuiMenuItem>
        <MuiMenuItem onClick={() => handleMenuAction('refresh')}>
          <RefreshIcon sx={{ mr: 1 }} fontSize="small" />
          Refresh
        </MuiMenuItem>
        {actionMenuServer?.config_type === 'command' && [
          <MuiMenuItem key="start" onClick={() => handleMenuAction('start')}>
            <StartIcon sx={{ mr: 1 }} fontSize="small" />
            Start Server
          </MuiMenuItem>,
          <MuiMenuItem key="stop" onClick={() => handleMenuAction('stop')}>
            <StopIcon sx={{ mr: 1 }} fontSize="small" />
            Stop Server
          </MuiMenuItem>
        ]}
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

      {/* Notification Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <Alert 
          onClose={() => setSnackbar({ ...snackbar, open: false })} 
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default MCPServerManager;