import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  TextField,
  IconButton,
  Paper,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  CircularProgress,
  Alert
} from '@mui/material';
import {
  Close as CloseIcon,
  Refresh as RefreshIcon,
  ContentCopy as CopyIcon,
  Download as DownloadIcon,
  Terminal as TerminalIcon
} from '@mui/icons-material';

interface MCPLogViewerProps {
  open: boolean;
  onClose: () => void;
  serverName?: string;
  serverId?: number | string;
}

const MCPLogViewer: React.FC<MCPLogViewerProps> = ({
  open,
  onClose,
  serverName = 'MCP Server',
  serverId
}) => {
  const [logs, setLogs] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lines, setLines] = useState(100);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    if (open) {
      fetchLogs();
    }
  }, [open, lines]);

  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    if (autoRefresh && open) {
      interval = setInterval(fetchLogs, 5000); // Refresh every 5 seconds
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh, open]);

  const fetchLogs = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // If no serverId, we can't fetch logs
      if (!serverId) {
        setError('No server ID provided');
        setLoading(false);
        return;
      }
      
      const endpoint = `/api/v1/mcp/servers/${serverId}/logs?lines=${lines}`;
      
      const response = await fetch(endpoint);
      if (response.ok) {
        const data = await response.json();
        setLogs(data.logs || 'No logs available');
      } else {
        setError('Failed to fetch logs');
      }
    } catch (err) {
      setError('Error fetching logs: ' + err);
    } finally {
      setLoading(false);
    }
  };

  const handleCopyLogs = () => {
    navigator.clipboard.writeText(logs);
  };

  const handleDownloadLogs = () => {
    const blob = new Blob([logs], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${serverName.replace(/\s+/g, '_')}_logs_${new Date().toISOString()}.log`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  const filteredLogs = searchTerm 
    ? logs.split('\n').filter(line => 
        line.toLowerCase().includes(searchTerm.toLowerCase())
      ).join('\n')
    : logs;

  return (
    <Dialog 
      open={open} 
      onClose={onClose} 
      maxWidth="lg" 
      fullWidth
      PaperProps={{
        sx: { height: '80vh', display: 'flex', flexDirection: 'column' }
      }}
    >
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <TerminalIcon />
        <Typography variant="h6" sx={{ flex: 1 }}>
          {serverName} Logs
        </Typography>
        <IconButton onClick={onClose} size="small">
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      
      <Box sx={{ px: 3, pb: 2, display: 'flex', gap: 2, alignItems: 'center' }}>
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Lines</InputLabel>
          <Select
            value={lines}
            onChange={(e) => setLines(Number(e.target.value))}
            label="Lines"
          >
            <MenuItem value={50}>Last 50</MenuItem>
            <MenuItem value={100}>Last 100</MenuItem>
            <MenuItem value={200}>Last 200</MenuItem>
            <MenuItem value={500}>Last 500</MenuItem>
            <MenuItem value={1000}>Last 1000</MenuItem>
          </Select>
        </FormControl>
        
        <TextField
          size="small"
          placeholder="Search logs..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          sx={{ flex: 1 }}
        />
        
        <Button
          startIcon={<RefreshIcon />}
          onClick={fetchLogs}
          disabled={loading}
          variant="outlined"
          size="small"
        >
          Refresh
        </Button>
        
        <Button
          onClick={() => setAutoRefresh(!autoRefresh)}
          variant={autoRefresh ? "contained" : "outlined"}
          size="small"
          color={autoRefresh ? "primary" : "inherit"}
        >
          Auto-Refresh: {autoRefresh ? 'ON' : 'OFF'}
        </Button>
        
        <IconButton onClick={handleCopyLogs} title="Copy logs">
          <CopyIcon />
        </IconButton>
        
        <IconButton onClick={handleDownloadLogs} title="Download logs">
          <DownloadIcon />
        </IconButton>
      </Box>
      
      <DialogContent sx={{ flex: 1, overflow: 'hidden', p: 0 }}>
        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress />
          </Box>
        )}
        
        {error && (
          <Alert severity="error" sx={{ m: 2 }}>
            {error}
          </Alert>
        )}
        
        {!loading && !error && (
          <Paper 
            variant="outlined" 
            sx={{ 
              height: '100%', 
              bgcolor: '#1e1e1e',
              color: '#e0e0e0',
              overflow: 'auto',
              p: 2,
              fontFamily: 'monospace',
              fontSize: '0.875rem',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word'
            }}
          >
            {filteredLogs || 'No logs available'}
          </Paper>
        )}
      </DialogContent>
      
      <DialogActions>
        <Typography variant="caption" color="text.secondary" sx={{ flex: 1, ml: 2 }}>
          {autoRefresh && 'Auto-refreshing every 5 seconds'}
        </Typography>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

export default MCPLogViewer;