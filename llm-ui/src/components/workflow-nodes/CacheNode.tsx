import React, { useState, useEffect } from 'react';
import { Handle, Position } from 'reactflow';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box, 
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  FormControlLabel,
  Checkbox,
  Switch,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Alert,
  IconButton,
  Tooltip,
  Slider,
  FormGroup,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  CircularProgress
} from '@mui/material';
import { 
  Storage as CacheIcon,
  ExpandMore as ExpandMoreIcon,
  Timer as TimerIcon,
  Key as KeyIcon,
  Settings as SettingsIcon,
  Info as InfoIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Speed as SpeedIcon,
  Visibility as InspectIcon,
  DeleteSweep as ClearCacheIcon
} from '@mui/icons-material';
import { selectMenuProps, selectEventHandlers, selectContainerStyle } from '../../utils/mui-select-props';
import PortalSelect from './PortalSelect';

interface CacheNodeProps {
  data: {
    label?: string;
    cacheKey?: string;
    cacheBackend?: 'redis' | 'memory' | 'disk' | 'hybrid';
    ttl?: number;
    maxCacheSize?: number;
    invalidateOn?: string[];
    enableWarming?: boolean;
    warmingSchedule?: string;
    compressionEnabled?: boolean;
    executionData?: {
      status?: 'idle' | 'running' | 'success' | 'error';
      cacheHit?: boolean;
      cacheSize?: number;
      hitRate?: number;
      lastAccessed?: string;
    };
  };
  id: string;
  updateNodeData?: (nodeId: string, newData: any) => void;
  showIO?: boolean;
}

const CacheNode: React.FC<CacheNodeProps> = ({ data, id, updateNodeData, showIO = true }) => {
  const [cacheExpanded, setCacheExpanded] = useState(false);
  const [performanceExpanded, setPerformanceExpanded] = useState(false);
  
  const [label, setLabel] = useState(data.label || 'Cache Layer');
  const [cacheKey, setCacheKey] = useState(data.cacheKey || '');
  const [cacheBackend, setCacheBackend] = useState(data.cacheBackend || 'redis');
  const [ttl, setTtl] = useState(data.ttl || 3600);
  const [maxCacheSize, setMaxCacheSize] = useState(data.maxCacheSize || 100);
  const [invalidateOn, setInvalidateOn] = useState<string[]>(data.invalidateOn || ['input_change']);
  const [enableWarming, setEnableWarming] = useState(data.enableWarming || false);
  const [warmingSchedule, setWarmingSchedule] = useState(data.warmingSchedule || '0 */6 * * *');
  const [compressionEnabled, setCompressionEnabled] = useState(data.compressionEnabled || false);
  
  // Cache inspection dialog state
  const [inspectDialogOpen, setInspectDialogOpen] = useState(false);
  const [cacheInspectionData, setCacheInspectionData] = useState<any>(null);
  const [inspectLoading, setInspectLoading] = useState(false);
  const [deleteLoading, setDeleteLoading] = useState(false);


  // Update parent node data when local state changes
  useEffect(() => {
    if (updateNodeData) {
      updateNodeData(id, {
        type: 'CacheNode',
        label,
        cacheKey,
        cacheBackend,
        ttl,
        maxCacheSize,
        invalidateOn,
        enableWarming,
        warmingSchedule,
        compressionEnabled,
        // Only preserve serializable execution data
        executionData: data.executionData ? {
          status: data.executionData.status,
          cacheHit: data.executionData.cacheHit,
          cacheSize: data.executionData.cacheSize,
          hitRate: data.executionData.hitRate,
          lastAccessed: data.executionData.lastAccessed
        } : undefined
      });
    }
  }, [label, cacheKey, cacheBackend, ttl, maxCacheSize, invalidateOn, 
      enableWarming, warmingSchedule, compressionEnabled, data.executionData]);

  const getStatusColor = () => {
    switch (data.executionData?.status) {
      case 'running': return '#ff9800';
      case 'success': return '#4caf50';
      case 'error': return '#f44336';
      default: return '#ffc107';
    }
  };

  const hasExecutionData = data.executionData && data.executionData.status !== 'idle';
  const cacheHit = data.executionData?.cacheHit;

  const formatTTL = (seconds: number) => {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
    return `${Math.floor(seconds / 86400)}d`;
  };

  const invalidationOptions = [
    { value: 'input_change', label: 'Input Change' },
    { value: 'time_based', label: 'Time Based' },
    { value: 'manual', label: 'Manual' },
    { value: 'event_based', label: 'Event Based' },
    { value: 'size_limit', label: 'Size Limit' }
  ];

  const handleInspectCache = async () => {
    setInspectLoading(true);
    try {
      // The backend API will search for cache keys matching the node ID pattern
      const response = await fetch(`http://127.0.0.1:8000/api/v1/automation/cache/status/${id}`);
      
      if (response.ok) {
        const data = await response.json();
        //console.log('Cache inspection data:', data);
        setCacheInspectionData(data);
        setInspectDialogOpen(true);
      } else {
        //console.error('Failed to inspect cache:', response.status);
        const errorText = await response.text();
        //console.error('Error response:', errorText);
      }
    } catch (error) {
      //console.error('Error inspecting cache:', error);
    } finally {
      setInspectLoading(false);
    }
  };

  const handleDeleteCache = async () => {
    if (!confirm('Are you sure you want to clear the cache for this node?')) {
      return;
    }
    
    setDeleteLoading(true);
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/v1/automation/cache/clear-node/${id}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        const result = await response.json();
        alert(`Cache cleared successfully. ${result.message}`);
        // Close inspection dialog if open
        setInspectDialogOpen(false);
        setCacheInspectionData(null);
      } else {
        //console.error('Failed to delete cache');
        alert('Failed to clear cache');
      }
    } catch (error) {
      //console.error('Error deleting cache:', error);
      alert('Error clearing cache');
    } finally {
      setDeleteLoading(false);
    }
  };

  return (
    <>
    <Card 
      sx={{ 
        minWidth: 350,
        maxWidth: 450,
        border: `2px solid ${getStatusColor()}`,
        borderRadius: 2,
        bgcolor: theme => theme.palette.mode === 'dark' ? 'rgba(255, 193, 7, 0.08)' : 'rgba(255, 193, 7, 0.05)',
        position: 'relative',
        overflow: 'visible',
        '&:hover': {
          boxShadow: theme => `0 0 20px ${theme.palette.mode === 'dark' ? 'rgba(255, 193, 7, 0.3)' : 'rgba(255, 193, 7, 0.2)'}`,
        }
      }}
    >
      <CardContent sx={{ pb: 1, overflow: 'visible' }}>
        {/* Header */}
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          <CacheIcon sx={{ color: getStatusColor() }} />
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
          <Box display="flex" gap={0.5}>
            <Tooltip title="Inspect cache contents">
              <IconButton 
                size="small" 
                onClick={handleInspectCache}
                disabled={inspectLoading}
              >
                {inspectLoading ? <CircularProgress size={16} /> : <InspectIcon sx={{ fontSize: 18 }} />}
              </IconButton>
            </Tooltip>
            <Tooltip title="Clear cache">
              <IconButton 
                size="small" 
                onClick={handleDeleteCache}
                disabled={deleteLoading}
              >
                {deleteLoading ? <CircularProgress size={16} /> : <ClearCacheIcon sx={{ fontSize: 18 }} />}
              </IconButton>
            </Tooltip>
            <Tooltip title="Cache intermediate results to improve performance">
              <InfoIcon sx={{ fontSize: 18, color: 'text.secondary', mt: 0.5 }} />
            </Tooltip>
          </Box>
        </Box>

        {/* Cache Status */}
        {hasExecutionData && (
          <Alert 
            severity={cacheHit ? 'success' : 'info'} 
            sx={{ mb: 2 }}
            icon={cacheHit ? <SpeedIcon /> : <RefreshIcon />}
          >
            <Typography variant="caption" fontWeight={500}>
              {cacheHit ? 'Cache Hit' : 'Cache Miss'} - 
              {data.executionData?.hitRate ? ` Hit Rate: ${data.executionData.hitRate}%` : ''}
            </Typography>
            {data.executionData?.cacheSize && (
              <Typography variant="caption" display="block">
                Cache Size: {data.executionData.cacheSize} items
              </Typography>
            )}
          </Alert>
        )}

        {/* Cache Key */}
        <TextField
          fullWidth
          size="small"
          label="Cache Key Pattern"
          value={cacheKey}
          onChange={(e) => setCacheKey(e.target.value)}
          placeholder="{workflow_id}:{node_id}:{input_hash}"
          helperText="Variables: {workflow_id}, {node_id}, {input_hash}, {timestamp}"
          sx={{ mb: 2 }}
          InputProps={{
            startAdornment: <KeyIcon sx={{ mr: 1, color: 'text.secondary' }} />
          }}
        />

        {/* Cache Configuration */}
        <Accordion 
          expanded={cacheExpanded} 
          onChange={(_, isExpanded) => setCacheExpanded(isExpanded)}
          sx={{ mb: 1, overflow: 'visible' }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box display="flex" alignItems="center" gap={1} width="100%">
              <SettingsIcon sx={{ fontSize: 20 }} />
              <Typography variant="body2" fontWeight={500}>Cache Configuration</Typography>
              <Box flexGrow={1} />
              <Chip 
                label={`TTL: ${formatTTL(ttl)}`} 
                size="small" 
                variant="outlined"
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails sx={{ overflow: 'visible' }}>
            <Box display="flex" flexDirection="column" gap={2}>
              {/* Cache Backend */}
              <PortalSelect
                value={cacheBackend}
                onChange={(value) => setCacheBackend(value as any)}
                label="Cache Backend"
                options={[
                  { value: 'redis', label: 'Redis (Distributed)' },
                  { value: 'memory', label: 'In-Memory (Fast)' },
                  { value: 'disk', label: 'Disk (Persistent)' },
                  { value: 'hybrid', label: 'Hybrid (Memory + Disk)' }
                ]}
                size="small"
                fullWidth
              />

              {/* TTL Input */}
              <TextField
                size="small"
                type="number"
                label="Time to Live (TTL in seconds)"
                value={ttl}
                onChange={(e) => setTtl(parseInt(e.target.value) || 3600)}
                fullWidth
                helperText={`Current: ${formatTTL(ttl)} (60s = 1min, 3600s = 1hour, 86400s = 1day)`}
                inputProps={{ min: 60, max: 86400, step: 60 }}
              />

              {/* Max Cache Size */}
              <TextField
                size="small"
                type="number"
                label="Max Cache Entries"
                value={maxCacheSize}
                onChange={(e) => setMaxCacheSize(parseInt(e.target.value) || 100)}
                fullWidth
                helperText="Maximum number of entries to store"
              />

              {/* Compression */}
              <FormControlLabel
                control={
                  <Switch
                    checked={compressionEnabled}
                    onChange={(e) => setCompressionEnabled(e.target.checked)}
                    size="small"
                  />
                }
                label="Enable Compression"
              />

              {/* Invalidation Rules */}
              <PortalSelect
                value={invalidateOn}
                onChange={(value) => setInvalidateOn(value as string[])}
                label="Invalidate On"
                options={invalidationOptions}
                multiple
                size="small"
              />
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Performance & Warming */}
        <Accordion 
          expanded={performanceExpanded} 
          onChange={(_, isExpanded) => setPerformanceExpanded(isExpanded)}
          sx={{ overflow: 'visible' }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box display="flex" alignItems="center" gap={1} width="100%">
              <TimerIcon sx={{ fontSize: 20 }} />
              <Typography variant="body2" fontWeight={500}>Performance & Warming</Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Box display="flex" flexDirection="column" gap={2}>
              {/* Cache Warming */}
              <FormControlLabel
                control={
                  <Switch
                    checked={enableWarming}
                    onChange={(e) => setEnableWarming(e.target.checked)}
                    size="small"
                  />
                }
                label="Enable Cache Warming"
              />

              {enableWarming && (
                <TextField
                  size="small"
                  label="Warming Schedule (Cron)"
                  value={warmingSchedule}
                  onChange={(e) => setWarmingSchedule(e.target.value)}
                  placeholder="0 */6 * * *"
                  helperText="Cron expression for cache warming"
                  fullWidth
                />
              )}

              {/* Cache Stats */}
              {hasExecutionData && (
                <Box bgcolor="action.hover" p={1} borderRadius={1}>
                  <Typography variant="caption" color="text.secondary">
                    Performance Metrics:
                  </Typography>
                  <Box mt={1}>
                    <Typography variant="body2">
                      Hit Rate: {data.executionData?.hitRate || 0}%
                    </Typography>
                    <Typography variant="body2">
                      Last Accessed: {data.executionData?.lastAccessed 
                        ? new Date(data.executionData.lastAccessed).toLocaleTimeString()
                        : 'Never'}
                    </Typography>
                  </Box>
                </Box>
              )}
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Input/Output Info */}
        {showIO && (
          <Box mt={2} p={1} bgcolor="action.hover" borderRadius={1}>
            <Typography variant="caption" color="text.secondary" fontWeight={500}>
              Pass-through node - caches data flowing through
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
    
    {/* Cache Inspection Dialog */}
    <Dialog
      open={inspectDialogOpen}
      onClose={() => setInspectDialogOpen(false)}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>Cache Inspection</DialogTitle>
      <DialogContent>
        {cacheInspectionData ? (
          <Box>
            <Typography variant="subtitle1" gutterBottom fontWeight={600}>
              Node: {cacheInspectionData.node_id}
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Cache Key: {cacheInspectionData.cache_key}
            </Typography>
            
            {cacheInspectionData.exists ? (
              <>
                <Alert severity="success" sx={{ my: 2 }}>
                  Cache Status: {cacheInspectionData.status}
                </Alert>
                
                {cacheInspectionData.metadata && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>Metadata:</Typography>
                    <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: 1, borderColor: 'divider' }}>
                      <Typography variant="body2">
                        Size: {cacheInspectionData.metadata.size_kb} KB
                      </Typography>
                      <Typography variant="body2">
                        Type: {cacheInspectionData.metadata.data_type}
                      </Typography>
                      {cacheInspectionData.metadata.age_hours !== undefined && (
                        <Typography variant="body2">
                          Age: {cacheInspectionData.metadata.age_hours.toFixed(1)} hours
                        </Typography>
                      )}
                    </Box>
                  </Box>
                )}
                
                {cacheInspectionData.preview && (
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>Preview:</Typography>
                    <Box sx={{ 
                      bgcolor: 'background.paper', 
                      p: 2, 
                      borderRadius: 1, 
                      border: 1, 
                      borderColor: 'divider',
                      maxHeight: 300,
                      overflow: 'auto'
                    }}>
                      <pre style={{ margin: 0, fontSize: '0.85rem' }}>
                        {cacheInspectionData.preview.preview_text || JSON.stringify(cacheInspectionData.preview, null, 2)}
                      </pre>
                    </Box>
                  </Box>
                )}
              </>
            ) : (
              <Alert severity="warning" sx={{ my: 2 }}>
                {cacheInspectionData.message || 'No cache data found'}
              </Alert>
            )}
          </Box>
        ) : (
          <CircularProgress />
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setInspectDialogOpen(false)}>Close</Button>
        {cacheInspectionData?.exists && (
          <Button 
            onClick={handleDeleteCache} 
            color="error"
            disabled={deleteLoading}
          >
            {deleteLoading ? <CircularProgress size={20} /> : 'Clear Cache'}
          </Button>
        )}
      </DialogActions>
    </Dialog>
    </>
  );
};

export default CacheNode;