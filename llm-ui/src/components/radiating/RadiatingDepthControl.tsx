import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Slider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Tooltip,
  Chip,
  Grid,
  Divider,
  Alert,
  IconButton,
  Collapse
} from '@mui/material';
import {
  Save as SaveIcon,
  Refresh as ResetIcon,
  Info as InfoIcon,
  Speed as SpeedIcon,
  AccountTree as StrategyIcon,
  FilterAlt as FilterIcon,
  ExpandMore as ExpandIcon,
  ExpandLess as CollapseIcon
} from '@mui/icons-material';
import { RadiatingConfig, RadiatingConfigRequest, RadiatingConfigResponse } from '../../types/radiating';

interface RadiatingDepthControlProps {
  conversationId?: string;
  onConfigChange?: (config: RadiatingConfig) => void;
  disabled?: boolean;
  compact?: boolean;
  hideActions?: boolean;
}

const DEFAULT_CONFIG: RadiatingConfig = {
  enabled: true,
  maxDepth: 3,
  strategy: 'breadth-first',
  relevanceThreshold: 0.5,
  maxEntitiesPerLevel: 20,
  includeRelationships: true,
  autoExpand: false,
  cacheResults: true,
  timeoutMs: 30000
};

const STRATEGY_INFO = {
  'breadth-first': {
    name: 'Breadth-First',
    description: 'Explores all entities at the current depth before moving deeper. Best for comprehensive coverage.',
    icon: '📊',
    performance: 'Balanced'
  },
  'depth-first': {
    name: 'Depth-First',
    description: 'Follows promising paths deeply before exploring alternatives. Best for focused exploration.',
    icon: '🎯',
    performance: 'Fast'
  },
  'best-first': {
    name: 'Best-First',
    description: 'Prioritizes entities with highest relevance scores. Best for quality results.',
    icon: '⭐',
    performance: 'Optimal'
  },
  'adaptive': {
    name: 'Adaptive',
    description: 'Dynamically adjusts strategy based on results. Best for complex queries.',
    icon: '🧠',
    performance: 'Smart'
  }
};

const RadiatingDepthControl: React.FC<RadiatingDepthControlProps> = ({
  conversationId,
  onConfigChange,
  disabled = false,
  compact = false,
  hideActions = false
}) => {
  const [config, setConfig] = useState<RadiatingConfig>(DEFAULT_CONFIG);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [expanded, setExpanded] = useState(!compact);
  const [hasChanges, setHasChanges] = useState(false);

  // Load saved configuration
  useEffect(() => {
    const loadConfig = async () => {
      try {
        // Load from localStorage first
        const savedConfig = localStorage.getItem(`radiating-config-${conversationId || 'global'}`);
        if (savedConfig) {
          setConfig(JSON.parse(savedConfig));
        }

        // Sync with backend
        const response = await fetch('/api/v1/settings/radiating', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          }
        });

        if (response.ok) {
          const data = await response.json();
          if (data.settings) {
            // Map API response structure to component config
            const apiConfig = {
              enabled: data.settings.enabled ?? true,
              maxDepth: data.settings.max_depth ?? 3,
              strategy: data.settings.default_strategy ?? 'breadth-first',
              relevanceThreshold: data.settings.relevance_threshold ?? 0.5,
              maxEntitiesPerLevel: data.settings.max_entities_per_level ?? 20,
              includeRelationships: data.settings.include_relationships ?? true,
              autoExpand: data.settings.auto_expand ?? false,
              cacheResults: data.settings.cache_results ?? true,
              timeoutMs: data.settings.timeout_ms ?? 30000
            };
            setConfig(apiConfig);
            localStorage.setItem(
              `radiating-config-${conversationId || 'global'}`,
              JSON.stringify(apiConfig)
            );
          }
        }
      } catch (err) {
        console.warn('Failed to load radiating config:', err);
      }
    };

    loadConfig();
  }, [conversationId]);

  const handleConfigUpdate = (updates: Partial<RadiatingConfig>) => {
    const newConfig = { ...config, ...updates };
    setConfig(newConfig);
    setHasChanges(true);

    // Notify parent immediately for preview
    if (onConfigChange) {
      onConfigChange(newConfig);
    }
  };

  const handleSave = async () => {
    setLoading(true);
    setError(null);
    setSuccess(false);

    try {
      // Map component config to API structure
      const apiPayload = {
        settings: {
          enabled: config.enabled,
          max_depth: config.maxDepth,
          default_strategy: config.strategy,
          relevance_threshold: config.relevanceThreshold,
          max_entities_per_level: config.maxEntitiesPerLevel,
          include_relationships: config.includeRelationships,
          auto_expand: config.autoExpand,
          cache_results: config.cacheResults,
          timeout_ms: config.timeoutMs,
          conversation_id: conversationId
        },
        persist_to_db: true,
        reload_cache: true
      };

      const response = await fetch('/api/v1/settings/radiating', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(apiPayload)
      });

      if (!response.ok) {
        throw new Error(`Failed to save configuration: ${response.statusText}`);
      }

      const data = await response.json();
      
      if (response.ok) {
        // Save to localStorage
        localStorage.setItem(
          `radiating-config-${conversationId || 'global'}`,
          JSON.stringify(config)
        );
        
        setSuccess(true);
        setHasChanges(false);
        
        // Clear success message after 3 seconds
        setTimeout(() => setSuccess(false), 3000);
      } else {
        throw new Error('Failed to save configuration');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      // Clear error after 5 seconds
      setTimeout(() => setError(null), 5000);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setConfig(DEFAULT_CONFIG);
    setHasChanges(true);
    if (onConfigChange) {
      onConfigChange(DEFAULT_CONFIG);
    }
  };

  const getDepthPreview = (depth: number) => {
    const levels = [];
    for (let i = 0; i <= depth; i++) {
      levels.push(
        <Box
          key={i}
          sx={{
            width: 10 + i * 15,
            height: 10 + i * 15,
            border: '2px solid',
            borderColor: i === 0 ? 'primary.main' : 'primary.light',
            borderRadius: '50%',
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            opacity: 1 - (i * 0.15)
          }}
        />
      );
    }
    return levels;
  };

  return (
    <Paper
      elevation={2}
      sx={{
        p: compact ? 1.5 : 2,
        backgroundColor: 'background.paper',
        borderRadius: 2
      }}
    >
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <StrategyIcon color="primary" />
          <Typography variant="h6">
            Radiating Configuration
          </Typography>
        </Box>
        
        {compact && (
          <IconButton
            size="small"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? <CollapseIcon /> : <ExpandIcon />}
          </IconButton>
        )}
      </Box>

      <Collapse in={expanded || !compact}>
        <Grid container spacing={3}>
          {/* Traversal Depth */}
          <Grid item xs={12} md={6}>
            <Box>
              <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <SpeedIcon fontSize="small" />
                Traversal Depth
                <Tooltip title="How many levels deep to explore from the initial entity">
                  <InfoIcon fontSize="small" sx={{ opacity: 0.6 }} />
                </Tooltip>
              </Typography>
              
              <Box sx={{ px: 2, py: 1 }}>
                <Slider
                  value={config.maxDepth}
                  onChange={(_, value) => handleConfigUpdate({ maxDepth: value as number })}
                  min={1}
                  max={5}
                  marks
                  step={1}
                  valueLabelDisplay="on"
                  disabled={disabled || loading}
                  sx={{ mb: 1 }}
                />
                
                {/* Visual depth preview */}
                <Box sx={{ 
                  height: 80, 
                  position: 'relative',
                  backgroundColor: 'action.hover',
                  borderRadius: 1,
                  overflow: 'hidden'
                }}>
                  {getDepthPreview(config.maxDepth)}
                </Box>
                
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                  Current: Level {config.maxDepth} • Estimated entities: ~{Math.pow(config.maxEntitiesPerLevel, config.maxDepth)}
                </Typography>
              </Box>
            </Box>
          </Grid>

          {/* Strategy Selection */}
          <Grid item xs={12} md={6}>
            <Box>
              <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <StrategyIcon fontSize="small" />
                Exploration Strategy
              </Typography>
              
              <FormControl fullWidth size="small" disabled={disabled || loading}>
                <Select
                  value={config.strategy}
                  onChange={(e) => handleConfigUpdate({ 
                    strategy: e.target.value as RadiatingConfig['strategy'] 
                  })}
                >
                  {Object.entries(STRATEGY_INFO).map(([key, info]) => (
                    <MenuItem key={key} value={key}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                        <Typography>{info.icon}</Typography>
                        <Box sx={{ flex: 1 }}>
                          <Typography variant="body2">{info.name}</Typography>
                          <Typography variant="caption" color="text.secondary">
                            {info.performance}
                          </Typography>
                        </Box>
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              
              {/* Strategy description */}
              <Alert 
                severity="info" 
                icon={false}
                sx={{ mt: 1, py: 0.5 }}
              >
                <Typography variant="caption">
                  {STRATEGY_INFO[config.strategy].description}
                </Typography>
              </Alert>
            </Box>
          </Grid>

          {/* Relevance Threshold */}
          <Grid item xs={12}>
            <Box>
              <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <FilterIcon fontSize="small" />
                Relevance Threshold
                <Tooltip title="Minimum relevance score required to include an entity">
                  <InfoIcon fontSize="small" sx={{ opacity: 0.6 }} />
                </Tooltip>
              </Typography>
              
              <Box sx={{ px: 2 }}>
                <Slider
                  value={config.relevanceThreshold}
                  onChange={(_, value) => handleConfigUpdate({ relevanceThreshold: value as number })}
                  min={0.1}
                  max={1.0}
                  step={0.1}
                  marks={[
                    { value: 0.1, label: 'Low' },
                    { value: 0.5, label: 'Medium' },
                    { value: 1.0, label: 'High' }
                  ]}
                  valueLabelDisplay="on"
                  valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
                  disabled={disabled || loading}
                />
                
                {/* Threshold quality indicator */}
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                  <Chip
                    label="More Results"
                    size="small"
                    color={config.relevanceThreshold <= 0.3 ? 'primary' : 'default'}
                    variant={config.relevanceThreshold <= 0.3 ? 'filled' : 'outlined'}
                  />
                  <Chip
                    label="Balanced"
                    size="small"
                    color={config.relevanceThreshold > 0.3 && config.relevanceThreshold <= 0.7 ? 'primary' : 'default'}
                    variant={config.relevanceThreshold > 0.3 && config.relevanceThreshold <= 0.7 ? 'filled' : 'outlined'}
                  />
                  <Chip
                    label="High Quality"
                    size="small"
                    color={config.relevanceThreshold > 0.7 ? 'primary' : 'default'}
                    variant={config.relevanceThreshold > 0.7 ? 'filled' : 'outlined'}
                  />
                </Box>
              </Box>
            </Box>
          </Grid>

          {/* Advanced Settings */}
          <Grid item xs={12}>
            <Divider sx={{ my: 1 }} />
            <Typography variant="subtitle2" gutterBottom>
              Advanced Settings
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6} md={3}>
                <TextField
                  label="Max Entities/Level"
                  type="number"
                  size="small"
                  fullWidth
                  value={config.maxEntitiesPerLevel}
                  onChange={(e) => handleConfigUpdate({ 
                    maxEntitiesPerLevel: parseInt(e.target.value) || 20 
                  })}
                  disabled={disabled || loading}
                  InputProps={{
                    inputProps: { min: 1, max: 100 }
                  }}
                />
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={config.includeRelationships}
                      onChange={(e) => handleConfigUpdate({ 
                        includeRelationships: e.target.checked 
                      })}
                      disabled={disabled || loading}
                    />
                  }
                  label="Include Relationships"
                />
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={config.autoExpand}
                      onChange={(e) => handleConfigUpdate({ 
                        autoExpand: e.target.checked 
                      })}
                      disabled={disabled || loading}
                    />
                  }
                  label="Auto-Expand"
                />
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={config.cacheResults}
                      onChange={(e) => handleConfigUpdate({ 
                        cacheResults: e.target.checked 
                      })}
                      disabled={disabled || loading}
                    />
                  }
                  label="Cache Results"
                />
              </Grid>
            </Grid>
          </Grid>
        </Grid>

        {/* Action Buttons - Only show if not hidden */}
        {!hideActions && (
          <Box sx={{ display: 'flex', gap: 2, mt: 3, justifyContent: 'flex-end' }}>
            <Button
              variant="outlined"
              startIcon={<ResetIcon />}
              onClick={handleReset}
              disabled={disabled || loading}
            >
              Reset
            </Button>
            
            <Button
              variant="contained"
              startIcon={<SaveIcon />}
              onClick={handleSave}
              disabled={disabled || loading || !hasChanges}
            >
              {loading ? 'Saving...' : 'Save Settings'}
            </Button>
          </Box>
        )}

        {/* Status Messages */}
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
        
        {success && (
          <Alert severity="success" sx={{ mt: 2 }}>
            Configuration saved successfully!
          </Alert>
        )}
      </Collapse>
    </Paper>
  );
};

// Add missing imports
import { TextField, FormControlLabel, Switch } from '@mui/material';

export default RadiatingDepthControl;