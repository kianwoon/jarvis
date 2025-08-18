import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  TextField,
  Button,
  Alert,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Chip,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  Save as SaveIcon,
  RestartAlt as ResetIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  ContentCopy as CopyIcon,
  FileDownload as ExportIcon,
  FileUpload as ImportIcon,
  ExpandMore as ExpandIcon,
  Hub as RadiatingIcon,
  Settings as SettingsIcon,
  Bookmark as PresetIcon,
  Speed as PerformanceIcon,
  Palette as VisualizationIcon,
  Code as PromptsIcon,
  Psychology as AIIcon
} from '@mui/icons-material';
import { 
  RadiatingSettings as RadiatingSettingsType,
  RadiatingConfig,
  RadiatingPreset 
} from '../../types/radiating';
import RadiatingDepthControl from '../radiating/RadiatingDepthControl';
import RadiatingPromptsSettings from './RadiatingPromptsSettings';
import RadiatingModelSettings from './RadiatingModelSettings';

interface RadiatingSettingsProps {
  onSettingsChange?: (settings: RadiatingSettingsType) => void;
}

const DEFAULT_SETTINGS: RadiatingSettingsType & { prompts?: any; model_config?: any } = {
  defaultConfig: {
    enabled: true,
    maxDepth: 3,
    strategy: 'breadth-first',
    relevanceThreshold: 0.5,
    maxEntitiesPerLevel: 20,
    includeRelationships: true,
    autoExpand: false,
    cacheResults: true,
    timeoutMs: 30000
  },
  prompts: {},
  model_config: {
    model: 'llama3.1:8b',
    max_tokens: 4096,
    temperature: 0.7,
    context_length: 128000,
    model_server: 'http://localhost:11434',
    system_prompt: '',
    llm_mode: 'non-thinking'
  },
  presets: [
    {
      id: 'quick',
      name: 'Quick Exploration',
      description: 'Fast, shallow exploration for quick results',
      config: {
        enabled: true,
        maxDepth: 2,
        strategy: 'breadth-first',
        relevanceThreshold: 0.7,
        maxEntitiesPerLevel: 10,
        includeRelationships: false,
        autoExpand: false,
        cacheResults: true,
        timeoutMs: 15000
      }
    },
    {
      id: 'deep',
      name: 'Deep Analysis',
      description: 'Comprehensive exploration with maximum coverage',
      config: {
        enabled: true,
        maxDepth: 5,
        strategy: 'adaptive',
        relevanceThreshold: 0.3,
        maxEntitiesPerLevel: 30,
        includeRelationships: true,
        autoExpand: true,
        cacheResults: true,
        timeoutMs: 60000
      }
    },
    {
      id: 'focused',
      name: 'Focused Search',
      description: 'High-quality results with strict relevance filtering',
      config: {
        enabled: true,
        maxDepth: 3,
        strategy: 'best-first',
        relevanceThreshold: 0.8,
        maxEntitiesPerLevel: 15,
        includeRelationships: true,
        autoExpand: false,
        cacheResults: true,
        timeoutMs: 30000
      }
    }
  ],
  visualizationPreferences: {
    nodeSize: 'relevance',
    linkThickness: 'weight',
    colorScheme: 'depth',
    layout: 'force',
    showLabels: true,
    animationSpeed: 1
  }
};

const RadiatingSettings: React.FC<RadiatingSettingsProps> = ({ onSettingsChange }) => {
  const [settings, setSettings] = useState<RadiatingSettingsType & { prompts?: any; model_config?: any }>(DEFAULT_SETTINGS);
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [presetDialog, setPresetDialog] = useState(false);
  const [editingPreset, setEditingPreset] = useState<RadiatingPreset | null>(null);
  const [newPreset, setNewPreset] = useState<Partial<RadiatingPreset>>({
    name: '',
    description: '',
    config: DEFAULT_SETTINGS.defaultConfig
  });

  // Load settings from API
  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/settings/radiating');
      if (response.ok) {
        const data = await response.json();
        console.log('=== LOAD SETTINGS DEBUG ===');
        console.log('API Response:', data);
        console.log('Has model_config in response:', !!data.settings?.model_config);
        
        // Extract settings from the API response structure
        if (data.settings) {
          // CRITICAL: Always ensure model_config exists
          const modelConfigFromAPI = data.settings.model_config || DEFAULT_SETTINGS.model_config;
          
          // Map API response to component state structure
          const mappedSettings: RadiatingSettingsType & { prompts?: any; model_config?: any } = {
            defaultConfig: {
              enabled: data.settings.enabled ?? true,
              maxDepth: data.settings.max_depth ?? 3,
              strategy: data.settings.default_strategy ?? 'breadth-first',
              relevanceThreshold: data.settings.relevance_threshold ?? 0.5,
              maxEntitiesPerLevel: data.settings.max_entities_per_level ?? 20,
              includeRelationships: data.settings.include_relationships ?? true,
              autoExpand: data.settings.auto_expand ?? false,
              cacheResults: data.settings.cache_results ?? true,
              timeoutMs: data.settings.timeout_ms ?? 30000
            },
            presets: data.settings.presets ?? DEFAULT_SETTINGS.presets,
            visualizationPreferences: data.settings.visualization_preferences ?? {
              nodeSize: data.settings.visualization_preferences?.node_size ?? 'relevance',
              linkThickness: data.settings.visualization_preferences?.link_thickness ?? 'weight',
              colorScheme: data.settings.visualization_preferences?.color_scheme ?? 'depth',
              layout: data.settings.visualization_preferences?.layout ?? 'force',
              showLabels: data.settings.visualization_preferences?.show_labels ?? true,
              animationSpeed: data.settings.visualization_preferences?.animation_speed ?? 1
            },
            prompts: data.settings.prompts || {},
            model_config: modelConfigFromAPI  // ALWAYS use the validated model_config
          };
          
          console.log('Mapped settings with model_config:', {
            hasModelConfig: !!mappedSettings.model_config,
            modelConfig: mappedSettings.model_config
          });
          
          setSettings(mappedSettings);
          
          // Save to localStorage as backup
          localStorage.setItem('radiating-settings', JSON.stringify(mappedSettings));
        } else {
          // If no settings in response, use defaults
          console.log('No settings in response, using defaults');
          setSettings(DEFAULT_SETTINGS);
        }
      }
    } catch (err) {
      console.error('Failed to load settings:', err);
      // Try to load from localStorage
      const cached = localStorage.getItem('radiating-settings');
      if (cached) {
        try {
          setSettings(JSON.parse(cached));
        } catch (e) {
          console.error('Failed to parse cached settings:', e);
          setSettings(DEFAULT_SETTINGS);
        }
      } else {
        setSettings(DEFAULT_SETTINGS);
      }
    } finally {
      setLoading(false);
    }
  };

  const saveSettings = async () => {
    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      // CRITICAL DEBUG: Log the exact state being saved
      console.log('=== SAVE SETTINGS DEBUG ===');
      console.log('Active Tab:', activeTab);
      console.log('Settings state:', {
        hasPrompts: !!settings.prompts,
        promptsKeys: Object.keys(settings.prompts || {}),
        hasModelConfig: !!settings.model_config,
        modelConfig: settings.model_config,
        hasDefaultConfig: !!settings.defaultConfig,
        hasPresets: !!settings.presets,
        hasVisualization: !!settings.visualizationPreferences
      });
      console.log('Full settings object:', JSON.stringify(settings, null, 2));

      // DEFENSIVE: Ensure critical fields are ALWAYS present
      const modelConfig = settings.model_config || DEFAULT_SETTINGS.model_config;
      const prompts = settings.prompts || {};
      
      // Validate that we have model_config
      if (!modelConfig || Object.keys(modelConfig).length === 0) {
        console.error('WARNING: model_config is missing or empty!');
      }

      // Map component state to API structure (SettingsUpdate model)
      const apiPayload = {
        settings: {
          enabled: settings.defaultConfig.enabled,
          max_depth: settings.defaultConfig.maxDepth,
          default_strategy: settings.defaultConfig.strategy,
          relevance_threshold: settings.defaultConfig.relevanceThreshold,
          max_entities_per_level: settings.defaultConfig.maxEntitiesPerLevel,
          include_relationships: settings.defaultConfig.includeRelationships,
          auto_expand: settings.defaultConfig.autoExpand,
          cache_results: settings.defaultConfig.cacheResults,
          timeout_ms: settings.defaultConfig.timeoutMs,
          presets: settings.presets,
          visualization_preferences: {
            node_size: settings.visualizationPreferences.nodeSize,
            link_thickness: settings.visualizationPreferences.linkThickness,
            color_scheme: settings.visualizationPreferences.colorScheme,
            layout: settings.visualizationPreferences.layout,
            show_labels: settings.visualizationPreferences.showLabels,
            animation_speed: settings.visualizationPreferences.animationSpeed
          },
          // Include existing coverage and synthesis settings if they exist
          coverage: {
            gap_threshold: 0.6,
            overlap_threshold: 0.8,
            enable_gap_detection: true,
            redundancy_threshold: 0.9,
            completeness_threshold: 0.7,
            enable_coverage_metrics: true,
            enable_overlap_detection: true,
            metric_calculation_interval: 100,
            enable_completeness_checking: true,
            enable_redundancy_elimination: true
          },
          synthesis: {
            merge_strategy: 'weighted',
            enable_synthesis: true,
            confidence_threshold: 0.7,
            contradiction_resolution: 'highest_confidence',
            enable_source_tracking: true,
            max_synthesis_sources: 10,
            enable_confidence_scoring: true,
            enable_contradiction_detection: true
          },
          // Include prompts - guaranteed to be an object
          prompts: prompts,
          // Include model configuration - guaranteed to have default values
          model_config: modelConfig,
          cache_ttl: 3600
        },
        persist_to_db: true,
        reload_cache: true
      };

      // Debug: Log what we're sending to API
      console.log('API payload being sent:', {
        hasPrompts: !!apiPayload.settings.prompts,
        hasModelConfig: !!apiPayload.settings.model_config,
        modelConfig: apiPayload.settings.model_config,
        promptsKeys: Object.keys(apiPayload.settings.prompts || {}),
        settingsKeys: Object.keys(apiPayload.settings)
      });

      const response = await fetch('/api/v1/settings/radiating', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(apiPayload)
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Settings saved successfully:', result);
        setSuccess('Settings saved successfully!');
        localStorage.setItem('radiating-settings', JSON.stringify(settings));
        
        if (onSettingsChange) {
          onSettingsChange(settings);
        }
        
        setTimeout(() => setSuccess(null), 3000);
      } else {
        const errorText = await response.text();
        console.error('Save failed:', errorText);
        throw new Error(`Failed to save settings: ${response.status} ${response.statusText}`);
      }
    } catch (err) {
      console.error('Error saving settings:', err);
      setError(err instanceof Error ? err.message : 'Unknown error occurred while saving');
      setTimeout(() => setError(null), 5000);
    } finally {
      setLoading(false);
    }
  };

  const handleConfigChange = (config: RadiatingConfig) => {
    setSettings({
      ...settings,
      defaultConfig: config
    });
  };

  const handlePresetSave = () => {
    if (!newPreset.name || !newPreset.description) {
      setError('Please provide a name and description for the preset');
      return;
    }

    const preset: RadiatingPreset = {
      id: editingPreset?.id || `preset-${Date.now()}`,
      name: newPreset.name,
      description: newPreset.description,
      config: newPreset.config || settings.defaultConfig
    };

    const updatedPresets = editingPreset
      ? settings.presets.map(p => p.id === editingPreset.id ? preset : p)
      : [...settings.presets, preset];

    setSettings({
      ...settings,
      presets: updatedPresets
    });

    setPresetDialog(false);
    setEditingPreset(null);
    setNewPreset({
      name: '',
      description: '',
      config: DEFAULT_SETTINGS.defaultConfig
    });
  };

  const handlePresetDelete = (id: string) => {
    setSettings({
      ...settings,
      presets: settings.presets.filter(p => p.id !== id)
    });
  };

  const handlePresetApply = (preset: RadiatingPreset) => {
    setSettings({
      ...settings,
      defaultConfig: preset.config
    });
    setSuccess(`Applied preset: ${preset.name}`);
    setTimeout(() => setSuccess(null), 3000);
  };

  const handleExport = () => {
    const dataStr = JSON.stringify(settings, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `radiating-settings-${Date.now()}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  const handleImport = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const imported = JSON.parse(e.target?.result as string);
        setSettings(imported);
        setSuccess('Settings imported successfully!');
        setTimeout(() => setSuccess(null), 3000);
      } catch (err) {
        setError('Failed to import settings. Invalid file format.');
        setTimeout(() => setError(null), 5000);
      }
    };
    reader.readAsText(file);
  };

  return (
    <Paper elevation={2} sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <RadiatingIcon color="primary" sx={{ fontSize: 32 }} />
          <Typography variant="h5">
            Radiating Coverage Settings
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<ExportIcon />}
            onClick={handleExport}
          >
            Export
          </Button>
          
          <Button
            variant="outlined"
            component="label"
            startIcon={<ImportIcon />}
          >
            Import
            <input
              type="file"
              accept=".json"
              hidden
              onChange={handleImport}
            />
          </Button>
        </Box>
      </Box>

      {/* Status Messages */}
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      {success && <Alert severity="success" sx={{ mb: 2 }}>{success}</Alert>}

      {/* Tabs */}
      <Tabs value={activeTab} onChange={(event, newValue) => setActiveTab(newValue)} sx={{ mb: 3 }}>
        <Tab label="Configuration" icon={<SettingsIcon />} iconPosition="start" />
        <Tab label="Model" icon={<AIIcon />} iconPosition="start" />
        <Tab label="Presets" icon={<PresetIcon />} iconPosition="start" />
        <Tab label="Performance" icon={<PerformanceIcon />} iconPosition="start" />
        <Tab label="Visualization" icon={<VisualizationIcon />} iconPosition="start" />
        <Tab label="Prompts" icon={<PromptsIcon />} iconPosition="start" />
      </Tabs>

      {/* Tab Content */}
      {activeTab === 0 && (
        <Box>
          <RadiatingDepthControl
            onConfigChange={handleConfigChange}
            compact={false}
            hideActions={true}
          />
        </Box>
      )}

      {activeTab === 1 && (
        <Box>
          <RadiatingModelSettings
            modelConfig={settings.model_config || DEFAULT_SETTINGS.model_config}
            onChange={(config) => {
              console.log('Model config updated:', config);
              setSettings(prevSettings => ({ 
                ...prevSettings, 
                model_config: config 
              }));
            }}
            onShowSuccess={setSuccess}
          />
        </Box>
      )}

      {activeTab === 2 && (
        <Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="h6">Configuration Presets</Typography>
            <Button
              startIcon={<AddIcon />}
              onClick={() => {
                setEditingPreset(null);
                setNewPreset({
                  name: '',
                  description: '',
                  config: settings.defaultConfig
                });
                setPresetDialog(true);
              }}
            >
              Add Preset
            </Button>
          </Box>

          <List>
            {settings.presets.map((preset) => (
              <ListItem
                key={preset.id}
                sx={{
                  border: 1,
                  borderColor: 'divider',
                  borderRadius: 1,
                  mb: 1
                }}
              >
                <ListItemIcon>
                  <PresetIcon />
                </ListItemIcon>
                <ListItemText
                  primary={preset.name}
                  secondary={
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        {preset.description}
                      </Typography>
                      <Box sx={{ mt: 1, display: 'flex', gap: 0.5 }}>
                        <Chip label={`Depth: ${preset.config.maxDepth}`} size="small" />
                        <Chip label={preset.config.strategy} size="small" />
                        <Chip label={`${(preset.config.relevanceThreshold * 100).toFixed(0)}% relevance`} size="small" />
                      </Box>
                    </Box>
                  }
                />
                <ListItemSecondaryAction>
                  <Button
                    size="small"
                    onClick={() => handlePresetApply(preset)}
                    sx={{ mr: 1 }}
                  >
                    Apply
                  </Button>
                  <IconButton
                    size="small"
                    onClick={() => {
                      setEditingPreset(preset);
                      setNewPreset(preset);
                      setPresetDialog(true);
                    }}
                  >
                    <EditIcon />
                  </IconButton>
                  <IconButton
                    size="small"
                    onClick={() => handlePresetDelete(preset.id)}
                  >
                    <DeleteIcon />
                  </IconButton>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
        </Box>
      )}

      {activeTab === 3 && (
        <Box>
          <Typography variant="h6" gutterBottom>Performance Settings</Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Timeout (ms)"
                type="number"
                value={settings.defaultConfig.timeoutMs}
                onChange={(e) => setSettings({
                  ...settings,
                  defaultConfig: {
                    ...settings.defaultConfig,
                    timeoutMs: parseInt(e.target.value) || 30000
                  }
                })}
                helperText="Maximum time for radiating traversal"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Max Entities Per Level"
                type="number"
                value={settings.defaultConfig.maxEntitiesPerLevel}
                onChange={(e) => setSettings({
                  ...settings,
                  defaultConfig: {
                    ...settings.defaultConfig,
                    maxEntitiesPerLevel: parseInt(e.target.value) || 20
                  }
                })}
                helperText="Limit entities at each depth level"
              />
            </Grid>
            
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.defaultConfig.cacheResults}
                    onChange={(e) => setSettings({
                      ...settings,
                      defaultConfig: {
                        ...settings.defaultConfig,
                        cacheResults: e.target.checked
                      }
                    })}
                  />
                }
                label="Cache Results"
              />
              <Typography variant="caption" display="block" color="text.secondary">
                Store radiating results for faster repeated queries
              </Typography>
            </Grid>
          </Grid>
        </Box>
      )}

      {activeTab === 4 && (
        <Box>
          <Typography variant="h6" gutterBottom>Visualization Preferences</Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Node Size</InputLabel>
                <Select
                  value={settings.visualizationPreferences.nodeSize}
                  onChange={(e) => setSettings({
                    ...settings,
                    visualizationPreferences: {
                      ...settings.visualizationPreferences,
                      nodeSize: e.target.value as any
                    }
                  })}
                >
                  <MenuItem value="fixed">Fixed Size</MenuItem>
                  <MenuItem value="relevance">Based on Relevance</MenuItem>
                  <MenuItem value="connections">Based on Connections</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Color Scheme</InputLabel>
                <Select
                  value={settings.visualizationPreferences.colorScheme}
                  onChange={(e) => setSettings({
                    ...settings,
                    visualizationPreferences: {
                      ...settings.visualizationPreferences,
                      colorScheme: e.target.value as any
                    }
                  })}
                >
                  <MenuItem value="depth">By Depth Level</MenuItem>
                  <MenuItem value="type">By Entity Type</MenuItem>
                  <MenuItem value="relevance">By Relevance Score</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Default Layout</InputLabel>
                <Select
                  value={settings.visualizationPreferences.layout}
                  onChange={(e) => setSettings({
                    ...settings,
                    visualizationPreferences: {
                      ...settings.visualizationPreferences,
                      layout: e.target.value as any
                    }
                  })}
                >
                  <MenuItem value="force">Force-Directed</MenuItem>
                  <MenuItem value="radial">Radial</MenuItem>
                  <MenuItem value="hierarchical">Hierarchical</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Animation Speed"
                type="number"
                value={settings.visualizationPreferences.animationSpeed}
                onChange={(e) => setSettings({
                  ...settings,
                  visualizationPreferences: {
                    ...settings.visualizationPreferences,
                    animationSpeed: parseFloat(e.target.value) || 1
                  }
                })}
                inputProps={{ min: 0.1, max: 3, step: 0.1 }}
                helperText="Speed multiplier for animations"
              />
            </Grid>
            
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.visualizationPreferences.showLabels}
                    onChange={(e) => setSettings({
                      ...settings,
                      visualizationPreferences: {
                        ...settings.visualizationPreferences,
                        showLabels: e.target.checked
                      }
                    })}
                  />
                }
                label="Show Labels by Default"
              />
            </Grid>
          </Grid>
        </Box>
      )}

      {activeTab === 5 && (
        <Box>
          <RadiatingPromptsSettings
            settings={settings}
            onUpdate={(updatedSettings) => {
              // CRITICAL FIX: Only update the prompts field, preserve everything else
              // The child component should only be updating prompts
              setSettings(prevSettings => ({
                ...prevSettings,
                prompts: updatedSettings.prompts,
                // Ensure model_config is NEVER lost
                model_config: prevSettings.model_config || DEFAULT_SETTINGS.model_config
              }));
            }}
          />
        </Box>
      )}

      {/* Save Button Bar */}
      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
        <Button
          variant="outlined"
          startIcon={<ResetIcon />}
          onClick={() => {
            setSettings(DEFAULT_SETTINGS);
            setSuccess('Settings reset to defaults');
            setTimeout(() => setSuccess(null), 3000);
          }}
        >
          Reset to Defaults
        </Button>
        <Button
          variant="contained"
          startIcon={<SaveIcon />}
          onClick={saveSettings}
          disabled={loading}
        >
          {loading ? 'Saving...' : 'Save All Settings'}
        </Button>
      </Box>

      {/* Preset Dialog */}
      <Dialog
        open={presetDialog}
        onClose={() => setPresetDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {editingPreset ? 'Edit Preset' : 'Add New Preset'}
        </DialogTitle>
        
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Preset Name"
                value={newPreset.name}
                onChange={(e) => setNewPreset({ ...newPreset, name: e.target.value })}
              />
            </Grid>
            
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description"
                multiline
                rows={2}
                value={newPreset.description}
                onChange={(e) => setNewPreset({ ...newPreset, description: e.target.value })}
              />
            </Grid>
            
            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom>
                Configuration
              </Typography>
              <RadiatingDepthControl
                onConfigChange={(config) => setNewPreset({ ...newPreset, config })}
                compact={true}
                hideActions={true}
              />
            </Grid>
          </Grid>
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setPresetDialog(false)}>Cancel</Button>
          <Button variant="contained" onClick={handlePresetSave}>
            {editingPreset ? 'Update' : 'Save'}
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
};

export default RadiatingSettings;