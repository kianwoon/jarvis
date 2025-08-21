import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  TextField,
  Button,
  Alert,
  Divider,
  IconButton,
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
  AccordionDetails,
  Tooltip,
  Card,
  CardContent,
  CardActions,
  CardHeader,
  Slider,
  InputAdornment,
  ToggleButton,
  ToggleButtonGroup,
  Snackbar,
  Stack,
  RadioGroup,
  Radio,
  CircularProgress,
  SelectChangeEvent,
  alpha,
  useTheme
} from '@mui/material';
import {
  Save as SaveIcon,
  RestartAlt as ResetIcon,
  ExpandMore as ExpandIcon,
  Settings as SettingsIcon,
  Speed as PerformanceIcon,
  Output as OutputIcon,
  Memory as CacheIcon,
  PlayCircle as ExecutionIcon,
  Analytics as AdvancedIcon,
  Search as SearchIcon,
  Info as InfoIcon,
  Bookmark as PresetIcon,
  CheckCircle as CheckIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  ModelTraining as ModelIcon,
  HighQuality as QualityIcon,
  Refresh as RefreshIcon,
  Psychology as PsychologyIcon,
  SmartToy as SmartToyIcon,
  AutoAwesome as AutoAwesomeIcon,
  Build as BuildIcon
} from '@mui/icons-material';

interface MetaTaskSettingsProps {
  settings: Record<string, any>;
  onChange: (key: string, value: any) => void;
  category?: string;
}

interface SettingSection {
  title: string;
  icon: React.ReactNode;
  description: string;
  fields: string[];
  defaultExpanded?: boolean;
}

interface PresetConfig {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  settings: Record<string, any>;
}

const MetaTaskSettings: React.FC<MetaTaskSettingsProps> = ({ settings, onChange }) => {
  const theme = useTheme();
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedSections, setExpandedSections] = useState<string[]>(['core']);
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const [showSuccess, setShowSuccess] = useState(false);
  const [localSettings, setLocalSettings] = useState(settings);
  const [activeTab, setActiveTab] = useState(0);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [availableModels, setAvailableModels] = useState<Array<{id: string, name: string, size: string, context_length: string}>>([]);

  // Helper function to safely get nested value from settings
  const getNestedValue = (obj: any, path: string): any => {
    return path.split('.').reduce((current, key) => current?.[key], obj);
  };

  // Helper function to set nested value in settings
  const setNestedValue = (obj: any, path: string, value: any): any => {
    const keys = path.split('.');
    const result = { ...obj };
    let current = result;
    
    for (let i = 0; i < keys.length - 1; i++) {
      const key = keys[i];
      if (!current[key]) {
        current[key] = {};
      } else {
        current[key] = { ...current[key] };
      }
      current = current[key];
    }
    
    current[keys[keys.length - 1]] = value;
    return result;
  };

  // Update local settings when props change
  useEffect(() => {
    setLocalSettings(settings);
  }, [settings]);

  // Fetch available models from Ollama
  const fetchAvailableModels = async () => {
    setModelsLoading(true);
    try {
      const response = await fetch('/api/v1/ollama/models');
      if (response.ok) {
        const data = await response.json();
        const modelList = data.models?.map((model: any) => ({
          id: model.name,
          name: model.name,
          size: model.size || 'Unknown',
          context_length: model.context_length || 'Unknown'
        })) || [];
        setAvailableModels(modelList);
      } else {
        console.error('Failed to fetch models from Ollama:', response.status);
        setAvailableModels([]);
      }
    } catch (error) {
      console.error('Failed to connect to Ollama:', error);
      setAvailableModels([]);
    } finally {
      setModelsLoading(false);
    }
  };

  // Fetch models on mount
  useEffect(() => {
    fetchAvailableModels();
  }, []);

  // Define setting sections with proper nested paths
  const sections: Record<string, SettingSection> = {
    core: {
      title: 'Core Configuration',
      icon: <SettingsIcon />,
      description: 'Main settings and default configurations',
      fields: ['enabled'], // Only 'enabled' is at root level
      defaultExpanded: true
    },
    execution: {
      title: 'Execution Settings',
      icon: <ExecutionIcon />,
      description: 'Phase execution, workflows, and retry logic',
      fields: [
        'execution.max_phases',
        'execution.phase_timeout_minutes', 
        'execution.retry_attempts',
        'execution.enable_streaming',
        'execution.parallel_phases',
        'execution.checkpoint_interval',
        'execution.error_handling'
      ]
    },
    quality: {
      title: 'Quality Control',
      icon: <QualityIcon />,
      description: 'Quality thresholds and validation rules',
      fields: [
        'quality_control.quality_check_model',
        'quality_control.factuality_threshold',
        'quality_control.coherence_threshold',
        'quality_control.completeness_threshold',
        'quality_control.auto_retry_on_low_quality',
        'quality_control.max_refinement_iterations',
        'quality_control.enable_cross_validation'
      ]
    },
    output: {
      title: 'Output Settings',
      icon: <OutputIcon />,
      description: 'Output format, size limits, and validation',
      fields: [
        'output.default_format',
        'output.include_metadata',
        'output.max_output_size_mb',
        'output.output_validation',
        'output.output_deduplication',
        'output.include_phase_outputs',
        'output.compress_large_outputs',
        'output.enable_structured_output'
      ]
    },
    caching: {
      title: 'Caching & Performance',
      icon: <CacheIcon />,
      description: 'Cache configuration and performance tuning',
      fields: [
        'caching.cache_templates',
        'caching.cache_workflows', 
        'caching.cache_ttl_hours',
        'caching.cache_max_size_mb',
        'caching.enable_distributed_cache',
        'caching.cache_eviction_policy',
        'caching.cache_compression'
      ]
    },
    advanced: {
      title: 'Advanced Settings',
      icon: <AdvancedIcon />,
      description: 'Profiling, tracing, and metrics collection',
      fields: [
        'advanced.log_level',
        'advanced.enable_metrics',
        'advanced.enable_tracing',
        'advanced.telemetry_endpoint',
        'advanced.performance_profiling',
        'advanced.debug_mode',
        'advanced.retry_backoff_strategy'
      ]
    }
  };


  // Define preset configurations with nested structure
  const presets: PresetConfig[] = [
    {
      id: 'performance',
      name: 'Performance Mode',
      description: 'Optimized for speed and efficiency',
      icon: <PerformanceIcon />,
      settings: {
        'execution.parallel_phases': true,
        'execution.enable_streaming': true,
        'execution.max_phases': 5,
        'execution.phase_timeout_minutes': 15,
        'caching.cache_templates': true,
        'caching.cache_workflows': true,
        'caching.cache_ttl_hours': 2,
        'output.compress_large_outputs': true,
        'quality_control.factuality_threshold': 0.7,
        'quality_control.coherence_threshold': 0.7,
        'quality_control.completeness_threshold': 0.7
      }
    },
    {
      id: 'quality',
      name: 'Quality Mode',
      description: 'Maximizes output quality and accuracy',
      icon: <QualityIcon />,
      settings: {
        'execution.parallel_phases': false,
        'execution.max_phases': 10,
        'execution.phase_timeout_minutes': 45,
        'quality_control.factuality_threshold': 0.9,
        'quality_control.coherence_threshold': 0.9,
        'quality_control.completeness_threshold': 0.9,
        'quality_control.auto_retry_on_low_quality': true,
        'quality_control.max_refinement_iterations': 5,
        'quality_control.enable_cross_validation': true,
        'output.output_validation': true
      }
    },
    {
      id: 'balanced',
      name: 'Balanced Mode',
      description: 'Balance between speed and quality',
      icon: <CheckIcon />,
      settings: {
        'execution.parallel_phases': false,
        'execution.max_phases': 7,
        'execution.phase_timeout_minutes': 30,
        'caching.cache_templates': true,
        'caching.cache_workflows': true,
        'caching.cache_ttl_hours': 1,
        'quality_control.factuality_threshold': 0.8,
        'quality_control.coherence_threshold': 0.8,
        'quality_control.completeness_threshold': 0.8,
        'quality_control.auto_retry_on_low_quality': true
      }
    }
  ];

  // Helper function to get field label
  const getFieldLabel = (key: string): string => {
    // Extract the last part of the path for nested fields
    const actualKey = key.includes('.') ? key.split('.').pop()! : key;
    
    // Remove prefixes and convert to readable format
    const cleanKey = actualKey
      .replace(/^meta_task_/, '')
      .replace(/_/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase());
    
    // Special cases for better readability
    const labelMap: Record<string, string> = {
      'Ttl': 'TTL',
      'Mb': 'MB',
      'Ms': 'ms',
      'Llm': 'LLM',
      'Api': 'API',
      'Url': 'URL',
      'Max Output Size Mb': 'Max Output Size (MB)',
      'Cache Ttl Hours': 'Cache TTL (hours)',
      'Phase Timeout Minutes': 'Phase Timeout (minutes)',
      'Cache Max Size Mb': 'Cache Max Size (MB)',
      'Quality Check Model': 'Quality Check Model',
      'Factuality Threshold': 'Factuality Threshold',
      'Coherence Threshold': 'Coherence Threshold',
      'Completeness Threshold': 'Completeness Threshold',
      'Max Refinement Iterations': 'Max Refinement Iterations',
      'Enable Cross Validation': 'Enable Cross Validation',
      'Enable Distributed Cache': 'Enable Distributed Cache',
      'Cache Eviction Policy': 'Cache Eviction Policy',
      'Enable Streaming': 'Enable Streaming',
      'Parallel Phases': 'Parallel Phases',
      'Checkpoint Interval': 'Checkpoint Interval',
      'Error Handling': 'Error Handling',
      'Output Deduplication': 'Output Deduplication',
      'Include Phase Outputs': 'Include Phase Outputs',
      'Compress Large Outputs': 'Compress Large Outputs',
      'Enable Structured Output': 'Enable Structured Output',
      'Log Level': 'Log Level',
      'Enable Metrics': 'Enable Metrics',
      'Enable Tracing': 'Enable Tracing',
      'Telemetry Endpoint': 'Telemetry Endpoint',
      'Performance Profiling': 'Performance Profiling',
      'Debug Mode': 'Debug Mode',
      'Retry Backoff Strategy': 'Retry Backoff Strategy',
      'Cache Compression': 'Cache Compression'
    };

    return labelMap[cleanKey] || cleanKey;
  };

  // Helper function to get field description
  const getFieldDescription = (key: string): string => {
    // Extract the last part for nested fields
    const actualKey = key.includes('.') ? key.split('.').pop()! : key;
    
    const descriptions: Record<string, string> = {
      enabled: 'Enable or disable meta-task processing',
      max_phases: 'Maximum number of execution phases allowed',
      phase_timeout_minutes: 'Timeout for each phase in minutes',
      retry_attempts: 'Number of retry attempts on failure',
      enable_streaming: 'Enable streaming responses for real-time output',
      parallel_phases: 'Execute multiple phases in parallel when possible',
      checkpoint_interval: 'Number of phases between checkpoints',
      error_handling: 'Strategy for handling errors during execution',
      quality_check_model: 'Model used for quality assessment',
      factuality_threshold: 'Minimum factuality score (0-1)',
      coherence_threshold: 'Minimum coherence score (0-1)',
      completeness_threshold: 'Minimum completeness score (0-1)',
      auto_retry_on_low_quality: 'Automatically retry when quality is below threshold',
      max_refinement_iterations: 'Maximum refinement attempts',
      enable_cross_validation: 'Enable cross-validation for quality checks',
      default_format: 'Default output format (markdown, json, html)',
      include_metadata: 'Include execution metadata in output',
      max_output_size_mb: 'Maximum output size in megabytes',
      output_validation: 'Validate output against expected format',
      output_deduplication: 'Remove duplicate content from output',
      include_phase_outputs: 'Include intermediate phase outputs',
      compress_large_outputs: 'Compress outputs exceeding size limits',
      enable_structured_output: 'Enable structured output formatting',
      cache_templates: 'Cache task templates for reuse',
      cache_workflows: 'Cache workflow definitions',
      cache_ttl_hours: 'Cache time-to-live in hours',
      cache_max_size_mb: 'Maximum cache size in megabytes',
      enable_distributed_cache: 'Enable distributed caching across nodes',
      cache_eviction_policy: 'Cache eviction policy (LRU, LFU, FIFO)',
      cache_compression: 'Enable cache compression to save space',
      log_level: 'Logging level for debugging',
      enable_metrics: 'Collect performance metrics',
      enable_tracing: 'Enable distributed tracing',
      telemetry_endpoint: 'Endpoint for telemetry data',
      performance_profiling: 'Enable performance profiling',
      debug_mode: 'Enable debug mode for detailed logging',
      retry_backoff_strategy: 'Backoff strategy for retries'
    };
    
    return descriptions[actualKey] || `Configure ${getFieldLabel(key).toLowerCase()}`;
  };

  // Helper function to render field based on type
  const renderField = (key: string, value: any) => {
    const label = getFieldLabel(key);
    const description = getFieldDescription(key);
    
    // Get the actual value from nested structure
    const actualValue = getNestedValue(localSettings, key);
    
    // Boolean fields - use Switch
    if (typeof actualValue === 'boolean' || key.includes('enabled') || key.includes('enable') || key.includes('require')) {
      return (
        <FormControlLabel
          control={
            <Switch
              checked={Boolean(actualValue)}
              onChange={(e) => handleChange(key, e.target.checked)}
              color="primary"
            />
          }
          label={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography>{label}</Typography>
              <Tooltip title={description}>
                <InfoIcon fontSize="small" sx={{ opacity: 0.6 }} />
              </Tooltip>
            </Box>
          }
        />
      );
    }
    
    // Numeric fields with ranges - use Slider
    if (key.includes('threshold')) {
      const min = 0;
      const max = 1;
      const step = 0.05;
      
      return (
        <Box sx={{ px: 1 }}>
          <Typography gutterBottom>
            {label}
            <Tooltip title={description}>
              <IconButton size="small" sx={{ ml: 1 }}>
                <InfoIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Typography>
          <Slider
            value={Number(actualValue) || 0.7}
            onChange={(_, newValue) => handleChange(key, newValue)}
            min={min}
            max={max}
            step={step}
            marks
            valueLabelDisplay="auto"
            sx={{ mt: 2, mb: 1 }}
          />
        </Box>
      );
    }
    
    // Select fields
    if (key.includes('default_format')) {
      return (
        <FormControl fullWidth variant="outlined" size="small">
          <InputLabel>{label}</InputLabel>
          <Select
            value={actualValue || 'markdown'}
            onChange={(e) => handleChange(key, e.target.value)}
            label={label}
          >
            <MenuItem value="markdown">Markdown</MenuItem>
            <MenuItem value="json">JSON</MenuItem>
            <MenuItem value="html">HTML</MenuItem>
            <MenuItem value="plain">Plain Text</MenuItem>
          </Select>
        </FormControl>
      );
    }
    
    if (key.includes('cache_eviction_policy')) {
      return (
        <FormControl fullWidth variant="outlined" size="small">
          <InputLabel>{label}</InputLabel>
          <Select
            value={actualValue || 'LRU'}
            onChange={(e) => handleChange(key, e.target.value)}
            label={label}
          >
            <MenuItem value="LRU">LRU (Least Recently Used)</MenuItem>
            <MenuItem value="LFU">LFU (Least Frequently Used)</MenuItem>
            <MenuItem value="FIFO">FIFO (First In First Out)</MenuItem>
          </Select>
        </FormControl>
      );
    }
    
    if (key.includes('error_handling')) {
      return (
        <FormControl fullWidth variant="outlined" size="small">
          <InputLabel>{label}</InputLabel>
          <Select
            value={actualValue || 'retry'}
            onChange={(e) => handleChange(key, e.target.value)}
            label={label}
          >
            <MenuItem value="retry">Retry</MenuItem>
            <MenuItem value="fallback">Fallback</MenuItem>
            <MenuItem value="fail_fast">Fail Fast</MenuItem>
          </Select>
        </FormControl>
      );
    }
    
    if (key.includes('log_level')) {
      return (
        <FormControl fullWidth variant="outlined" size="small">
          <InputLabel>{label}</InputLabel>
          <Select
            value={actualValue || 'INFO'}
            onChange={(e) => handleChange(key, e.target.value)}
            label={label}
          >
            <MenuItem value="DEBUG">DEBUG</MenuItem>
            <MenuItem value="INFO">INFO</MenuItem>
            <MenuItem value="WARNING">WARNING</MenuItem>
            <MenuItem value="ERROR">ERROR</MenuItem>
          </Select>
        </FormControl>
      );
    }
    
    if (key.includes('retry_backoff_strategy')) {
      return (
        <FormControl fullWidth variant="outlined" size="small">
          <InputLabel>{label}</InputLabel>
          <Select
            value={actualValue || 'exponential'}
            onChange={(e) => handleChange(key, e.target.value)}
            label={label}
          >
            <MenuItem value="linear">Linear</MenuItem>
            <MenuItem value="exponential">Exponential</MenuItem>
            <MenuItem value="fibonacci">Fibonacci</MenuItem>
          </Select>
        </FormControl>
      );
    }
    
    // Number fields with units
    if (key.includes('timeout') || key.includes('ttl') || key.includes('size') || 
        key.includes('limit') || key.includes('phases') || key.includes('attempts') ||
        key.includes('iterations') || key.includes('interval')) {
      let unit = '';
      if (key.includes('minutes')) unit = 'min';
      else if (key.includes('hours')) unit = 'hrs';
      else if (key.includes('mb')) unit = 'MB';
      else if (key.includes('ms')) unit = 'ms';
      else if (key.includes('interval')) unit = 'phases';
      
      return (
        <TextField
          fullWidth
          label={label}
          type="number"
          value={actualValue ?? ''}
          onChange={(e) => handleChange(key, parseInt(e.target.value) || 0)}
          size="small"
          variant="outlined"
          InputProps={{
            endAdornment: unit ? <InputAdornment position="end">{unit}</InputAdornment> : undefined
          }}
          helperText={description}
        />
      );
    }
    
    // Default text field
    return (
      <TextField
        fullWidth
        label={label}
        value={actualValue ?? ''}
        onChange={(e) => handleChange(key, e.target.value)}
        size="small"
        variant="outlined"
        helperText={description}
      />
    );
  };

  // Handle setting change
  const handleChange = (key: string, value: any) => {
    const newSettings = setNestedValue(localSettings, key, value);
    setLocalSettings(newSettings);
    // Pass the full nested path and value to parent
    onChange(key, value);
  };

  // Apply preset
  const applyPreset = (preset: PresetConfig) => {
    Object.entries(preset.settings).forEach(([key, value]) => {
      handleChange(key, value);
    });
    setSelectedPreset(preset.id);
    setShowSuccess(true);
  };

  // Reset section to defaults
  const resetSection = (sectionKey: string) => {
    const section = sections[sectionKey];
    section.fields.forEach(field => {
      // Reset to default values based on field type
      if (field === 'enabled') {
        handleChange(field, true);
      } else if (field.includes('enabled') || field.includes('enable')) {
        handleChange(field, false);
      } else if (field.includes('threshold')) {
        handleChange(field, 0.8);
      } else if (field.includes('max_phases')) {
        handleChange(field, 10);
      } else if (field.includes('phase_timeout_minutes')) {
        handleChange(field, 30);
      } else if (field.includes('cache_ttl_hours')) {
        handleChange(field, 72);
      } else if (field.includes('max_output_size_mb')) {
        handleChange(field, 10);
      } else if (field.includes('cache_max_size_mb')) {
        handleChange(field, 500);
      } else if (field.includes('retry_attempts')) {
        handleChange(field, 3);
      } else if (field.includes('max_refinement_iterations')) {
        handleChange(field, 3);
      } else if (field.includes('checkpoint_interval')) {
        handleChange(field, 5);
      } else if (field.includes('default_format')) {
        handleChange(field, 'markdown');
      } else if (field.includes('error_handling')) {
        handleChange(field, 'retry');
      } else if (field.includes('cache_eviction_policy')) {
        handleChange(field, 'LRU');
      } else if (field.includes('log_level')) {
        handleChange(field, 'INFO');
      } else if (field.includes('retry_backoff_strategy')) {
        handleChange(field, 'exponential');
      } else if (field.includes('quality_check_model')) {
        handleChange(field, 'qwen3:30b-a3b-instruct-2507-q4_K_M');
      } else {
        handleChange(field, '');
      }
    });
  };

  // Filter sections based on search
  const filteredSections = useMemo(() => {
    if (!searchQuery) return sections;
    
    const query = searchQuery.toLowerCase();
    const filtered: Record<string, SettingSection> = {};
    
    Object.entries(sections).forEach(([key, section]) => {
      const matchingFields = section.fields.filter(field => 
        field.toLowerCase().includes(query) ||
        getFieldLabel(field).toLowerCase().includes(query) ||
        getFieldDescription(field).toLowerCase().includes(query)
      );
      
      if (matchingFields.length > 0 || 
          section.title.toLowerCase().includes(query) ||
          section.description.toLowerCase().includes(query)) {
        filtered[key] = {
          ...section,
          fields: matchingFields.length > 0 ? matchingFields : section.fields
        };
      }
    });
    
    return filtered;
  }, [searchQuery]);

  // Handle accordion change
  const handleAccordionChange = (panel: string) => (_: React.SyntheticEvent, isExpanded: boolean) => {
    setExpandedSections(prev => 
      isExpanded 
        ? [...prev, panel]
        : prev.filter(p => p !== panel)
    );
  };

  // Render main settings tab content
  const renderSettingsTab = () => (
    <Box sx={{ width: '100%' }}>
      {/* Preset buttons */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="subtitle2" gutterBottom>
          Quick Presets
        </Typography>
        <Stack direction="row" spacing={2} sx={{ justifyContent: 'space-between', alignItems: 'center' }}>
          <Stack direction="row" spacing={2}>
            {presets.map(preset => (
              <Tooltip key={preset.id} title={preset.description}>
                <Button
                  variant={selectedPreset === preset.id ? 'contained' : 'outlined'}
                  startIcon={preset.icon}
                  onClick={() => applyPreset(preset)}
                  size="small"
                >
                  {preset.name}
                </Button>
              </Tooltip>
            ))}
          </Stack>
          <Button
            startIcon={<RefreshIcon />}
            onClick={async () => {
              try {
                const response = await fetch('/api/v1/settings/meta-task/cache/reload', {
                  method: 'POST'
                });
                if (response.ok) {
                  setShowSuccess(true);
                }
              } catch (error) {
                console.error('Failed to reload cache:', error);
              }
            }}
            size="small"
            variant="outlined"
          >
            Reload Cache
          </Button>
        </Stack>
      </Paper>

      {/* Settings sections with accordions */}
      <Grid container spacing={3}>
        <Grid item xs={12} lg={6}>
          {Object.entries(filteredSections)
            .slice(0, Math.ceil(Object.keys(filteredSections).length / 2))
            .map(([key, section]) => (
              <Accordion
                key={key}
                expanded={expandedSections.includes(key)}
                onChange={handleAccordionChange(key)}
                sx={{ mb: 2 }}
              >
                <AccordionSummary expandIcon={<ExpandIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                    {section.icon}
                    <Box sx={{ flexGrow: 1 }}>
                      <Typography variant="h6">{section.title}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {section.description}
                      </Typography>
                    </Box>
                    <Chip 
                      label={`${section.fields.length} settings`} 
                      size="small" 
                      sx={{ mr: 2 }}
                    />
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    {section.fields.map(field => (
                      <Grid item xs={12} key={field}>
                        {renderField(field, localSettings[field])}
                      </Grid>
                    ))}
                  </Grid>
                  <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                    <Button
                      size="small"
                      startIcon={<ResetIcon />}
                      onClick={() => resetSection(key)}
                    >
                      Reset Section
                    </Button>
                  </Box>
                </AccordionDetails>
              </Accordion>
            ))}
        </Grid>

        <Grid item xs={12} lg={6}>
          {Object.entries(filteredSections)
            .slice(Math.ceil(Object.keys(filteredSections).length / 2))
            .map(([key, section]) => (
              <Accordion
                key={key}
                expanded={expandedSections.includes(key)}
                onChange={handleAccordionChange(key)}
                sx={{ mb: 2 }}
              >
                <AccordionSummary expandIcon={<ExpandIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                    {section.icon}
                    <Box sx={{ flexGrow: 1 }}>
                      <Typography variant="h6">{section.title}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {section.description}
                      </Typography>
                    </Box>
                    <Chip 
                      label={`${section.fields.length} settings`} 
                      size="small" 
                      sx={{ mr: 2 }}
                    />
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    {section.fields.map(field => (
                      <Grid item xs={12} key={field}>
                        {renderField(field, localSettings[field])}
                      </Grid>
                    ))}
                  </Grid>
                  <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                    <Button
                      size="small"
                      startIcon={<ResetIcon />}
                      onClick={() => resetSection(key)}
                    >
                      Reset Section
                    </Button>
                  </Box>
                </AccordionDetails>
              </Accordion>
            ))}
        </Grid>
      </Grid>
    </Box>
  );

  // Render individual model tab with exact card-based design from LLM settings
  const renderModelTab = (modelType: string) => {
    const modelName = modelType.charAt(0).toUpperCase() + modelType.slice(1);
    const modelKey = `${modelType}_model`;
    const modelConfig = localSettings[modelKey] || {};
    
    return (
      <Box sx={{ width: '100%' }}>

        {/* Single Card with ALL fields - matching AI Models & LLM page */}
        <Card variant="outlined">
          <CardHeader 
            title={`${modelName} Model Configuration`}
            subheader={
              modelType === 'analyzer' ? "Configure the model for task analysis" : 
              modelType === 'reviewer' ? "Configure the model for output review" : 
              modelType === 'assembler' ? "Configure the model for result assembly" :
              "Configure the model for content generation"
            }
          />
          <CardContent>
            <Grid container spacing={3}>

              {/* Model Selection Dropdown */}
              <Grid item xs={12}>
                <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
                  <FormControl fullWidth>
                    <InputLabel id={`${modelType}-model-label`}>Model</InputLabel>
                    <Select
                      labelId={`${modelType}-model-label`}
                      value={modelConfig.model || ''}
                      label="Model"
                      onChange={(e: SelectChangeEvent) => {
                      const newModelName = e.target.value;
                      
                      // Handle custom model input
                      if (newModelName === 'custom') {
                        const customModel = prompt('Enter custom model name (e.g., mistral:7b, gemma2:9b):');
                        if (customModel) {
                          handleChange(`${modelKey}.model`, customModel);
                        }
                        return;
                      }
                      
                      handleChange(`${modelKey}.model`, newModelName);
                      
                      // Auto-update context length if available
                      const selectedModel = availableModels.find(m => m.name === newModelName);
                      if (selectedModel && selectedModel.context_length !== 'Unknown') {
                        const contextLength = parseInt(selectedModel.context_length.replace(/,/g, ''));
                        if (!isNaN(contextLength)) {
                          handleChange(`${modelKey}.context_length`, contextLength);
                          handleChange(`${modelKey}.max_tokens`, Math.floor(contextLength * 0.75));
                        }
                      }
                    }}
                    disabled={modelsLoading}
                  >
                    {availableModels.length > 0 ? (
                      availableModels.map((model) => (
                        <MenuItem key={model.id} value={model.name}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Typography>{model.name}</Typography>
                              {modelConfig.model === model.name && (
                                <Chip 
                                  label="Active" 
                                  size="small" 
                                  color="success" 
                                  icon={<CheckCircleIcon />}
                                />
                              )}
                            </Box>
                            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                              <Typography variant="caption" color="text.secondary">
                                {model.size}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                Context: {model.context_length}
                              </Typography>
                            </Box>
                          </Box>
                        </MenuItem>
                      ))
                    ) : (
                      <MenuItem disabled>
                        <Typography color="text.secondary">
                          {modelsLoading ? 'Loading models...' : 'No Ollama models available. Please check if Ollama server is running.'}
                        </Typography>
                      </MenuItem>
                    )}
                    <Divider />
                    <MenuItem value="custom">
                      <Typography color="primary">Enter custom model name...</Typography>
                    </MenuItem>
                    </Select>
                  </FormControl>
                  <Tooltip title="Refresh available models">
                    <IconButton 
                      onClick={fetchAvailableModels}
                      disabled={modelsLoading}
                      size="small"
                      sx={{ mb: 0.5 }}
                    >
                      {modelsLoading ? <CircularProgress size={20} /> : <RefreshIcon />}
                    </IconButton>
                  </Tooltip>
                </Box>
              </Grid>

              {/* Temperature */}
              <Grid item xs={12} md={6}>
                <Box>
                  <Typography gutterBottom>Temperature</Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Slider
                      value={modelConfig.temperature || 0.7}
                      onChange={(e, newValue) => handleChange(`${modelKey}.temperature`, newValue)}
                      min={0}
                      max={2}
                      step={0.1}
                      marks
                      valueLabelDisplay="auto"
                      sx={{ flex: 1 }}
                    />
                    <TextField
                      type="number"
                      value={modelConfig.temperature || 0.7}
                      onChange={(e) => handleChange(`${modelKey}.temperature`, parseFloat(e.target.value))}
                      inputProps={{ min: 0, max: 2, step: 0.1 }}
                      size="small"
                      sx={{ width: 80 }}
                    />
                  </Box>
                  <Typography variant="caption" color="text.secondary">
                    Controls randomness. Lower values make output more focused.
                  </Typography>
                </Box>
              </Grid>

              {/* Max Tokens */}
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Max Tokens"
                  type="number"
                  value={modelConfig.max_tokens || 4096}
                  onChange={(e) => handleChange(`${modelKey}.max_tokens`, parseInt(e.target.value))}
                  inputProps={{ min: 1, max: 128000 }}
                  size="small"
                />
              </Grid>

              {/* Top P */}
              <Grid item xs={12} md={6}>
                <Box>
                  <Typography gutterBottom>Top P</Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Slider
                      value={modelConfig.top_p || 0.9}
                      onChange={(e, newValue) => handleChange(`${modelKey}.top_p`, newValue)}
                      min={0}
                      max={1}
                      step={0.05}
                      marks
                      valueLabelDisplay="auto"
                      sx={{ flex: 1 }}
                    />
                    <TextField
                      type="number"
                      value={modelConfig.top_p || 0.9}
                      onChange={(e) => handleChange(`${modelKey}.top_p`, parseFloat(e.target.value))}
                      inputProps={{ min: 0, max: 1, step: 0.05 }}
                      size="small"
                      sx={{ width: 80 }}
                    />
                  </Box>
                  <Typography variant="caption" color="text.secondary">
                    Nucleus sampling threshold.
                  </Typography>
                </Box>
              </Grid>

              {/* Model Server URL */}
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Model Server URL"
                  type="text"
                  value={modelConfig.model_server || 'http://localhost:11434'}
                  onChange={(e) => handleChange(`${modelKey}.model_server`, e.target.value)}
                  placeholder="e.g., http://localhost:11434"
                  size="small"
                />
              </Grid>

              {/* System Prompt */}
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  multiline
                  minRows={8}
                  maxRows={30}
                  variant="outlined"
                  label="System Prompt"
                  value={modelConfig.system_prompt || ''}
                  onChange={(e) => handleChange(`${modelKey}.system_prompt`, e.target.value)}
                  placeholder={`Enter the system prompt for the ${modelType} model...`}
                  InputProps={{
                    sx: {
                      resize: 'vertical',
                      overflow: 'auto',
                      '& textarea': {
                        resize: 'vertical',
                        overflow: 'auto !important',
                        cursor: 'text',
                      },
                      '&::after': {
                        content: '""',
                        position: 'absolute',
                        bottom: 0,
                        right: 0,
                        width: 0,
                        height: 0,
                        borderStyle: 'solid',
                        borderWidth: '0 0 12px 12px',
                        borderColor: 'transparent transparent rgba(0, 0, 0, 0.2) transparent',
                        cursor: 'ns-resize',
                        pointerEvents: 'none',
                      },
                      '&:hover::after': {
                        borderColor: 'transparent transparent rgba(0, 0, 0, 0.4) transparent',
                      }
                    }
                  }}
                />
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Box>
    );
  };

  return (
    <Box sx={{ width: '100%', maxWidth: 1400, mx: 'auto' }}>
      {/* Top-level tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs 
          value={activeTab} 
          onChange={(_, newValue) => setActiveTab(newValue)}
          variant="fullWidth"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab label="Analyzer Model" icon={<PsychologyIcon />} iconPosition="start" />
          <Tab label="Generator Model" icon={<AutoAwesomeIcon />} iconPosition="start" />
          <Tab label="Reviewer Model" icon={<SmartToyIcon />} iconPosition="start" />
          <Tab label="Assembler Model" icon={<BuildIcon />} iconPosition="start" />
          <Tab label="Settings" icon={<SettingsIcon />} iconPosition="start" />
        </Tabs>
      </Paper>

      {/* Search bar */}
      {activeTab === 4 && (
        <Paper sx={{ p: 2, mb: 3 }}>
          <TextField
            fullWidth
            placeholder="Search settings..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            size="small"
            InputProps={{
              startAdornment: <SearchIcon sx={{ mr: 1, opacity: 0.6 }} />
            }}
          />
        </Paper>
      )}

      {/* Tab content */}
      <Box sx={{ p: 2 }}>
        {activeTab === 0 && renderModelTab('analyzer')}
        {activeTab === 1 && renderModelTab('generator')}
        {activeTab === 2 && renderModelTab('reviewer')}
        {activeTab === 3 && renderModelTab('assembler')}
        {activeTab === 4 && renderSettingsTab()}
      </Box>


      {/* Success notification */}
      <Snackbar
        open={showSuccess}
        autoHideDuration={3000}
        onClose={() => setShowSuccess(false)}
        message="Settings applied successfully"
      />
    </Box>
  );
};

export default MetaTaskSettings;