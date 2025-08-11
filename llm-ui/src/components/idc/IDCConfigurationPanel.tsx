import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  TextField,
  Button,
  Card,
  CardContent,
  CardHeader,
  Alert,
  Divider,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  IconButton,
  Tooltip,
  Chip,
  Slider,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  Save as SaveIcon,
  Refresh as RefreshIcon,
  CheckCircle as CheckCircleIcon,
  Settings as SettingsIcon,
  Code as CodeIcon,
  Gavel as GavelIcon,
  Compare as CompareIcon,
  Description as DescriptionIcon
} from '@mui/icons-material';

interface IDCConfiguration {
  extraction: {
    model: string;
    model_server?: string;
    system_prompt: string;
    max_tokens: number;
    temperature: number;
    top_p: number;
    top_k: number;
    enable_chunking: boolean;
    chunk_size: number;
    chunk_overlap: number;
    max_context_usage: number;
    confidence_threshold: number;
  };
  validation: {
    model: string;
    model_server?: string;
    system_prompt: string;
    max_tokens: number;
    temperature: number;
    top_p: number;
    max_context_usage: number;
    confidence_threshold: number;
    enable_structured_output: boolean;
  };
  comparison: {
    algorithm: string;
    similarity_threshold: number;
    ignore_whitespace: boolean;
    ignore_case: boolean;
    enable_fuzzy_matching: boolean;
    fuzzy_threshold: number;
  };
}

interface IDCConfigurationPanelProps {
  onConfigChange?: (config: IDCConfiguration) => void;
  onShowSuccess?: (message?: string) => void;
}

const IDCConfigurationPanel: React.FC<IDCConfigurationPanelProps> = ({
  onConfigChange,
  onShowSuccess
}) => {
  const [config, setConfig] = useState<IDCConfiguration | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [availableModels, setAvailableModels] = useState<Array<{
    name: string;
    id: string;
    size: string;
    modified: string;
    context_length: string;
  }>>([]);
  const [loadingModels, setLoadingModels] = useState(false);

  useEffect(() => {
    loadConfiguration();
  }, []);

  // Fix for model IDs that were incorrectly saved instead of model names
  useEffect(() => {
    if (config && availableModels.length > 0) {
      let needsUpdate = false;
      let updatedConfig = { ...config };

      // Check if extraction model is using an ID instead of name
      const extractionModelById = availableModels.find(m => m.id === config.extraction.model);
      if (extractionModelById) {
        updatedConfig.extraction.model = extractionModelById.name;
        needsUpdate = true;
        console.log('Migrating extraction model from ID to name:', extractionModelById.id, '->', extractionModelById.name);
      }

      // Check if validation model is using an ID instead of name
      const validationModelById = availableModels.find(m => m.id === config.validation.model);
      if (validationModelById) {
        updatedConfig.validation.model = validationModelById.name;
        needsUpdate = true;
        console.log('Migrating validation model from ID to name:', validationModelById.id, '->', validationModelById.name);
      }

      if (needsUpdate) {
        setConfig(updatedConfig);
      }
    }
  }, [config?.extraction.model, config?.validation.model, availableModels]);

  // Debug logging to verify system prompts are loaded
  useEffect(() => {
    if (config) {
      console.log('IDC Configuration loaded:', {
        hasExtractionPrompt: !!config.extraction?.system_prompt,
        extractionPromptLength: config.extraction?.system_prompt?.length || 0,
        hasValidationPrompt: !!config.validation?.system_prompt,
        validationPromptLength: config.validation?.system_prompt?.length || 0,
        extractionPromptPreview: config.extraction?.system_prompt?.substring(0, 50) + '...',
        validationPromptPreview: config.validation?.system_prompt?.substring(0, 50) + '...'
      });
    }
  }, [config]);

  // Helper function to parse context length from string (e.g., "262,144" -> 262144)
  const parseContextLength = (contextStr: string): number => {
    if (!contextStr || contextStr === 'N/A') return 128000; // Default fallback
    // Remove commas and parse
    const cleaned = contextStr.replace(/,/g, '').replace(/[^0-9]/g, '');
    const parsed = parseInt(cleaned);
    return isNaN(parsed) ? 128000 : parsed;
  };

  // Helper function to calculate max tokens based on context usage
  const calculateMaxTokens = (modelName: string, contextUsagePercent: number): number => {
    const model = availableModels.find(m => m.name === modelName);
    if (!model) return 8192; // Default fallback
    
    const contextLength = parseContextLength(model.context_length);
    const maxTokens = Math.floor(contextLength * contextUsagePercent);
    
    // Ensure within reasonable bounds
    return Math.min(Math.max(maxTokens, 1000), contextLength);
  };

  // Recalculate max tokens when models are loaded or config changes
  useEffect(() => {
    if (config && availableModels.length > 0) {
      // Check if max tokens need recalculation for extraction
      const extractionMaxTokens = calculateMaxTokens(
        config.extraction.model, 
        config.extraction.max_context_usage
      );
      
      // Check if max tokens need recalculation for validation
      const validationMaxTokens = calculateMaxTokens(
        config.validation.model,
        config.validation.max_context_usage
      );

      // Only update if values have changed to avoid infinite loops
      if (config.extraction.max_tokens !== extractionMaxTokens || 
          config.validation.max_tokens !== validationMaxTokens) {
        setConfig({
          ...config,
          extraction: {
            ...config.extraction,
            max_tokens: extractionMaxTokens
          },
          validation: {
            ...config.validation,
            max_tokens: validationMaxTokens
          }
        });
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [availableModels.length, config?.extraction.model, config?.validation.model, 
      config?.extraction.max_context_usage, config?.validation.max_context_usage]);

  const loadConfiguration = async () => {
    setLoading(true);
    setLoadingModels(true);
    try {
      const response = await fetch('/api/v1/idc/configuration');
      if (response.ok) {
        const data = await response.json();
        setConfig(data.configuration);
        
        // Set available models from the same response
        if (data.available_models && Array.isArray(data.available_models)) {
          setAvailableModels(data.available_models);
        }
        
        if (onConfigChange) {
          onConfigChange(data.configuration);
        }
      } else {
        console.error('Failed to load IDC configuration');
      }
    } catch (error) {
      console.error('Error loading IDC configuration:', error);
    } finally {
      setLoading(false);
      setLoadingModels(false);
    }
  };

  const fetchAvailableModels = async () => {
    // This is now handled by loadConfiguration()
    // Keeping this function for backward compatibility but it's no longer used
  };

  const saveConfiguration = async () => {
    if (!config) return;
    
    setSaving(true);
    try {
      const response = await fetch('/api/v1/idc/configuration', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          extraction_model: config.extraction.model,
          extraction_model_server: config.extraction.model_server || 'http://host.docker.internal:11434',
          extraction_system_prompt: config.extraction.system_prompt,
          extraction_temperature: config.extraction.temperature,
          extraction_max_tokens: config.extraction.max_tokens,
          extraction_max_context_usage: config.extraction.max_context_usage,
          extraction_confidence_threshold: config.extraction.confidence_threshold,
          enable_chunking: config.extraction.enable_chunking,
          chunk_size: config.extraction.chunk_size,
          chunk_overlap: config.extraction.chunk_overlap,
          validation_model: config.validation.model,
          validation_model_server: config.validation.model_server || 'http://host.docker.internal:11434',
          validation_system_prompt: config.validation.system_prompt,
          validation_temperature: config.validation.temperature,
          validation_max_tokens: config.validation.max_tokens,
          max_context_usage: config.validation.max_context_usage,
          quality_threshold: config.validation.confidence_threshold,
          enable_structured_output: config.validation.enable_structured_output,
          comparison_algorithm: config.comparison.algorithm,
          similarity_threshold: config.comparison.similarity_threshold,
          fuzzy_threshold: config.comparison.fuzzy_threshold,
          ignore_case: config.comparison.ignore_case,
          ignore_whitespace: config.comparison.ignore_whitespace,
          enable_fuzzy_matching: config.comparison.enable_fuzzy_matching
        })
      });

      if (response.ok) {
        if (onShowSuccess) {
          onShowSuccess('IDC configuration saved successfully');
        }
        await loadConfiguration(); // Reload to ensure consistency
      } else {
        console.error('Failed to save configuration');
      }
    } catch (error) {
      console.error('Error saving configuration:', error);
    } finally {
      setSaving(false);
    }
  };

  const reloadConfiguration = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/idc/configuration/reload', {
        method: 'POST'
      });
      if (response.ok) {
        await loadConfiguration();
        if (onShowSuccess) {
          onShowSuccess('Configuration reloaded from database');
        }
      }
    } catch (error) {
      console.error('Error reloading configuration:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFieldChange = (section: keyof IDCConfiguration, field: string, value: any) => {
    if (!config) return;
    
    let updatedSection = {
      ...config[section],
      [field]: value
    };

    // Auto-calculate max tokens when model or context usage changes
    if (section === 'extraction' || section === 'validation') {
      if (field === 'model' || field === 'max_context_usage') {
        const modelName = field === 'model' ? value : config[section].model;
        const contextUsage = field === 'max_context_usage' ? value : config[section].max_context_usage;
        
        // Calculate new max tokens
        const newMaxTokens = calculateMaxTokens(modelName, contextUsage);
        updatedSection = {
          ...updatedSection,
          max_tokens: newMaxTokens
        };
      }
    }
    
    setConfig({
      ...config,
      [section]: updatedSection
    });
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (!config) {
    return (
      <Alert severity="error">
        Failed to load IDC configuration. Please refresh the page.
      </Alert>
    );
  }

  // Format file size helper
  const formatFileSize = (sizeStr: string): string => {
    if (!sizeStr || sizeStr === 'Unknown') return sizeStr;
    if (sizeStr.includes('MB') || sizeStr.includes('GB') || sizeStr.includes('KB')) return sizeStr;
    
    const bytes = parseInt(sizeStr);
    if (isNaN(bytes)) return sizeStr;
    
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <Box>
      {/* Header Actions - Following Settings Page Pattern */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <SettingsIcon />
          IDC Configuration
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={reloadConfiguration}
            disabled={loading}
          >
            Reload
          </Button>
          <Button
            variant="contained"
            startIcon={<SaveIcon />}
            onClick={saveConfiguration}
            disabled={saving}
          >
            {saving ? 'Saving...' : 'Save Configuration'}
          </Button>
        </Box>
      </Box>

      {/* Extraction Configuration - Following Settings Card Pattern */}
      <Card sx={{ mb: 3 }}>
        <CardHeader
          avatar={<CodeIcon />}
          title="Extraction Configuration"
          subheader="Configure how documents are extracted to structured markdown"
        />
        <CardContent>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Extraction Model</InputLabel>
                <Select
                  value={config.extraction.model}
                  onChange={(e: SelectChangeEvent) => handleFieldChange('extraction', 'model', e.target.value)}
                  label="Extraction Model"
                  disabled={loadingModels}
                >
                  {availableModels.map((model) => (
                    <MenuItem key={model.id} value={model.name}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                        <Typography>{model.name}</Typography>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <Chip label={formatFileSize(model.size)} size="small" variant="outlined" />
                          <Chip label={model.context_length || 'N/A'} size="small" color="primary" variant="outlined" />
                        </Box>
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Model Server URL"
                value={config.extraction.model_server || ''}
                onChange={(e) => handleFieldChange('extraction', 'model_server', e.target.value)}
                placeholder="http://host.docker.internal:11434"
                helperText="Ollama server URL for extraction model"
              />
            </Grid>
            
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                label="Temperature"
                type="number"
                value={config.extraction.temperature}
                onChange={(e) => handleFieldChange('extraction', 'temperature', parseFloat(e.target.value))}
                inputProps={{ min: 0, max: 2, step: 0.1 }}
                helperText="Lower = more consistent"
              />
            </Grid>
            
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                label="Max Tokens (Auto)"
                type="number"
                value={config.extraction.max_tokens}
                InputProps={{
                  readOnly: true,
                }}
                helperText="Auto-calculated from context %"
                sx={{
                  '& .MuiInputBase-input': {
                    backgroundColor: 'action.hover',
                  }
                }}
              />
            </Grid>

            <Grid item xs={12} md={4}>
              <Box>
                <Typography gutterBottom>
                  Max Context Usage: {Math.round(config.extraction.max_context_usage * 100)}%
                </Typography>
                <Slider
                  value={config.extraction.max_context_usage * 100}
                  onChange={(_, value) => handleFieldChange('extraction', 'max_context_usage', (value as number) / 100)}
                  min={20}
                  max={80}
                  step={5}
                  marks={[
                    { value: 20, label: '20%' },
                    { value: 35, label: '35%' },
                    { value: 50, label: '50%' },
                    { value: 65, label: '65%' },
                    { value: 80, label: '80%' }
                  ]}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(value) => `${value}%`}
                />
                <Typography variant="caption" color="text.secondary">
                  Percentage of model's context window to use
                </Typography>
              </Box>
            </Grid>

            <Grid item xs={12} md={4}>
              <Box>
                <Typography gutterBottom>
                  Confidence Threshold: {config.extraction.confidence_threshold.toFixed(2)}
                </Typography>
                <Slider
                  value={config.extraction.confidence_threshold}
                  onChange={(_, value) => handleFieldChange('extraction', 'confidence_threshold', value as number)}
                  min={0.5}
                  max={1.0}
                  step={0.05}
                  marks={[
                    { value: 0.5, label: '0.5' },
                    { value: 0.75, label: '0.75' },
                    { value: 1.0, label: '1.0' }
                  ]}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(value) => value.toFixed(2)}
                />
                <Typography variant="caption" color="text.secondary">
                  Higher = stricter validation
                </Typography>
              </Box>
            </Grid>

            <Grid item xs={12} md={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.extraction.enable_chunking}
                    onChange={(e) => handleFieldChange('extraction', 'enable_chunking', e.target.checked)}
                  />
                }
                label="Enable Chunking"
              />
              <Typography variant="caption" color="text.secondary" display="block">
                Split large documents into smaller chunks for processing
              </Typography>
            </Grid>

            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Chunk Size"
                type="number"
                value={config.extraction.chunk_size}
                onChange={(e) => handleFieldChange('extraction', 'chunk_size', parseInt(e.target.value))}
                inputProps={{ min: 500, max: 10000, step: 100 }}
                helperText="Characters per chunk"
                disabled={!config.extraction.enable_chunking}
              />
            </Grid>

            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Chunk Overlap"
                type="number"
                value={config.extraction.chunk_overlap}
                onChange={(e) => handleFieldChange('extraction', 'chunk_overlap', parseInt(e.target.value))}
                inputProps={{ min: 0, max: 500, step: 10 }}
                helperText="Characters overlap between chunks"
                disabled={!config.extraction.enable_chunking}
              />
            </Grid>

            <Grid item xs={12}>
              <Box sx={{ border: '1px solid', borderColor: 'primary.main', borderRadius: 1, p: 2, bgcolor: 'action.hover' }}>
                <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  <DescriptionIcon color="primary" />
                  Extraction System Prompt
                </Typography>
                <TextField
                  fullWidth
                  multiline
                  rows={10}
                  label=""
                  placeholder="Enter the system prompt for document extraction..."
                  value={config.extraction.system_prompt || ''}
                  onChange={(e) => handleFieldChange('extraction', 'system_prompt', e.target.value)}
                  helperText="This prompt defines how the LLM extracts and formats documents into structured markdown."
                  variant="outlined"
                  sx={{ 
                    '& .MuiInputBase-root': { 
                      minHeight: '250px',
                      alignItems: 'flex-start',
                      backgroundColor: 'background.paper'
                    },
                    '& textarea': {
                      fontSize: '14px',
                      fontFamily: 'monospace',
                      lineHeight: 1.5
                    }
                  }}
                />
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Validation Configuration */}
      <Card sx={{ mb: 3 }}>
        <CardHeader
          avatar={<GavelIcon />}
          title="Validation Configuration"
          subheader="Configure how extracted documents are validated against references"
        />
        <CardContent>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Validation Model</InputLabel>
                <Select
                  value={config.validation.model}
                  onChange={(e: SelectChangeEvent) => handleFieldChange('validation', 'model', e.target.value)}
                  label="Validation Model"
                  disabled={loadingModels}
                >
                  {availableModels.map((model) => (
                    <MenuItem key={model.id} value={model.name}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                        <Typography>{model.name}</Typography>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <Chip label={formatFileSize(model.size)} size="small" variant="outlined" />
                          <Chip label={model.context_length || 'N/A'} size="small" color="primary" variant="outlined" />
                        </Box>
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Model Server URL"
                value={config.validation.model_server || ''}
                onChange={(e) => handleFieldChange('validation', 'model_server', e.target.value)}
                placeholder="http://host.docker.internal:11434"
                helperText="Ollama server URL for validation model"
              />
            </Grid>
            
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                label="Temperature"
                type="number"
                value={config.validation.temperature}
                onChange={(e) => handleFieldChange('validation', 'temperature', parseFloat(e.target.value))}
                inputProps={{ min: 0, max: 2, step: 0.1 }}
                helperText="Lower = more consistent"
              />
            </Grid>
            
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                label="Max Tokens (Auto)"
                type="number"
                value={config.validation.max_tokens}
                InputProps={{
                  readOnly: true,
                }}
                helperText="Auto-calculated from context %"
                sx={{
                  '& .MuiInputBase-input': {
                    backgroundColor: 'action.hover',
                  }
                }}
              />
            </Grid>

            <Grid item xs={12} md={4}>
              <Box>
                <Typography gutterBottom>
                  Max Context Usage: {Math.round(config.validation.max_context_usage * 100)}%
                </Typography>
                <Slider
                  value={config.validation.max_context_usage * 100}
                  onChange={(_, value) => handleFieldChange('validation', 'max_context_usage', (value as number) / 100)}
                  min={20}
                  max={80}
                  step={5}
                  marks={[
                    { value: 20, label: '20%' },
                    { value: 35, label: '35%' },
                    { value: 50, label: '50%' },
                    { value: 65, label: '65%' },
                    { value: 80, label: '80%' }
                  ]}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(value) => `${value}%`}
                />
                <Typography variant="caption" color="text.secondary">
                  Percentage of model's context window to use
                </Typography>
              </Box>
            </Grid>

            <Grid item xs={12} md={4}>
              <Box>
                <Typography gutterBottom>
                  Confidence Threshold: {config.validation.confidence_threshold.toFixed(2)}
                </Typography>
                <Slider
                  value={config.validation.confidence_threshold}
                  onChange={(_, value) => handleFieldChange('validation', 'confidence_threshold', value as number)}
                  min={0.5}
                  max={1.0}
                  step={0.05}
                  marks={[
                    { value: 0.5, label: '0.5' },
                    { value: 0.75, label: '0.75' },
                    { value: 1.0, label: '1.0' }
                  ]}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(value) => value.toFixed(2)}
                />
                <Typography variant="caption" color="text.secondary">
                  Higher = stricter validation
                </Typography>
              </Box>
            </Grid>

            <Grid item xs={12} md={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.validation.enable_structured_output}
                    onChange={(e) => handleFieldChange('validation', 'enable_structured_output', e.target.checked)}
                  />
                }
                label="Enable Structured Output"
              />
              <Typography variant="caption" color="text.secondary" display="block">
                Force structured JSON/XML output format for validation results
              </Typography>
            </Grid>

            <Grid item xs={12}>
              <Box sx={{ border: '1px solid', borderColor: 'secondary.main', borderRadius: 1, p: 2, bgcolor: 'action.hover' }}>
                <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  <DescriptionIcon color="secondary" />
                  Validation System Prompt
                </Typography>
                <TextField
                  fullWidth
                  multiline
                  rows={10}
                  label=""
                  placeholder="Enter the system prompt for document validation..."
                  value={config.validation.system_prompt || ''}
                  onChange={(e) => handleFieldChange('validation', 'system_prompt', e.target.value)}
                  helperText="This prompt defines how the LLM compares and validates documents against reference documents."
                  variant="outlined"
                  sx={{ 
                    '& .MuiInputBase-root': { 
                      minHeight: '250px',
                      alignItems: 'flex-start',
                      backgroundColor: 'background.paper'
                    },
                    '& textarea': {
                      fontSize: '14px',
                      fontFamily: 'monospace',
                      lineHeight: 1.5
                    }
                  }}
                />
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Comparison Configuration */}
      <Card sx={{ mb: 3 }}>
        <CardHeader
          avatar={<CompareIcon />}
          title="Comparison Configuration"
          subheader="Configure document comparison algorithms and thresholds"
        />
        <CardContent>
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <FormControl fullWidth>
                <InputLabel>Comparison Algorithm</InputLabel>
                <Select
                  value={config.comparison.algorithm}
                  onChange={(e: SelectChangeEvent) => handleFieldChange('comparison', 'algorithm', e.target.value)}
                  label="Comparison Algorithm"
                >
                  <MenuItem value="semantic">Semantic</MenuItem>
                  <MenuItem value="structural">Structural</MenuItem>
                  <MenuItem value="hybrid">Hybrid</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Similarity Threshold"
                type="number"
                value={config.comparison.similarity_threshold}
                onChange={(e) => handleFieldChange('comparison', 'similarity_threshold', parseFloat(e.target.value))}
                inputProps={{ min: 0, max: 1, step: 0.05 }}
                helperText="Minimum similarity score"
              />
            </Grid>

            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Fuzzy Match Threshold"
                type="number"
                value={config.comparison.fuzzy_threshold}
                onChange={(e) => handleFieldChange('comparison', 'fuzzy_threshold', parseFloat(e.target.value))}
                inputProps={{ min: 0, max: 1, step: 0.05 }}
                helperText="Fuzzy matching threshold"
                disabled={!config.comparison.enable_fuzzy_matching}
              />
            </Grid>

            <Grid item xs={12} md={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.comparison.enable_fuzzy_matching}
                    onChange={(e) => handleFieldChange('comparison', 'enable_fuzzy_matching', e.target.checked)}
                  />
                }
                label="Enable Fuzzy Matching"
              />
              <Typography variant="caption" color="text.secondary" display="block">
                Use fuzzy string matching for inexact comparisons
              </Typography>
            </Grid>

            <Grid item xs={12} md={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.comparison.ignore_case}
                    onChange={(e) => handleFieldChange('comparison', 'ignore_case', e.target.checked)}
                  />
                }
                label="Ignore Case"
              />
              <Typography variant="caption" color="text.secondary" display="block">
                Case-insensitive text comparison
              </Typography>
            </Grid>

            <Grid item xs={12} md={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.comparison.ignore_whitespace}
                    onChange={(e) => handleFieldChange('comparison', 'ignore_whitespace', e.target.checked)}
                  />
                }
                label="Ignore Whitespace"
              />
              <Typography variant="caption" color="text.secondary" display="block">
                Ignore whitespace differences in text comparison
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Info Alert */}
      <Alert severity="info" icon={<CheckCircleIcon />}>
        <Typography variant="body2">
          <strong>Configuration Tips:</strong>
        </Typography>
        <Typography variant="body2" sx={{ mt: 1 }}>
          • Use lower temperatures (0.1-0.3) for consistent extraction and validation results
        </Typography>
        <Typography variant="body2">
          • Max Tokens are auto-calculated: Context Length × Max Context Usage %
        </Typography>
        <Typography variant="body2">
          • Example: 262,144 context × 35% = 91,750 max tokens (up to 80% for large contexts)
        </Typography>
        <Typography variant="body2">
          • Chunking splits large documents: Enable chunking for better processing of long documents
        </Typography>
        <Typography variant="body2">
          • System prompts define the behavior - customize them for your specific use case
        </Typography>
        <Typography variant="body2">
          • Higher confidence thresholds mean more items flagged for human review
        </Typography>
      </Alert>
    </Box>
  );
};

export default IDCConfigurationPanel;