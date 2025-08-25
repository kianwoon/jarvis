import React, { useState, useEffect } from 'react';
import {
  Box,
  TextField,
  Typography,
  Chip,
  Button,
  Card,
  CardContent,
  CardHeader,
  Select,
  MenuItem,
  InputLabel,
  FormControl,
  CircularProgress,
  Tooltip,
  Alert,
  SelectChangeEvent,
  Radio,
  RadioGroup,
  FormControlLabel,
  Paper,
  Grid
} from '@mui/material';
import {
  Save as SaveIcon,
  CheckCircle as CheckCircleIcon,
  Cached as CacheIcon
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import '../../styles/settings-theme.css';

// Notebook-specific LLM settings interface
interface NotebookLLMSettings {
  model: string;
  temperature: number;
  top_p: number;
  max_tokens: number;
  mode: 'thinking' | 'non-thinking';
  streaming_enabled: boolean;
  context_length: number;
  model_server: string;
  system_prompt: string;
  repeat_penalty: number;
}

// Model interface (matching existing pattern)
interface Model {
  name: string;
  id: string;
  size: string;
  modified: string;
  context_length: string;
}

interface NotebookSettingsProps {
  notebookId: string;
  onSettingsChange?: (settings: NotebookLLMSettings) => void;
}

const NotebookSettings: React.FC<NotebookSettingsProps> = ({ 
  notebookId,
  onSettingsChange 
}) => {
  const { enqueueSnackbar } = useSnackbar();
  
  // State management - Initialize empty, load from database
  const [settings, setSettings] = useState<NotebookLLMSettings | null>(null);
  
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load settings on mount
  useEffect(() => {
    loadSettings();
    fetchAvailableModels();
  }, [notebookId]);

  // Load notebook-specific settings
  const loadSettings = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`/api/v1/settings/notebook_llm`);
      
      if (response.ok) {
        const data = await response.json();
        // Extract notebook_llm settings from the API response
        if (data.notebook_llm) {
          setSettings(data.notebook_llm);
          onSettingsChange?.(data.notebook_llm);
        } else {
          throw new Error('No notebook_llm settings found in response');
        }
      } else if (response.status === 404) {
        // No settings found, create defaults
        const defaultSettings = {
          model: 'llama3.1:8b',
          temperature: 0.7,
          top_p: 0.9,
          max_tokens: 4096,
          mode: 'thinking' as const,
          streaming_enabled: true,
          context_length: 128000,
          model_server: 'http://localhost:11434',
          system_prompt: 'You are a helpful assistant for Jupyter notebook interactions. Provide clear, concise responses that are suitable for notebook environments.',
          repeat_penalty: 1.1
        };
        setSettings(defaultSettings);
        enqueueSnackbar('Using default settings - no notebook LLM config found', { variant: 'info' });
      } else {
        throw new Error(`Failed to load settings: ${response.status}`);
      }
    } catch (error) {
      console.error('Failed to load notebook settings:', error);
      setError(error instanceof Error ? error.message : 'Failed to load settings');
      enqueueSnackbar('Failed to load notebook settings', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  // Fetch available models (matching existing pattern)
  const fetchAvailableModels = async () => {
    try {
      const response = await fetch('/api/v1/ollama/models');
      if (response.ok) {
        const data = await response.json();
        
        if (data.success) {
          setModels(data.models || []);
        } else {
          // Use fallback models if Ollama not available
          if (data.fallback_models && data.fallback_models.length > 0) {
            setModels(data.fallback_models);
          } else {
            setModels([
              { name: 'llama3.1:8b', id: 'fallback-01', size: '4.7 GB', modified: 'N/A', context_length: '128,000' },
              { name: 'deepseek-r1:8b', id: 'fallback-04', size: '4.9 GB', modified: 'N/A', context_length: '65,536' }
            ]);
          }
        }
      }
    } catch (error) {
      console.error('Failed to fetch models:', error);
      setModels([
        { name: 'llama3.1:8b', id: 'fallback-01', size: '4.7 GB', modified: 'N/A', context_length: '128,000' },
        { name: 'deepseek-r1:8b', id: 'fallback-04', size: '4.9 GB', modified: 'N/A', context_length: '65,536' }
      ]);
    }
  };

  // Save settings
  const saveSettings = async () => {
    if (!settings) {
      enqueueSnackbar('No settings to save', { variant: 'warning' });
      return;
    }

    setSaving(true);
    setError(null);
    
    try {
      const response = await fetch('/api/v1/settings/notebook_llm', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          notebook_llm: settings
        })
      });

      if (response.ok) {
        enqueueSnackbar('Notebook settings saved successfully', { variant: 'success' });
        onSettingsChange?.(settings);
      } else {
        throw new Error(`Failed to save settings: ${response.status}`);
      }
    } catch (error) {
      console.error('Failed to save notebook settings:', error);
      setError(error instanceof Error ? error.message : 'Failed to save settings');
      enqueueSnackbar('Failed to save notebook settings', { variant: 'error' });
    } finally {
      setSaving(false);
    }
  };

  // Handle field changes
  const handleFieldChange = (field: keyof NotebookLLMSettings, value: any) => {
    if (!settings) return;
    
    setSettings(prev => prev ? ({
      ...prev,
      [field]: value
    }) : null);
    setError(null);
  };

  // Handle model change with auto-context update (matching existing pattern)
  const handleModelChange = (event: SelectChangeEvent<string>) => {
    const newModelName = event.target.value;
    const selectedModel = models.find(m => m.name === newModelName);
    
    handleFieldChange('model', newModelName);
    
    // Auto-update context window size based on model (matching existing pattern)
    if (selectedModel && selectedModel.context_length !== 'Unknown') {
      const contextLength = parseInt(selectedModel.context_length.replace(/,/g, ''));
      if (!isNaN(contextLength)) {
        handleFieldChange('context_length', contextLength);
        // Auto-suggest max_tokens as 75% of context window
        const suggestedMaxTokens = Math.floor(contextLength * 0.75);
        handleFieldChange('max_tokens', suggestedMaxTokens);
      }
    }
  };

  const selectedModel = models.find(m => m.name === settings?.model);

  if (loading || !settings) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" p={4}>
        <CircularProgress />
        <Typography ml={2}>Loading notebook settings...</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', p: 2 }}>
      {/* Header */}
      <Card variant="outlined" sx={{ mb: 3 }}>
        <CardHeader
          title="Notebook LLM Settings"
          subheader="Configure language model settings specific to this notebook"
          action={
            <Button
              variant="contained"
              color="primary"
              onClick={saveSettings}
              disabled={saving}
              startIcon={saving ? <CircularProgress size={16} /> : <SaveIcon />}
            >
              {saving ? 'Saving...' : 'Save Settings'}
            </Button>
          }
        />
      </Card>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* LLM Mode Selection */}
      <Card variant="outlined" sx={{ mb: 3 }}>
        <CardHeader 
          title="LLM Mode Selection"
          subheader="Select between thinking and non-thinking modes"
        />
        <CardContent>
          <FormControl component="fieldset">
            <RadioGroup
              value={settings?.mode || 'thinking'}
              onChange={(e) => handleFieldChange('mode', e.target.value as 'thinking' | 'non-thinking')}
            >
              <FormControlLabel 
                value="thinking" 
                control={<Radio />} 
                label={
                  <Box>
                    <Typography variant="body2" fontWeight={600}>
                      Thinking Mode
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Enable step-by-step reasoning with &lt;think&gt; tags
                    </Typography>
                  </Box>
                }
              />
              <FormControlLabel 
                value="non-thinking" 
                control={<Radio />} 
                label={
                  <Box>
                    <Typography variant="body2" fontWeight={600}>
                      Non-Thinking Mode
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Direct responses without explicit reasoning steps
                    </Typography>
                  </Box>
                }
              />
            </RadioGroup>
          </FormControl>
        </CardContent>
      </Card>

      {/* LLM Model Configuration */}
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardHeader 
          title="LLM Model Configuration"
          subheader="Select and configure your language model"
          action={
            <Tooltip title="Update available models and reload LLM cache with current settings">
              <Button 
                size="small" 
                onClick={async (event) => {
                  event.preventDefault();
                  event.stopPropagation();
                  
                  try {
                    if (!settings) {
                      enqueueSnackbar('No settings loaded yet', { variant: 'warning' });
                      return;
                    }

                    // Step 1: Save current settings
                    const saveResponse = await fetch('/api/v1/settings/notebook_llm', {
                      method: 'PUT',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({
                        notebook_llm: settings
                      })
                    });
                    
                    if (!saveResponse.ok) {
                      throw new Error(`Save failed with status: ${saveResponse.status}`);
                    }
                    
                    // Step 2: Fetch available models
                    await fetchAvailableModels();
                    
                    // Step 3: Show success message
                    enqueueSnackbar('Settings saved, models updated successfully!', { variant: 'success' });
                  } catch (error) {
                    console.error('Error in update process:', error);
                    enqueueSnackbar(`Update failed: ${error instanceof Error ? error.message : 'Unknown error'}`, { variant: 'error' });
                  }
                }}
                startIcon={<CacheIcon />}
                variant="outlined"
                color="primary"
              >
                UPDATE MODELS & CACHE
              </Button>
            </Tooltip>
          }
        />
        <CardContent>
          {/* Model Selector */}
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel id="model-select-label">Select Model</InputLabel>
            <Select
              labelId="model-select-label"
              value={settings?.model || ''}
              label="Select Model"
              onChange={handleModelChange}
            >
              {models.map((model, index) => (
                <MenuItem key={`${model.id}-${model.name}-${index}`} value={model.name}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography>{model.name}</Typography>
                      {settings?.model === model.name && (
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
              ))}
            </Select>
          </FormControl>

          {/* Current Model Information Panel */}
          {selectedModel && (
            <Paper sx={{ p: 3, backgroundColor: 'action.hover', border: '1px solid', borderColor: 'primary.light' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <CheckCircleIcon color="success" />
                <Typography variant="h6" color="primary">
                  Current Model: {selectedModel.name}
                </Typography>
              </Box>
              
              <Grid container spacing={3}>
                <Grid item xs={12} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2, backgroundColor: 'background.paper', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                      Model Size
                    </Typography>
                    <Typography variant="h6" color="primary">
                      {selectedModel.size}
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2, backgroundColor: 'background.paper', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                      Context Length
                    </Typography>
                    <Typography variant="h6" color="primary">
                      {selectedModel.context_length}
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2, backgroundColor: 'background.paper', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                      Last Modified
                    </Typography>
                    <Typography variant="body2" color="text.primary">
                      {selectedModel.modified}
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2, backgroundColor: 'background.paper', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                      Model ID
                    </Typography>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {selectedModel.id}
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
              
              {selectedModel.context_length !== 'Unknown' && (
                <Alert severity="info" sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    This model supports up to <strong>{selectedModel.context_length}</strong> tokens in context. 
                    Larger contexts allow for more detailed conversations but may increase processing time and costs.
                  </Typography>
                </Alert>
              )}
            </Paper>
          )}
          
          {!selectedModel && settings?.model && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              Model "{settings?.model}" not found in available models. Please refresh the model list or select a different model.
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Model Parameters Card */}
      <Card variant="outlined" sx={{ mb: 3 }}>
        <CardHeader 
          title="Model Parameters"
          subheader="Configure model behavior and performance settings"
        />
        <CardContent>
          <div className="jarvis-form-grid">
            <Tooltip title="Maximum number of tokens the model can generate in a single response">
              <TextField
                label="Max Tokens"
                type="number"
                variant="outlined"
                value={settings?.max_tokens || 0}
                onChange={(e) => handleFieldChange('max_tokens', parseInt(e.target.value))}
                inputProps={{ min: 1, max: settings?.context_length || 200000 }}
                InputLabelProps={{ shrink: true }}
                fullWidth
                helperText={`1 to ${settings?.context_length || 200000} tokens`}
              />
            </Tooltip>
            
            <TextField
              label="Model Server"
              variant="outlined"
              value={settings?.model_server || ''}
              onChange={(e) => handleFieldChange('model_server', e.target.value)}
              InputLabelProps={{ shrink: true }}
              fullWidth
              helperText="URL of the model server (e.g., http://localhost:11434)"
            />
            
            <div className="jarvis-form-group full-width">
              <TextField
                label="System Prompt"
                variant="outlined"
                value={settings?.system_prompt || ''}
                onChange={(e) => handleFieldChange('system_prompt', e.target.value)}
                InputLabelProps={{ shrink: true }}
                multiline
                minRows={3}
                maxRows={20}
                fullWidth
                helperText="Instructions for the model's behavior and role"
                sx={{
                  '& .MuiInputBase-root': {
                    resize: 'vertical',
                    minHeight: '80px'
                  },
                  '& .MuiInputBase-input': {
                    resize: 'vertical'
                  }
                }}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Performance Parameters Card */}
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardHeader 
          title="Performance Parameters"
          subheader="Model behavior and response quality settings"
        />
        <CardContent>
          <div className="jarvis-form-grid">
            <Tooltip title="Maximum context window size for the model. Auto-updated when model changes">
              <TextField
                label="Context Length"
                type="number"
                variant="outlined"
                value={settings?.context_length || 0}
                onChange={(e) => handleFieldChange('context_length', parseInt(e.target.value))}
                inputProps={{ min: 1024, max: 200000 }}
                InputLabelProps={{ shrink: true }}
                fullWidth
                helperText="Model's context window in tokens"
              />
            </Tooltip>
            
            <Tooltip title="Penalty for repeating tokens. Values > 1 discourage repetition">
              <TextField
                label="Repeat Penalty"
                type="number"
                variant="outlined"
                value={settings?.repeat_penalty || 0}
                onChange={(e) => handleFieldChange('repeat_penalty', parseFloat(e.target.value))}
                inputProps={{ min: 0.1, max: 2.0, step: 0.1 }}
                InputLabelProps={{ shrink: true }}
                fullWidth
                helperText="0.1 to 2.0 (1.0 = no penalty)"
              />
            </Tooltip>
            
            <Tooltip title="Controls randomness in model outputs. Lower values (0.1-0.3) are more focused, higher values (0.7-1.0) are more creative">
              <TextField
                label="Temperature"
                type="number"
                variant="outlined"
                value={settings?.temperature || 0}
                onChange={(e) => handleFieldChange('temperature', parseFloat(e.target.value))}
                inputProps={{ min: 0, max: 1, step: 0.1 }}
                InputLabelProps={{ shrink: true }}
                fullWidth
                helperText="0.0 (deterministic) to 1.0 (creative)"
              />
            </Tooltip>
            
            <Tooltip title="Nucleus sampling parameter. Controls the cumulative probability cutoff for token selection">
              <TextField
                label="Top P"
                type="number"
                variant="outlined"
                value={settings?.top_p || 0}
                onChange={(e) => handleFieldChange('top_p', parseFloat(e.target.value))}
                inputProps={{ min: 0, max: 1, step: 0.1 }}
                InputLabelProps={{ shrink: true }}
                fullWidth
                helperText="0.0 to 1.0 (nucleus sampling)"
              />
            </Tooltip>
          </div>
        </CardContent>
      </Card>
    </Box>
  );
};

export default NotebookSettings;