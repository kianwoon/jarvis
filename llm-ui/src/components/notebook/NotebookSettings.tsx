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
  Grid,
  Switch,
  FormControlLabel as MuiFormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider
} from '@mui/material';
import {
  Save as SaveIcon,
  CheckCircle as CheckCircleIcon,
  Cached as CacheIcon,
  ExpandMore as ExpandMoreIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import '../../styles/settings-theme.css';

// Source template interface
interface SourceTemplate {
  active: boolean;
  template: string;
  description: string;
  variables: string[];
}

// Notebook source templates interface
interface NotebookSourceTemplates {
  base_source_intro: SourceTemplate;
  mixed_sources_detail: SourceTemplate;
  memory_only_detail: SourceTemplate;
  document_only_detail: SourceTemplate;
  no_specific_detail: SourceTemplate;
  synthesis_instruction: SourceTemplate;
  memory_context_explanation: SourceTemplate;
  comprehensive_answer_instruction: SourceTemplate;
}

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
  const [sourceTemplates, setSourceTemplates] = useState<NotebookSourceTemplates | null>(null);
  const [templatesLoading, setTemplatesLoading] = useState(false);
  const [templatesSaving, setTemplatesSaving] = useState(false);
  
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load settings on mount
  useEffect(() => {
    loadSettings();
    fetchAvailableModels();
    loadSourceTemplates();
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

  // Load source templates
  const loadSourceTemplates = async () => {
    setTemplatesLoading(true);
    
    try {
      const response = await fetch('/api/v1/settings/notebook_source_templates');
      
      if (response.ok) {
        const data = await response.json();
        setSourceTemplates(data.settings || getDefaultSourceTemplates());
      } else if (response.status === 404) {
        // No templates found, use defaults
        setSourceTemplates(getDefaultSourceTemplates());
        enqueueSnackbar('Using default source templates - no custom templates found', { variant: 'info' });
      } else {
        throw new Error(`Failed to load source templates: ${response.status}`);
      }
    } catch (error) {
      console.error('Failed to load source templates:', error);
      setSourceTemplates(getDefaultSourceTemplates());
      enqueueSnackbar('Failed to load source templates, using defaults', { variant: 'warning' });
    } finally {
      setTemplatesLoading(false);
    }
  };

  // Get default source templates
  const getDefaultSourceTemplates = (): NotebookSourceTemplates => {
    return {
      base_source_intro: {
        active: true,
        template: "You have access to {total_sources} relevant information sources from this notebook",
        description: "Base template for introducing available sources",
        variables: ["total_sources"]
      },
      mixed_sources_detail: {
        active: true,
        template: ", including {document_count} document{document_plural} and {memory_count} personal memor{memory_plural}. ",
        description: "Template for when both documents and memories are present",
        variables: ["document_count", "document_plural", "memory_count", "memory_plural"]
      },
      memory_only_detail: {
        active: true,
        template: ", including {memory_count} personal memor{memory_plural}. ",
        description: "Template for memory-only sources",
        variables: ["memory_count", "memory_plural"]
      },
      document_only_detail: {
        active: true,
        template: ", including {document_count} document{document_plural}. ",
        description: "Template for document-only sources",
        variables: ["document_count", "document_plural"]
      },
      no_specific_detail: {
        active: true,
        template: ". ",
        description: "Template fallback for general sources",
        variables: []
      },
      synthesis_instruction: {
        active: true,
        template: "When responding, synthesize information from ALL provided sources - both documents and memories contain valuable context. ",
        description: "Instruction for synthesizing all source types",
        variables: []
      },
      memory_context_explanation: {
        active: true,
        template: "Memories typically contain personal experiences, recent developments, or contextual information that complements the formal document content. ",
        description: "Explanation of what memories contain",
        variables: []
      },
      comprehensive_answer_instruction: {
        active: true,
        template: "Provide comprehensive answers that integrate insights from all available sources. If the context does not contain enough information, say so clearly.",
        description: "Instruction for comprehensive responses",
        variables: []
      }
    };
  };

  // Save source templates
  const saveSourceTemplates = async () => {
    if (!sourceTemplates) {
      enqueueSnackbar('No source templates to save', { variant: 'warning' });
      return;
    }

    setTemplatesSaving(true);
    
    try {
      const response = await fetch('/api/v1/settings/notebook_source_templates', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          settings: sourceTemplates
        })
      });

      if (response.ok) {
        enqueueSnackbar('Source templates saved successfully', { variant: 'success' });
      } else {
        throw new Error(`Failed to save source templates: ${response.status}`);
      }
    } catch (error) {
      console.error('Failed to save source templates:', error);
      enqueueSnackbar('Failed to save source templates', { variant: 'error' });
    } finally {
      setTemplatesSaving(false);
    }
  };

  // Handle source template changes
  const handleSourceTemplateChange = (templateKey: keyof NotebookSourceTemplates, field: keyof SourceTemplate, value: any) => {
    if (!sourceTemplates) return;
    
    setSourceTemplates(prev => prev ? ({
      ...prev,
      [templateKey]: {
        ...prev[templateKey],
        [field]: value
      }
    }) : null);
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

      {/* Source Integration Templates Card */}
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardHeader 
          title="Source Integration Templates"
          subheader="Configure how the LLM integrates notebook sources (documents and memories)"
          avatar={<SettingsIcon color="primary" />}
          action={
            <Button
              variant="contained"
              color="secondary"
              onClick={saveSourceTemplates}
              disabled={templatesSaving || !sourceTemplates}
              startIcon={templatesSaving ? <CircularProgress size={16} /> : <SaveIcon />}
            >
              {templatesSaving ? 'Saving...' : 'Save Templates'}
            </Button>
          }
        />
        <CardContent>
          {templatesLoading ? (
            <Box display="flex" justifyContent="center" alignItems="center" p={4}>
              <CircularProgress size={24} />
              <Typography ml={2}>Loading source templates...</Typography>
            </Box>
          ) : sourceTemplates ? (
            <>
              <Alert severity="info" sx={{ mb: 3 }}>
                <Typography variant="body2">
                  These templates control how the LLM introduces and processes different types of sources from your notebook. 
                  Available variables: <strong>{'{total_sources}'}</strong>, <strong>{'{document_count}'}</strong>, 
                  <strong>{'{memory_count}'}</strong>, <strong>{'{document_plural}'}</strong>, <strong>{'{memory_plural}'}</strong>
                </Typography>
              </Alert>
              
              <Box sx={{ mb: 2 }}>
                {Object.entries(sourceTemplates).map(([key, template]) => {
                  const templateKey = key as keyof NotebookSourceTemplates;
                  const friendlyName = key.split('_').map(word => 
                    word.charAt(0).toUpperCase() + word.slice(1)
                  ).join(' ');
                  
                  return (
                    <Accordion key={key} sx={{ mb: 1 }}>
                      <AccordionSummary 
                        expandIcon={<ExpandMoreIcon />}
                        sx={{ 
                          backgroundColor: 'action.hover',
                          '&:hover': { backgroundColor: 'action.selected' },
                          borderRadius: 1
                        }}
                      >
                        <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                          <Box sx={{ flex: 1 }}>
                            <Typography variant="h6" sx={{ fontSize: '14px', fontWeight: 600 }}>
                              {friendlyName}
                            </Typography>
                            <Typography variant="body2" color="text.secondary" sx={{ fontSize: '12px' }}>
                              {template.description}
                            </Typography>
                          </Box>
                          <MuiFormControlLabel
                            control={
                              <Switch
                                checked={template.active}
                                onChange={(e) => handleSourceTemplateChange(templateKey, 'active', e.target.checked)}
                                onClick={(e) => e.stopPropagation()}
                                size="small"
                              />
                            }
                            label={template.active ? 'Active' : 'Inactive'}
                            sx={{ mr: 2 }}
                            onClick={(e) => e.stopPropagation()}
                          />
                        </Box>
                      </AccordionSummary>
                      <AccordionDetails sx={{ backgroundColor: 'background.paper' }}>
                        <Box sx={{ mb: 2 }}>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                            Variables: {template.variables.length > 0 ? 
                              template.variables.map(v => `{${v}}`).join(', ') : 
                              'None'
                            }
                          </Typography>
                          <Divider sx={{ my: 2 }} />
                        </Box>
                        
                        <TextField
                          label="Template Content"
                          value={template.template}
                          onChange={(e) => handleSourceTemplateChange(templateKey, 'template', e.target.value)}
                          multiline
                          minRows={2}
                          maxRows={8}
                          fullWidth
                          variant="outlined"
                          disabled={!template.active}
                          helperText={!template.active ? 'Template is inactive' : 'Use variables like {total_sources}, {document_count}, {memory_count}'}
                          sx={{
                            '& .MuiInputBase-root': {
                              fontFamily: 'monospace',
                              fontSize: '13px',
                              lineHeight: 1.4,
                              backgroundColor: template.active ? 'background.paper' : 'action.disabledBackground'
                            }
                          }}
                        />
                        
                        <Box sx={{ mt: 2, p: 2, backgroundColor: 'action.hover', borderRadius: 1 }}>
                          <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600, display: 'block', mb: 1 }}>
                            PREVIEW:
                          </Typography>
                          <Typography 
                            variant="body2" 
                            sx={{ 
                              fontFamily: 'monospace', 
                              fontSize: '12px',
                              backgroundColor: 'background.paper',
                              p: 1.5,
                              borderRadius: 0.5,
                              border: '1px solid',
                              borderColor: 'divider',
                              whiteSpace: 'pre-wrap',
                              color: template.active ? 'text.primary' : 'text.disabled'
                            }}
                          >
                            {template.template.replace(/{(\w+)}/g, (match, variable) => {
                              // Show example values for preview
                              const examples: Record<string, string> = {
                                'total_sources': '5',
                                'document_count': '3',
                                'memory_count': '2',
                                'document_plural': 's',
                                'memory_plural': 'ies'
                              };
                              return examples[variable] || match;
                            })}
                          </Typography>
                        </Box>
                      </AccordionDetails>
                    </Accordion>
                  );
                })}
              </Box>
              
              <Alert severity="success" sx={{ mt: 2 }}>
                <Typography variant="body2">
                  These templates will be automatically applied when the notebook LLM processes sources. 
                  Active templates will be combined to create contextual prompts for different source scenarios.
                </Typography>
              </Alert>
            </>
          ) : (
            <Alert severity="warning">
              <Typography>Failed to load source templates. Please refresh the page.</Typography>
            </Alert>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default NotebookSettings;