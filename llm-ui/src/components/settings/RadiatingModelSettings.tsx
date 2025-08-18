import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
  FormControl,
  FormControlLabel,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Paper,
  Grid,
  Button,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  Chip,
  IconButton,
  Tooltip,
  RadioGroup,
  Radio
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Info as InfoIcon,
  Computer as ComputerIcon,
  Memory as MemoryIcon,
  AccessTime as AccessTimeIcon
} from '@mui/icons-material';

interface ModelInfo {
  name: string;
  id: string;
  size: string;
  modified: string;
  context_length: string;
}

interface RadiatingModelSettingsProps {
  modelConfig: any;
  onChange: (config: any) => void;
  onShowSuccess?: (message?: string) => void;
}

const RadiatingModelSettings: React.FC<RadiatingModelSettingsProps> = ({
  modelConfig,
  onChange,
  onShowSuccess
}) => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState(modelConfig?.model || 'llama3.1:8b');
  const [maxTokens, setMaxTokens] = useState(modelConfig?.max_tokens || 4096);
  const [temperature, setTemperature] = useState(modelConfig?.temperature || 0.7);
  const [systemPrompt, setSystemPrompt] = useState(modelConfig?.system_prompt || '');
  const [modelServer, setModelServer] = useState(modelConfig?.model_server || 'http://localhost:11434');
  const [contextLength, setContextLength] = useState(modelConfig?.context_length || 128000);
  const [llmMode, setLlmMode] = useState(modelConfig?.llm_mode || 'non-thinking');

  // Fetch available models on mount
  useEffect(() => {
    fetchAvailableModels();
  }, []);

  const fetchAvailableModels = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/ollama/models');
      if (response.ok) {
        const data = await response.json();
        
        if (data.success && data.models) {
          setModels(data.models);
        } else if (data.fallback_models) {
          setModels(data.fallback_models);
        } else {
          // Fallback models if API fails
          setModels([
            { name: 'llama3.1:8b', id: 'llama3.1:8b', size: '4.7 GB', modified: 'N/A', context_length: '128000' },
            { name: 'deepseek-r1:8b', id: 'deepseek-r1:8b', size: '4.9 GB', modified: 'N/A', context_length: '65536' },
            { name: 'qwen2.5:7b', id: 'qwen2.5:7b', size: '4.4 GB', modified: 'N/A', context_length: '32768' }
          ]);
        }
      }
    } catch (error) {
      console.error('Failed to fetch models:', error);
      // Use fallback models
      setModels([
        { name: 'llama3.1:8b', id: 'llama3.1:8b', size: '4.7 GB', modified: 'N/A', context_length: '128000' },
        { name: 'deepseek-r1:8b', id: 'deepseek-r1:8b', size: '4.9 GB', modified: 'N/A', context_length: '65536' },
        { name: 'qwen2.5:7b', id: 'qwen2.5:7b', size: '4.4 GB', modified: 'N/A', context_length: '32768' }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleModelChange = (value: string) => {
    setSelectedModel(value);
    
    // Find the selected model info
    const modelInfo = models.find(m => m.name === value || m.id === value);
    if (modelInfo) {
      // Update context length based on model
      const ctxLength = parseInt(modelInfo.context_length.replace(/,/g, '')) || 128000;
      setContextLength(ctxLength);
      
      // Suggest max tokens as 75% of context length
      const suggestedMaxTokens = Math.floor(ctxLength * 0.75);
      setMaxTokens(Math.min(suggestedMaxTokens, ctxLength));
    }
    
    updateConfig({
      model: value,
      context_length: contextLength,
      max_tokens: maxTokens
    });
  };

  const handleMaxTokensChange = (event: Event, value: number | number[]) => {
    const tokens = value as number;
    setMaxTokens(tokens);
    updateConfig({ max_tokens: tokens });
  };

  const handleTemperatureChange = (event: Event, value: number | number[]) => {
    const temp = value as number;
    setTemperature(temp);
    updateConfig({ temperature: temp });
  };

  const handleSystemPromptChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    const prompt = event.target.value;
    setSystemPrompt(prompt);
    updateConfig({ system_prompt: prompt });
  };

  const handleModelServerChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const server = event.target.value;
    setModelServer(server);
    updateConfig({ model_server: server });
  };

  const updateConfig = (updates: any) => {
    const newConfig = {
      model: selectedModel,
      max_tokens: maxTokens,
      temperature: temperature,
      system_prompt: systemPrompt,
      model_server: modelServer,
      context_length: contextLength,
      llm_mode: llmMode,
      ...updates
    };
    onChange(newConfig);
  };

  const handleLlmModeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const mode = event.target.value;
    setLlmMode(mode);
    updateConfig({ llm_mode: mode });
  };

  const getSelectedModelInfo = () => {
    return models.find(m => m.name === selectedModel || m.id === selectedModel);
  };

  const modelInfo = getSelectedModelInfo();

  // Default system prompt for radiating coverage
  const defaultSystemPrompt = `You are an intelligent entity and relationship extraction system for the Radiating Coverage feature.

Your role is to:
1. Extract relevant entities from the provided text
2. Identify relationships between entities
3. Discover implicit connections and patterns
4. Provide confidence scores for extracted information
5. Adapt to any domain without predefined entity types

Guidelines:
- Focus on extracting meaningful entities that contribute to understanding
- Identify both explicit and implicit relationships
- Provide confidence scores between 0 and 1
- Output structured JSON for easy parsing
- Be domain-agnostic and adaptive`;

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Model Configuration
      </Typography>
      
      <Grid container spacing={3}>
        {/* LLM Mode Selection */}
        <Grid item xs={12}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle1" gutterBottom fontWeight="medium">
                LLM Mode Selection
              </Typography>
              <FormControl component="fieldset">
                <RadioGroup
                  value={llmMode}
                  onChange={handleLlmModeChange}
                  row
                >
                  <FormControlLabel
                    value="thinking"
                    control={<Radio />}
                    label={
                      <Box>
                        <Typography variant="body2" fontWeight="medium">
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
                        <Typography variant="body2" fontWeight="medium">
                          Non-Thinking Mode
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Direct responses without explicit reasoning steps
                        </Typography>
                      </Box>
                    }
                    sx={{ ml: 4 }}
                  />
                </RadioGroup>
              </FormControl>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                Non-thinking mode is recommended for faster entity extraction and relationship discovery
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Model Selection */}
        <Grid item xs={12}>
          <FormControl fullWidth>
            <InputLabel>Model</InputLabel>
            <Select
              value={selectedModel}
              onChange={(e) => handleModelChange(e.target.value)}
              disabled={loading}
              endAdornment={
                loading ? <CircularProgress size={20} sx={{ mr: 2 }} /> : (
                  <IconButton onClick={fetchAvailableModels} size="small" sx={{ mr: 1 }}>
                    <RefreshIcon />
                  </IconButton>
                )
              }
            >
              {models.map((model) => (
                <MenuItem key={model.id} value={model.name}>
                  <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                    <Typography>{model.name}</Typography>
                    <Box sx={{ ml: 'auto', display: 'flex', gap: 1 }}>
                      <Chip label={model.size} size="small" />
                      <Chip label={`${model.context_length} tokens`} size="small" />
                    </Box>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        {/* Model Info Card */}
        {modelInfo && (
          <Grid item xs={12}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle2" gutterBottom color="text.secondary">
                  Model Information
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={4}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <MemoryIcon fontSize="small" color="action" />
                      <Typography variant="body2">
                        Size: <strong>{modelInfo.size}</strong>
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <ComputerIcon fontSize="small" color="action" />
                      <Typography variant="body2">
                        Context: <strong>{modelInfo.context_length} tokens</strong>
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <AccessTimeIcon fontSize="small" color="action" />
                      <Typography variant="body2">
                        Modified: <strong>{modelInfo.modified}</strong>
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Model Server URL */}
        <Grid item xs={12}>
          <TextField
            fullWidth
            label="Model Server URL"
            value={modelServer}
            onChange={handleModelServerChange}
            helperText="Ollama server endpoint for model inference"
          />
        </Grid>

        {/* Max Tokens Slider */}
        <Grid item xs={12}>
          <Box>
            <Typography gutterBottom>
              Max Tokens: {maxTokens.toLocaleString()}
            </Typography>
            <Slider
              value={maxTokens}
              onChange={handleMaxTokensChange}
              min={100}
              max={contextLength}
              step={100}
              marks={[
                { value: 100, label: '100' },
                { value: Math.floor(contextLength / 2), label: `${Math.floor(contextLength / 2).toLocaleString()}` },
                { value: contextLength, label: `${contextLength.toLocaleString()}` }
              ]}
            />
            <Typography variant="caption" color="text.secondary">
              Maximum number of tokens the model can generate for entity extraction and relationship discovery
            </Typography>
          </Box>
        </Grid>

        {/* Temperature Slider */}
        <Grid item xs={12}>
          <Box>
            <Typography gutterBottom>
              Temperature: {temperature.toFixed(2)}
            </Typography>
            <Slider
              value={temperature}
              onChange={handleTemperatureChange}
              min={0}
              max={2}
              step={0.01}
              marks={[
                { value: 0, label: '0 (Deterministic)' },
                { value: 0.7, label: '0.7 (Balanced)' },
                { value: 1.4, label: '1.4 (Creative)' },
                { value: 2, label: '2 (Random)' }
              ]}
            />
            <Typography variant="caption" color="text.secondary">
              Controls randomness in extraction. Lower values = more consistent, higher values = more diverse discoveries
            </Typography>
          </Box>
        </Grid>

        {/* System Prompt */}
        <Grid item xs={12}>
          <Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <Typography>System Prompt</Typography>
              <Tooltip title="Defines how the model should extract entities and relationships">
                <InfoIcon fontSize="small" color="action" />
              </Tooltip>
              {!systemPrompt && (
                <Button
                  size="small"
                  onClick={() => {
                    setSystemPrompt(defaultSystemPrompt);
                    updateConfig({ system_prompt: defaultSystemPrompt });
                  }}
                >
                  Use Default
                </Button>
              )}
            </Box>
            <TextField
              fullWidth
              multiline
              rows={8}
              value={systemPrompt}
              onChange={handleSystemPromptChange}
              placeholder={defaultSystemPrompt}
              helperText="System prompt that guides entity extraction and relationship discovery behavior"
            />
          </Box>
        </Grid>

        {/* Advanced Settings Info */}
        <Grid item xs={12}>
          <Alert severity="info">
            <Typography variant="body2">
              These model settings will be used by all Radiating Coverage components including:
              <ul style={{ margin: '8px 0' }}>
                <li>Entity extraction from documents and queries</li>
                <li>Relationship discovery between entities</li>
                <li>Query expansion and semantic analysis</li>
                <li>Pattern detection and inference</li>
              </ul>
            </Typography>
          </Alert>
        </Grid>
      </Grid>
    </Box>
  );
};

export default RadiatingModelSettings;