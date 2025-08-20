import React, { useState, useEffect } from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Paper,
  Container,
  ThemeProvider,
  createTheme,
  CssBaseline,
  Tabs,
  Tab,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  LinearProgress,
  Alert,
  Divider,
  Switch,
  FormControlLabel,
  CircularProgress,
  Slider,
  Stack
} from '@mui/material';
import {
  LightMode as LightModeIcon,
  DarkMode as DarkModeIcon,
  Add as AddIcon,
  PlayArrow as PlayIcon,
  Settings as SettingsIcon,
  Description as DocumentIcon,
  AccountTree as WorkflowIcon,
  Timeline as TimelineIcon,
  Analytics as AnalyticsIcon,
  RateReview as ReviewIcon,
  Build as BuildIcon,
  AutoAwesome as GeneratorIcon,
  Refresh as RefreshIcon,
  Save as SaveIcon,
  CheckCircle as CheckCircleIcon,
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  Schedule as ScheduleIcon,
  Code as CodeIcon,
  ModelTraining as ModelIcon,
  Tune as TuneIcon,
  SettingsApplications as AdvancedIcon,
  EditNote as PromptIcon
} from '@mui/icons-material';

interface Template {
  id: string;
  name: string;
  description: string;
  template_type: string;
  template_config: any;
  is_active: boolean;
}

interface Workflow {
  id: string;
  name: string;
  status: string;
  progress: any;
  created_at: string;
}

interface ModelInfo {
  name: string;
  size: string;
  context_length: number;
  modified_at: string;
  model_id: string;
  details?: {
    format: string;
    family: string;
    families?: string[];
    parameter_size: string;
    quantization_level: string;
  };
}

interface MetaTaskModel {
  model: string;
  max_tokens: number;
  temperature: number;
  top_p: number;
  top_k: number;
  min_p: number;
  repeat_penalty: number;
  system_prompt?: string;
}

interface MetaTaskSettings {
  // General settings
  output_format: string;
  max_output_size_mb: number;
  cache_enabled: boolean;
  cache_ttl_hours: number;
  cache_workflows: boolean;
  
  // Execution settings
  execution_retry_attempts: number;
  execution_timeout_minutes: number;
  parallel_execution: boolean;
  
  // Quality control
  quality_control_enabled: boolean;
  quality_min_score: number;
  quality_auto_retry: boolean;
  
  // Model configurations
  analyzer_model: MetaTaskModel;
  reviewer_model: MetaTaskModel;
  assembler_model: MetaTaskModel;
  generator_model: MetaTaskModel;
}

interface SettingsCategory {
  id: string;
  name: string;
  icon: React.ReactNode;
  description: string;
}

function MetaTaskApp() {
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('jarvis-dark-mode');
    return saved ? JSON.parse(saved) : false;
  });

  const [templates, setTemplates] = useState<Template[]>([]);
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');
  const [workflowName, setWorkflowName] = useState<string>('');
  const [inputData, setInputData] = useState<any>({});
  const [createWorkflowOpen, setCreateWorkflowOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState<string>('');
  
  // Settings management
  const [selectedCategory, setSelectedCategory] = useState('meta_task');
  const [activeModelTab, setActiveModelTab] = useState(0);
  const [metaTaskSettings, setMetaTaskSettings] = useState<MetaTaskSettings | null>(null);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  
  // Track active submenu for each model tab
  const [activeSubmenus, setActiveSubmenus] = useState<Record<string, number>>({
    analyzer_model: 0,
    reviewer_model: 0,
    assembler_model: 0,
    generator_model: 0
  });

  // Settings categories for sidebar
  const settingsCategories: SettingsCategory[] = [
    {
      id: 'meta_task',
      name: 'Meta-Task System',
      icon: <WorkflowIcon />,
      description: 'Configure meta-task models and execution settings'
    },
    {
      id: 'templates',
      name: 'Templates & Workflows',
      icon: <DocumentIcon />,
      description: 'Manage templates and active workflows'
    }
  ];

  // Model tabs configuration - Settings first, then models
  const modelTabs = [
    {
      id: 'settings',
      label: 'Settings',
      icon: <SettingsIcon />,
      description: 'General execution, caching, and quality control settings'
    },
    { 
      id: 'analyzer_model', 
      label: 'Analyzer Model', 
      icon: <AnalyticsIcon />,
      description: 'Analyzes requirements and plans document structure'
    },
    { 
      id: 'reviewer_model', 
      label: 'Reviewer Model', 
      icon: <ReviewIcon />,
      description: 'Reviews and validates generated content for quality'
    },
    { 
      id: 'assembler_model', 
      label: 'Assembler Model', 
      icon: <BuildIcon />,
      description: 'Assembles components into cohesive documents'
    },
    { 
      id: 'generator_model', 
      label: 'Generator Model', 
      icon: <GeneratorIcon />,
      description: 'Generates detailed content for each section'
    }
  ];

  // Create theme
  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#2196f3',
      },
      secondary: {
        main: '#ff9800',
      },
      background: {
        default: darkMode ? '#121212' : '#f5f5f5',
        paper: darkMode ? '#1e1e1e' : '#ffffff',
      },
    },
  });

  const toggleDarkMode = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem('jarvis-dark-mode', JSON.stringify(newDarkMode));
  };

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    switch (newValue) {
      case 0:
        window.location.href = '/';
        break;
      case 1:
        window.location.href = '/multi-agent.html';
        break;
      case 2:
        window.location.href = '/workflow.html';
        break;
      case 3:
        // Already on meta-tasks page
        break;
      case 4:
        window.location.href = '/settings.html';
        break;
      case 5:
        window.location.href = '/knowledge-graph.html';
        break;
      case 6:
        window.location.href = '/idc.html';
        break;
    }
  };

  useEffect(() => {
    loadTemplates();
    loadWorkflows();
    if (selectedCategory === 'meta_task') {
      loadMetaTaskSettings();
      loadAvailableModels();
    }
  }, [selectedCategory]);

  // Load available models from Ollama
  const loadAvailableModels = async () => {
    try {
      setLoadingModels(true);
      const response = await fetch('http://localhost:11434/api/tags');
      if (response.ok) {
        const data = await response.json();
        const models: ModelInfo[] = data.models?.map((model: any) => ({
          name: model.name,
          size: formatBytes(model.size),
          context_length: model.details?.context_length || 8192,
          modified_at: new Date(model.modified_at).toLocaleDateString(),
          model_id: model.digest || model.name,
          details: {
            format: model.details?.format || 'GGUF',
            family: model.details?.family || 'Unknown',
            families: model.details?.families || [],
            parameter_size: model.details?.parameter_size || 'Unknown',
            quantization_level: model.details?.quantization_level || 'Q4_0'
          }
        })) || [];
        setAvailableModels(models);
      }
    } catch (err) {
      console.error('Failed to load models:', err);
      // Fallback models if Ollama is not available
      setAvailableModels([
        {
          name: 'qwen3:30b-a3b',
          size: '30B',
          context_length: 32768,
          modified_at: new Date().toLocaleDateString(),
          model_id: 'qwen3:30b-a3b'
        },
        {
          name: 'llama3.1:latest',
          size: '8B',
          context_length: 8192,
          modified_at: new Date().toLocaleDateString(),
          model_id: 'llama3.1:latest'
        },
        {
          name: 'qwen2.5:latest',
          size: '7B',
          context_length: 32768,
          modified_at: new Date().toLocaleDateString(),
          model_id: 'qwen2.5:latest'
        }
      ]);
    } finally {
      setLoadingModels(false);
    }
  };

  // Load meta-task settings
  const loadMetaTaskSettings = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/settings/meta_task');
      if (response.ok) {
        const data = await response.json();
        const settings = data.settings || {};
        // Initialize with defaults if not present
        const defaultModel: MetaTaskModel = {
          model: 'qwen3:30b-a3b',
          max_tokens: 4096,
          temperature: 0.7,
          top_p: 0.9,
          top_k: 40,
          min_p: 0.0,
          repeat_penalty: 1.1
        };
        
        setMetaTaskSettings({
          // General settings
          output_format: settings.output_format || 'markdown',
          max_output_size_mb: settings.max_output_size_mb || 10,
          cache_enabled: settings.cache_enabled !== undefined ? settings.cache_enabled : true,
          cache_ttl_hours: settings.cache_ttl_hours || 1,
          cache_workflows: settings.cache_workflows !== undefined ? settings.cache_workflows : true,
          
          // Execution settings
          execution_retry_attempts: settings.execution_retry_attempts || 3,
          execution_timeout_minutes: settings.execution_timeout_minutes || 5,
          parallel_execution: settings.parallel_execution !== undefined ? settings.parallel_execution : false,
          
          // Quality control
          quality_control_enabled: settings.quality_control_enabled !== undefined ? settings.quality_control_enabled : true,
          quality_min_score: settings.quality_min_score || 0.7,
          quality_auto_retry: settings.quality_auto_retry !== undefined ? settings.quality_auto_retry : true,
          
          // Model configurations
          analyzer_model: settings.analyzer_model || { ...defaultModel, system_prompt: 'You are an expert document analyzer...' },
          reviewer_model: settings.reviewer_model || { ...defaultModel, system_prompt: 'You are a quality reviewer...' },
          assembler_model: settings.assembler_model || { ...defaultModel, system_prompt: 'You are a document assembler...' },
          generator_model: settings.generator_model || { ...defaultModel, system_prompt: 'You are a content generator...' }
        });
      }
    } catch (err) {
      setError('Failed to load meta-task settings');
    } finally {
      setLoading(false);
    }
  };

  // Save meta-task settings
  const saveMetaTaskSettings = async () => {
    try {
      setSaving(true);
      setError('');
      setSuccess('');
      
      const response = await fetch('/api/v1/settings/meta_task', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          settings: metaTaskSettings,
          persist_to_db: true,
          reload_cache: true
        })
      });

      if (response.ok) {
        setSuccess('Settings saved successfully!');
        setTimeout(() => setSuccess(''), 3000);
      } else {
        throw new Error('Failed to save settings');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  // Helper function to format bytes
  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const loadTemplates = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/meta-task/templates');
      if (response.ok) {
        const data = await response.json();
        setTemplates(data.templates || []);
      } else {
        setError('Failed to load templates');
      }
    } catch (err) {
      setError('Error loading templates');
    } finally {
      setLoading(false);
    }
  };

  const loadWorkflows = async () => {
    try {
      // This endpoint would need to be implemented
      const response = await fetch('/api/v1/meta-task/workflows');
      if (response.ok) {
        const data = await response.json();
        setWorkflows(data.workflows || []);
      }
    } catch (err) {
      console.log('Workflows endpoint not implemented yet');
    }
  };

  const createWorkflow = async () => {
    try {
      setLoading(true);
      setError('');
      
      const response = await fetch('/api/v1/meta-task/workflows', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          template_id: selectedTemplate,
          name: workflowName,
          input_data: inputData
        })
      });

      if (response.ok) {
        setCreateWorkflowOpen(false);
        setWorkflowName('');
        setInputData({});
        setSelectedTemplate('');
        loadWorkflows();
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to create workflow');
      }
    } catch (err) {
      setError('Error creating workflow');
    } finally {
      setLoading(false);
    }
  };

  const executeWorkflow = async (workflowId: string) => {
    try {
      // This would open a streaming connection to monitor execution
      window.open(`/api/v1/meta-task/workflows/${workflowId}/execute`, '_blank');
    } catch (err) {
      setError('Error executing workflow');
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'running': return 'primary';
      case 'failed': return 'error';
      case 'pending': return 'default';
      default: return 'default';
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <Box sx={{ 
          position: 'sticky', 
          top: 0, 
          zIndex: 1100, 
          bgcolor: 'background.default',
          boxShadow: 1
        }}>
          <AppBar position="static">
            <Toolbar>
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                Jarvis AI Assistant - Meta-Tasks
              </Typography>
              <IconButton onClick={toggleDarkMode} color="inherit">
                {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
              </IconButton>
            </Toolbar>
          </AppBar>

          {/* Navigation Tabs */}
          <Box sx={{ borderBottom: 1, borderColor: 'divider', bgcolor: 'background.paper' }}>
            <Tabs 
              value={3}
              onChange={handleTabChange} 
              aria-label="jarvis modes"
              centered
              sx={{
                '& .MuiTab-root': {
                  fontSize: '1rem',
                  fontWeight: 600,
                  textTransform: 'none',
                  minWidth: 120,
                  padding: '12px 24px',
                  '&.Mui-selected': {
                    color: 'primary.main',
                    fontWeight: 700
                  }
                }
              }}
            >
              <Tab label="Standard Chat" />
              <Tab label="Multi-Agent" />
              <Tab label="Workflow" />
              <Tab label="Meta-Tasks" />
              <Tab label="Settings" />
              <Tab label="Knowledge Graph" />
              <Tab label="IDC" />
            </Tabs>
          </Box>
        </Box>

        {/* Main Content with Left Nav + Modern Design */}
        <Container maxWidth={false} sx={{ flex: 1, py: 2, overflow: 'hidden', width: '100%' }}>
          <Grid container spacing={2} sx={{ height: '100%' }}>
            {/* Settings Categories Sidebar */}
            <Grid item xs={12} md={2.5}>
              <Paper sx={{ height: '100%', overflow: 'auto' }}>
                <List>
                  <ListItem>
                    <Typography variant="h6" color="primary">
                      Settings Categories
                    </Typography>
                  </ListItem>
                  <Divider />
                  {settingsCategories.map((category) => (
                    <ListItem
                      key={category.id}
                      component="div"
                      sx={{ 
                        cursor: 'pointer',
                        bgcolor: selectedCategory === category.id ? 'action.selected' : 'transparent',
                        '&:hover': { bgcolor: 'action.hover' }
                      }}
                      onClick={() => setSelectedCategory(category.id)}
                    >
                      <ListItemIcon>
                        {category.icon}
                      </ListItemIcon>
                      <ListItemText
                        primary={category.name}
                        secondary={category.description}
                      />
                    </ListItem>
                  ))}
                </List>
              </Paper>
            </Grid>

            {/* Settings Content */}
            <Grid item xs={12} md={9.5}>
              <Paper sx={{ height: '100%', overflow: 'auto', p: 3 }}>
                {selectedCategory === 'meta_task' ? (
                  <Box>
                    {/* Header */}
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="h4" gutterBottom>
                        Meta-Task System
                      </Typography>
                      <Typography variant="body1" color="text.secondary">
                        Configure meta-task models and execution settings for complex multi-phase document generation
                      </Typography>
                    </Box>

                    {error && (
                      <Alert severity="error" sx={{ mb: 2 }}>
                        {error}
                      </Alert>
                    )}

                    {success && (
                      <Alert severity="success" sx={{ mb: 2 }}>
                        {success}
                      </Alert>
                    )}

                    {/* Model Tabs */}
                    <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                      <Tabs 
                        value={activeModelTab} 
                        onChange={(_, newValue) => setActiveModelTab(newValue)}
                        variant="scrollable"
                        scrollButtons="auto"
                      >
                        {modelTabs.map((tab, index) => (
                          <Tab 
                            key={tab.id}
                            label={tab.label}
                            icon={tab.icon}
                            iconPosition="start"
                            value={index}
                          />
                        ))}
                      </Tabs>
                    </Box>

                    {/* Tab Content */}
                    {modelTabs.map((tab, index) => (
                      <Box
                        key={tab.id}
                        hidden={activeModelTab !== index}
                        sx={{ pt: 3 }}
                      >
                        {activeModelTab === index && metaTaskSettings && (
                          <Box>
                            {/* Conditional rendering based on whether it's Settings tab or Model tab */}
                            {index === 0 ? (
                              /* Settings Tab Content */
                              <>
                                {/* General Settings Card */}
                                <Card sx={{ mb: 3 }}>
                                  <CardContent>
                                    <Typography variant="h6" gutterBottom>
                                      General Settings
                                    </Typography>
                                    <Grid container spacing={2}>
                                      <Grid item xs={12} md={6}>
                                        <FormControl fullWidth>
                                          <InputLabel>Output Format</InputLabel>
                                          <Select
                                            value={metaTaskSettings.output_format}
                                            onChange={(e) => {
                                              const newSettings = { ...metaTaskSettings };
                                              newSettings.output_format = e.target.value;
                                              setMetaTaskSettings(newSettings);
                                            }}
                                            label="Output Format"
                                          >
                                            <MenuItem value="markdown">Markdown</MenuItem>
                                            <MenuItem value="json">JSON</MenuItem>
                                            <MenuItem value="text">Plain Text</MenuItem>
                                            <MenuItem value="html">HTML</MenuItem>
                                          </Select>
                                        </FormControl>
                                      </Grid>
                                      <Grid item xs={12} md={6}>
                                        <TextField
                                          fullWidth
                                          label="Max Output Size (MB)"
                                          type="number"
                                          value={metaTaskSettings.max_output_size_mb}
                                          onChange={(e) => {
                                            const newSettings = { ...metaTaskSettings };
                                            newSettings.max_output_size_mb = parseInt(e.target.value) || 1;
                                            setMetaTaskSettings(newSettings);
                                          }}
                                          inputProps={{ min: 1, max: 100 }}
                                        />
                                      </Grid>
                                      <Grid item xs={12} md={6}>
                                        <FormControlLabel
                                          control={
                                            <Switch
                                              checked={metaTaskSettings.cache_enabled}
                                              onChange={(e) => {
                                                const newSettings = { ...metaTaskSettings };
                                                newSettings.cache_enabled = e.target.checked;
                                                setMetaTaskSettings(newSettings);
                                              }}
                                            />
                                          }
                                          label="Cache Enabled"
                                        />
                                      </Grid>
                                      <Grid item xs={12} md={6}>
                                        <TextField
                                          fullWidth
                                          label="Cache TTL (Hours)"
                                          type="number"
                                          value={metaTaskSettings.cache_ttl_hours}
                                          onChange={(e) => {
                                            const newSettings = { ...metaTaskSettings };
                                            newSettings.cache_ttl_hours = parseInt(e.target.value) || 1;
                                            setMetaTaskSettings(newSettings);
                                          }}
                                          inputProps={{ min: 1, max: 24 }}
                                          disabled={!metaTaskSettings.cache_enabled}
                                        />
                                      </Grid>
                                      <Grid item xs={12}>
                                        <FormControlLabel
                                          control={
                                            <Switch
                                              checked={metaTaskSettings.cache_workflows}
                                              onChange={(e) => {
                                                const newSettings = { ...metaTaskSettings };
                                                newSettings.cache_workflows = e.target.checked;
                                                setMetaTaskSettings(newSettings);
                                              }}
                                            />
                                          }
                                          label="Cache Workflows"
                                        />
                                      </Grid>
                                    </Grid>
                                  </CardContent>
                                </Card>

                                {/* Execution Settings Card */}
                                <Card sx={{ mb: 3 }}>
                                  <CardContent>
                                    <Typography variant="h6" gutterBottom>
                                      Execution Settings
                                    </Typography>
                                    <Grid container spacing={2}>
                                      <Grid item xs={12} md={4}>
                                        <TextField
                                          fullWidth
                                          label="Execution Retry Attempts"
                                          type="number"
                                          value={metaTaskSettings.execution_retry_attempts}
                                          onChange={(e) => {
                                            const newSettings = { ...metaTaskSettings };
                                            newSettings.execution_retry_attempts = parseInt(e.target.value) || 0;
                                            setMetaTaskSettings(newSettings);
                                          }}
                                          inputProps={{ min: 0, max: 10 }}
                                        />
                                      </Grid>
                                      <Grid item xs={12} md={4}>
                                        <TextField
                                          fullWidth
                                          label="Execution Timeout (Minutes)"
                                          type="number"
                                          value={metaTaskSettings.execution_timeout_minutes}
                                          onChange={(e) => {
                                            const newSettings = { ...metaTaskSettings };
                                            newSettings.execution_timeout_minutes = parseInt(e.target.value) || 1;
                                            setMetaTaskSettings(newSettings);
                                          }}
                                          inputProps={{ min: 1, max: 30 }}
                                        />
                                      </Grid>
                                      <Grid item xs={12} md={4}>
                                        <FormControlLabel
                                          control={
                                            <Switch
                                              checked={metaTaskSettings.parallel_execution}
                                              onChange={(e) => {
                                                const newSettings = { ...metaTaskSettings };
                                                newSettings.parallel_execution = e.target.checked;
                                                setMetaTaskSettings(newSettings);
                                              }}
                                            />
                                          }
                                          label="Parallel Execution"
                                        />
                                      </Grid>
                                    </Grid>
                                  </CardContent>
                                </Card>

                                {/* Quality Control Card */}
                                <Card sx={{ mb: 3 }}>
                                  <CardContent>
                                    <Typography variant="h6" gutterBottom>
                                      Quality Control
                                    </Typography>
                                    <Grid container spacing={2}>
                                      <Grid item xs={12}>
                                        <FormControlLabel
                                          control={
                                            <Switch
                                              checked={metaTaskSettings.quality_control_enabled}
                                              onChange={(e) => {
                                                const newSettings = { ...metaTaskSettings };
                                                newSettings.quality_control_enabled = e.target.checked;
                                                setMetaTaskSettings(newSettings);
                                              }}
                                            />
                                          }
                                          label="Quality Control Enabled"
                                        />
                                      </Grid>
                                      <Grid item xs={12} md={6}>
                                        <Typography gutterBottom>
                                          Minimum Quality Score: {metaTaskSettings.quality_min_score.toFixed(2)}
                                        </Typography>
                                        <Slider
                                          value={metaTaskSettings.quality_min_score}
                                          onChange={(_, value) => {
                                            const newSettings = { ...metaTaskSettings };
                                            newSettings.quality_min_score = value as number;
                                            setMetaTaskSettings(newSettings);
                                          }}
                                          min={0}
                                          max={1}
                                          step={0.05}
                                          marks
                                          valueLabelDisplay="auto"
                                          disabled={!metaTaskSettings.quality_control_enabled}
                                        />
                                      </Grid>
                                      <Grid item xs={12} md={6}>
                                        <FormControlLabel
                                          control={
                                            <Switch
                                              checked={metaTaskSettings.quality_auto_retry}
                                              onChange={(e) => {
                                                const newSettings = { ...metaTaskSettings };
                                                newSettings.quality_auto_retry = e.target.checked;
                                                setMetaTaskSettings(newSettings);
                                              }}
                                              disabled={!metaTaskSettings.quality_control_enabled}
                                            />
                                          }
                                          label="Auto Retry on Low Quality"
                                        />
                                      </Grid>
                                    </Grid>
                                  </CardContent>
                                </Card>
                              </>
                            ) : (
                              /* Model Tab Content with Submenus */
                              <>
                                {/* Get the correct model based on tab index */}
                                {(() => {
                                  const modelKey = modelTabs[index].id as keyof MetaTaskSettings;
                                  const model = metaTaskSettings[modelKey] as MetaTaskModel;
                                  const currentSubmenu = activeSubmenus[modelKey] || 0;
                                  
                                  // Define submenu tabs for model configuration
                                  const modelSubmenus = [
                                    { label: 'Model Selection', icon: <ModelIcon />, value: 0 },
                                    { label: 'Parameters', icon: <TuneIcon />, value: 1 },
                                    { label: 'Advanced Settings', icon: <AdvancedIcon />, value: 2 },
                                    { label: 'System Prompt', icon: <PromptIcon />, value: 3 }
                                  ];
                                  
                                  return (
                                    <>
                                      {/* Secondary Tabs for Model Configuration */}
                                      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
                                        <Tabs 
                                          value={currentSubmenu}
                                          onChange={(_, newValue) => {
                                            setActiveSubmenus(prev => ({ ...prev, [modelKey]: newValue }));
                                          }}
                                          variant="scrollable"
                                          scrollButtons="auto"
                                          sx={{
                                            '& .MuiTab-root': {
                                              minHeight: 48,
                                              textTransform: 'none',
                                              fontSize: '0.875rem'
                                            }
                                          }}
                                        >
                                          {modelSubmenus.map((submenu) => (
                                            <Tab 
                                              key={submenu.value}
                                              label={submenu.label}
                                              icon={submenu.icon}
                                              iconPosition="start"
                                              value={submenu.value}
                                            />
                                          ))}
                                        </Tabs>
                                      </Box>
                                      
                                      {/* Model Selection Tab */}
                                      {currentSubmenu === 0 && (
                                        <>
                                          {/* Model Info Card */}
                                      <Card sx={{ mb: 3, bgcolor: 'background.default' }}>
                                        <CardContent>
                                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                            <Chip 
                                              label="Active" 
                                              color="success" 
                                              size="small" 
                                              icon={<CheckCircleIcon />}
                                              sx={{ mr: 2 }}
                                            />
                                            <Typography variant="h6">
                                              Current Model: {model.model}
                                            </Typography>
                                          </Box>
                                
                                          <Grid container spacing={2}>
                                            <Grid item xs={3}>
                                              <Box>
                                                <Typography variant="caption" color="text.secondary">
                                                  <MemoryIcon sx={{ fontSize: 14, mr: 0.5 }} />
                                                  Model Size
                                                </Typography>
                                                <Typography variant="body2" fontWeight="bold">
                                                  {availableModels.find(m => m.name === model.model)?.size || 'Unknown'}
                                                </Typography>
                                              </Box>
                                            </Grid>
                                            <Grid item xs={3}>
                                              <Box>
                                                <Typography variant="caption" color="text.secondary">
                                                  <CodeIcon sx={{ fontSize: 14, mr: 0.5 }} />
                                                  Context Length
                                                </Typography>
                                                <Typography variant="body2" fontWeight="bold">
                                                  {availableModels.find(m => m.name === model.model)?.context_length || 8192} tokens
                                                </Typography>
                                              </Box>
                                            </Grid>
                                            <Grid item xs={3}>
                                              <Box>
                                                <Typography variant="caption" color="text.secondary">
                                                  <ScheduleIcon sx={{ fontSize: 14, mr: 0.5 }} />
                                                  Last Modified
                                                </Typography>
                                                <Typography variant="body2" fontWeight="bold">
                                                  {availableModels.find(m => m.name === model.model)?.modified_at || 'Unknown'}
                                                </Typography>
                                              </Box>
                                            </Grid>
                                            <Grid item xs={3}>
                                              <Box>
                                                <Typography variant="caption" color="text.secondary">
                                                  <SpeedIcon sx={{ fontSize: 14, mr: 0.5 }} />
                                                  Model ID
                                                </Typography>
                                                <Typography variant="body2" fontWeight="bold" sx={{ 
                                                  overflow: 'hidden', 
                                                  textOverflow: 'ellipsis',
                                                  whiteSpace: 'nowrap'
                                                }}>
                                                  {availableModels.find(m => m.name === model.model)?.model_id || model.model}
                                                </Typography>
                                              </Box>
                                            </Grid>
                                          </Grid>
                                          
                                          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                                            {modelTabs[index].description}
                                          </Typography>
                                        </CardContent>
                                      </Card>

                                          {/* Model Selection Card */}
                                          <Card sx={{ mb: 3 }}>
                                            <CardContent>
                                              <Typography variant="h6" gutterBottom>
                                                Model Selection
                                              </Typography>
                                              
                                              <FormControl fullWidth sx={{ mb: 3 }}>
                                                <InputLabel>Select Model</InputLabel>
                                                <Select
                                                  value={model.model}
                                                  onChange={(e) => {
                                                    const newSettings = { ...metaTaskSettings };
                                                    (newSettings[modelKey] as MetaTaskModel).model = e.target.value;
                                                    setMetaTaskSettings(newSettings);
                                                  }}
                                                  label="Select Model"
                                                  disabled={loadingModels}
                                                >
                                                  {availableModels.map((availableModel) => (
                                                    <MenuItem key={availableModel.model_id} value={availableModel.name}>
                                                      <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                                                        <Typography>{availableModel.name}</Typography>
                                                        <Typography variant="caption" color="text.secondary">
                                                          {availableModel.size} • {availableModel.context_length} tokens
                                                        </Typography>
                                                      </Box>
                                                    </MenuItem>
                                                  ))}
                                                </Select>
                                              </FormControl>
                                            </CardContent>
                                          </Card>
                                        </>
                                      )}
                                      
                                      {/* Parameters Tab */}
                                      {currentSubmenu === 1 && (
                                        <Card>
                                          <CardContent>
                                            <Typography variant="h6" gutterBottom>
                                              Model Parameters
                                            </Typography>

                                            {/* Model Parameters */}
                                            <Grid container spacing={2}>
                                              <Grid item xs={12} md={6}>
                                                <TextField
                                                  fullWidth
                                                  label="Max Tokens"
                                                  type="number"
                                                  value={model.max_tokens}
                                                  onChange={(e) => {
                                                    const newSettings = { ...metaTaskSettings };
                                                    (newSettings[modelKey] as MetaTaskModel).max_tokens = parseInt(e.target.value) || 0;
                                                    setMetaTaskSettings(newSettings);
                                                  }}
                                                  inputProps={{ min: 1, max: 32768 }}
                                                  helperText="Maximum number of tokens to generate"
                                                />
                                              </Grid>
                                              
                                              {/* Temperature Slider */}
                                              <Grid item xs={12} md={6}>
                                                <Typography gutterBottom>Temperature: {model.temperature}</Typography>
                                                <Slider
                                                  value={model.temperature}
                                                  onChange={(_, value) => {
                                                    const newSettings = { ...metaTaskSettings };
                                                    (newSettings[modelKey] as MetaTaskModel).temperature = value as number;
                                                    setMetaTaskSettings(newSettings);
                                                  }}
                                                  min={0}
                                                  max={2}
                                                  step={0.1}
                                                  marks
                                                  valueLabelDisplay="auto"
                                                />
                                                <Typography variant="caption" color="text.secondary">
                                                  Controls randomness in generation (0 = deterministic, 2 = very creative)
                                                </Typography>
                                              </Grid>
                                              
                                              {/* Top P Slider */}
                                              <Grid item xs={12} md={6}>
                                                <Typography gutterBottom>Top P: {model.top_p}</Typography>
                                                <Slider
                                                  value={model.top_p}
                                                  onChange={(_, value) => {
                                                    const newSettings = { ...metaTaskSettings };
                                                    (newSettings[modelKey] as MetaTaskModel).top_p = value as number;
                                                    setMetaTaskSettings(newSettings);
                                                  }}
                                                  min={0}
                                                  max={1}
                                                  step={0.05}
                                                  marks
                                                  valueLabelDisplay="auto"
                                                />
                                                <Typography variant="caption" color="text.secondary">
                                                  Nucleus sampling parameter (0.9 = use top 90% probability tokens)
                                                </Typography>
                                              </Grid>
                                            </Grid>
                                          </CardContent>
                                        </Card>
                                      )}
                                      
                                      {/* Advanced Settings Tab */}
                                      {currentSubmenu === 2 && (
                                        <Card>
                                          <CardContent>
                                            <Typography variant="h6" gutterBottom>
                                              Advanced Settings
                                            </Typography>

                                            {/* Advanced Parameters */}
                                            <Grid container spacing={2}>
                                              <Grid item xs={12} md={4}>
                                                <TextField
                                                  fullWidth
                                                  label="Top K"
                                                  type="number"
                                                  value={model.top_k}
                                                  onChange={(e) => {
                                                    const newSettings = { ...metaTaskSettings };
                                                    (newSettings[modelKey] as MetaTaskModel).top_k = parseInt(e.target.value) || 0;
                                                    setMetaTaskSettings(newSettings);
                                                  }}
                                                  inputProps={{ min: 0, max: 100 }}
                                                  helperText="Limit token selection to top K tokens"
                                                />
                                              </Grid>
                                              <Grid item xs={12} md={4}>
                                                <TextField
                                                  fullWidth
                                                  label="Min P"
                                                  type="number"
                                                  value={model.min_p}
                                                  onChange={(e) => {
                                                    const newSettings = { ...metaTaskSettings };
                                                    (newSettings[modelKey] as MetaTaskModel).min_p = parseFloat(e.target.value) || 0;
                                                    setMetaTaskSettings(newSettings);
                                                  }}
                                                  inputProps={{ min: 0, max: 1, step: 0.01 }}
                                                  helperText="Minimum probability threshold"
                                                />
                                              </Grid>
                                              <Grid item xs={12} md={4}>
                                                <TextField
                                                  fullWidth
                                                  label="Repeat Penalty"
                                                  type="number"
                                                  value={model.repeat_penalty}
                                                  onChange={(e) => {
                                                    const newSettings = { ...metaTaskSettings };
                                                    (newSettings[modelKey] as MetaTaskModel).repeat_penalty = parseFloat(e.target.value) || 1;
                                                    setMetaTaskSettings(newSettings);
                                                  }}
                                                  inputProps={{ min: 0.1, max: 2, step: 0.1 }}
                                                  helperText="Penalty for repeating tokens (1.0 = no penalty)"
                                                />
                                              </Grid>
                                            </Grid>
                                          </CardContent>
                                        </Card>
                                      )}

                                      {/* System Prompt Tab */}
                                      {currentSubmenu === 3 && (
                                        <Card>
                                          <CardContent>
                                            <Typography variant="h6" gutterBottom>
                                              System Prompt
                                            </Typography>
                                            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                              Define the behavior and role of the {modelTabs[index].label.toLowerCase()} in the meta-task workflow
                                            </Typography>
                                            <TextField
                                              fullWidth
                                              multiline
                                              rows={12}
                                              value={model.system_prompt || ''}
                                              onChange={(e) => {
                                                const newSettings = { ...metaTaskSettings };
                                                (newSettings[modelKey] as MetaTaskModel).system_prompt = e.target.value;
                                                setMetaTaskSettings(newSettings);
                                              }}
                                              placeholder={`Enter system prompt for ${modelTabs[index].label}...\n\nExample:\nYou are an expert ${modelTabs[index].label.toLowerCase().replace(' model', '')} responsible for...`}
                                              variant="outlined"
                                              sx={{
                                                '& .MuiInputBase-input': {
                                                  fontFamily: 'monospace',
                                                  fontSize: '0.9rem'
                                                }
                                              }}
                                            />
                                          </CardContent>
                                        </Card>
                                      )}
                                    </>
                                  );
                                })()}
                              </>
                            )}

                          </Box>
                        )}
                      </Box>
                    ))}
                    
                    {/* Action Buttons - Outside of tabs, always visible */}
                    <Box sx={{ mt: 3, display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
                      <Button
                        variant="outlined"
                        startIcon={<RefreshIcon />}
                        onClick={() => {
                          loadMetaTaskSettings();
                          loadAvailableModels();
                        }}
                        disabled={loading || loadingModels}
                      >
                        {loading || loadingModels ? (
                          <>
                            <CircularProgress size={16} sx={{ mr: 1 }} />
                            Refreshing...
                          </>
                        ) : (
                          'REFRESH SETTINGS'
                        )}
                      </Button>
                      <Button
                        variant="contained"
                        startIcon={<SaveIcon />}
                        onClick={saveMetaTaskSettings}
                        disabled={saving || !metaTaskSettings}
                      >
                        {saving ? (
                          <>
                            <CircularProgress size={16} sx={{ mr: 1, color: 'white' }} />
                            Saving...
                          </>
                        ) : (
                          'UPDATE MODELS & CACHE'
                        )}
                      </Button>
                    </Box>
                  </Box>
                ) : (
                  /* Templates & Workflows view */
                  <Grid container spacing={3}>
                    <Grid item xs={12}>
                      <Typography variant="h4" gutterBottom>
                        Templates & Workflows
                      </Typography>
                      <Typography variant="body1" color="text.secondary" gutterBottom>
                        Manage templates and active workflows for meta-task execution
                      </Typography>
                    </Grid>
                    
                    {/* Templates Section */}
                    <Grid item xs={12} md={6}>
                      <Paper sx={{ p: 3 }}>
                        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                          <Typography variant="h5">
                            Templates
                          </Typography>
                          <Button
                            variant="contained"
                            startIcon={<AddIcon />}
                            onClick={() => setCreateWorkflowOpen(true)}
                            disabled={templates.length === 0}
                          >
                            Create Workflow
                          </Button>
                        </Box>

                        {templates.length === 0 ? (
                          <Typography variant="body1" color="text.secondary">
                            No templates available. Templates are created through database migrations.
                          </Typography>
                        ) : (
                          <Grid container spacing={2}>
                            {templates.map((template) => (
                              <Grid item xs={12} key={template.id}>
                                <Card variant="outlined">
                                  <CardContent>
                                    <Typography variant="h6">
                                      {template.name}
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary" paragraph>
                                      {template.description}
                                    </Typography>
                                    <Chip 
                                      label={template.template_type} 
                                      size="small" 
                                      color="primary" 
                                      variant="outlined" 
                                    />
                                    <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                                      Phases: {template.template_config?.phases?.length || 0}
                                    </Typography>
                                  </CardContent>
                                  <CardActions>
                                    <Button 
                                      size="small" 
                                      startIcon={<PlayIcon />}
                                      onClick={() => {
                                        setSelectedTemplate(template.id);
                                        setCreateWorkflowOpen(true);
                                      }}
                                    >
                                      Use Template
                                    </Button>
                                  </CardActions>
                                </Card>
                              </Grid>
                            ))}
                          </Grid>
                        )}
                      </Paper>
                    </Grid>

                    {/* Workflows Section */}
                    <Grid item xs={12} md={6}>
                      <Paper sx={{ p: 3 }}>
                        <Typography variant="h5" mb={2}>
                          Active Workflows
                        </Typography>

                        {workflows.length === 0 ? (
                          <Typography variant="body1" color="text.secondary">
                            No active workflows. Create a workflow from a template to get started.
                          </Typography>
                        ) : (
                          <List>
                            {workflows.map((workflow) => (
                              <ListItem key={workflow.id} divider>
                                <ListItemText
                                  primary={workflow.name}
                                  secondary={`Created: ${new Date(workflow.created_at).toLocaleDateString()}`}
                                />
                                <Chip 
                                  label={workflow.status}
                                  color={getStatusColor(workflow.status) as any}
                                  size="small"
                                  sx={{ mr: 1 }}
                                />
                                <IconButton 
                                  onClick={() => executeWorkflow(workflow.id)}
                                  disabled={workflow.status === 'running'}
                                >
                                  <PlayIcon />
                                </IconButton>
                              </ListItem>
                            ))}
                          </List>
                        )}
                      </Paper>
                    </Grid>
                  </Grid>
                )}
              </Paper>
            </Grid>
          </Grid>
        </Container>

        {/* Create Workflow Dialog */}
        <Dialog 
          open={createWorkflowOpen} 
          onClose={() => setCreateWorkflowOpen(false)}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>Create New Workflow</DialogTitle>
          <DialogContent>
            <Box sx={{ pt: 1 }}>
              <FormControl fullWidth margin="normal">
                <InputLabel>Template</InputLabel>
                <Select
                  value={selectedTemplate}
                  onChange={(e) => setSelectedTemplate(e.target.value)}
                  label="Template"
                >
                  {templates.map((template) => (
                    <MenuItem key={template.id} value={template.id}>
                      {template.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <TextField
                fullWidth
                label="Workflow Name"
                value={workflowName}
                onChange={(e) => setWorkflowName(e.target.value)}
                margin="normal"
                required
              />

              <TextField
                fullWidth
                label="Input Data (JSON)"
                value={JSON.stringify(inputData, null, 2)}
                onChange={(e) => {
                  try {
                    setInputData(JSON.parse(e.target.value));
                  } catch {
                    // Invalid JSON, keep as string
                  }
                }}
                multiline
                rows={6}
                margin="normal"
                placeholder='{"topic": "Your topic here", "target_length": 10}'
                helperText="Enter the input data as JSON. Required fields depend on the selected template."
              />
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setCreateWorkflowOpen(false)}>
              Cancel
            </Button>
            <Button 
              onClick={createWorkflow}
              variant="contained"
              disabled={!selectedTemplate || !workflowName || loading}
            >
              Create Workflow
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </ThemeProvider>
  );
}

export default MetaTaskApp;