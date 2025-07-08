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
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Grid,
  Card,
  CardContent,
  CardHeader,
  TextField,
  Switch,
  FormControlLabel,
  Button,
  Alert,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip
} from '@mui/material';
import {
  LightMode as LightModeIcon,
  DarkMode as DarkModeIcon,
  Chat as ChatIcon,
  Group as GroupIcon,
  AccountTree as WorkflowIcon,
  Settings as SettingsIcon,
  Psychology as AIIcon,
  Search as SearchIcon,
  Storage as StorageIcon,
  Extension as IntegrationsIcon,
  Speed as PerformanceIcon,
  Computer as SystemIcon,
  ExpandMore as ExpandMoreIcon,
  Save as SaveIcon,
  Refresh as RefreshIcon,
  Code as CodeIcon,
  Description as YamlIcon
} from '@mui/icons-material';
import SettingsFormRenderer from './components/settings/SettingsFormRenderer';

interface SettingsCategory {
  id: string;
  name: string;
  icon: React.ReactNode;
  description: string;
}

interface SettingsData {
  [key: string]: any;
}

const settingsCategories: SettingsCategory[] = [
  {
    id: 'llm',
    name: 'AI Models & LLM',
    icon: <AIIcon />,
    description: 'Model selection, parameters, and query classification settings'
  },
  {
    id: 'rag',
    name: 'RAG & Search',
    icon: <SearchIcon />,
    description: 'Document retrieval, search strategies, and reranking settings'
  },
  {
    id: 'storage',
    name: 'Storage & Databases',
    icon: <StorageIcon />,
    description: 'Vector databases, embedding models, and storage configuration'
  },
  {
    id: 'langfuse',
    name: 'Monitoring & Tracing',
    icon: <IntegrationsIcon />,
    description: 'Langfuse monitoring, tracing, and cost tracking settings'
  },
  {
    id: 'large_generation',
    name: 'Performance Optimization',
    icon: <PerformanceIcon />,
    description: 'Large generation settings, memory management, and performance tuning'
  },
  {
    id: 'mcp',
    name: 'MCP Tools & Integrations',
    icon: <SystemIcon />,
    description: 'MCP server configurations, tool management, and API settings'
  },
  {
    id: 'self_reflection',
    name: 'Self-Reflection & Quality',
    icon: <AIIcon />,
    description: 'Self-reflection modes, quality evaluation, and refinement strategies'
  },
  {
    id: 'agent_behaviors',
    name: 'Agent Behaviors',
    icon: <GroupIcon />,
    description: 'Agent behavior definitions, capabilities, and communication patterns'
  },
  {
    id: 'query_patterns',
    name: 'Query Classification',
    icon: <SearchIcon />,
    description: 'Query classification patterns, smart routing, and classification rules'
  },
  {
    id: 'collection_registry',
    name: 'Collection Management',
    icon: <StorageIcon />,
    description: 'Collection metadata, access controls, and registry management'
  },
  {
    id: 'environment',
    name: 'Environment & Runtime',
    icon: <SystemIcon />,
    description: 'Environment variables, runtime configuration, and system overrides'
  }
];

function SettingsApp() {
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('jarvis-dark-mode');
    return saved ? JSON.parse(saved) : false;
  });

  const [selectedCategory, setSelectedCategory] = useState('llm');
  const [settingsData, setSettingsData] = useState<Record<string, SettingsData>>({});
  const [loading, setLoading] = useState<Record<string, boolean>>({});
  const [saving, setSaving] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<Record<string, string>>({});
  const [success, setSuccess] = useState<Record<string, boolean>>({});

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
        window.location.href = '/?tab=2';
        break;
      case 3:
        // Already on settings page
        break;
    }
  };

  const loadSettings = async (category: string) => {
    if (settingsData[category]) return; // Already loaded

    setLoading(prev => ({ ...prev, [category]: true }));
    setError(prev => ({ ...prev, [category]: '' }));

    try {
      let response;
      let data;

      // Handle different endpoint patterns based on category
      if (category === 'collection_registry') {
        // Collection registry might use a different endpoint
        response = await fetch('/api/v1/collections');
        if (response.ok) {
          data = await response.json();
        } else {
          data = { collections: [] };
        }
      } else if (category === 'environment') {
        // Environment variables from config
        response = await fetch('/api/v1/system/config');
        if (response.ok) {
          data = await response.json();
        } else {
          data = { environment_variables: {} };
        }
      } else {
        // Standard settings endpoint
        response = await fetch(`/api/v1/settings/${category}`);
        if (!response.ok) {
          throw new Error(`Failed to load ${category} settings: ${response.statusText}`);
        }
        data = await response.json();
      }

      setSettingsData(prev => ({ ...prev, [category]: data }));
    } catch (err) {
      setError(prev => ({ 
        ...prev, 
        [category]: err instanceof Error ? err.message : 'Failed to load settings'
      }));
    } finally {
      setLoading(prev => ({ ...prev, [category]: false }));
    }
  };

  const saveSettings = async (category: string, data: SettingsData) => {
    setSaving(prev => ({ ...prev, [category]: true }));
    setError(prev => ({ ...prev, [category]: '' }));
    setSuccess(prev => ({ ...prev, [category]: false }));

    try {
      const response = await fetch(`/api/v1/settings/${category}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        throw new Error(`Failed to save ${category} settings: ${response.statusText}`);
      }

      setSuccess(prev => ({ ...prev, [category]: true }));
      setTimeout(() => {
        setSuccess(prev => ({ ...prev, [category]: false }));
      }, 3000);
    } catch (err) {
      setError(prev => ({ 
        ...prev, 
        [category]: err instanceof Error ? err.message : 'Failed to save settings'
      }));
    } finally {
      setSaving(prev => ({ ...prev, [category]: false }));
    }
  };

  const handleCategorySelect = (categoryId: string) => {
    setSelectedCategory(categoryId);
    loadSettings(categoryId);
  };

  const handleFieldChange = (category: string, field: string, value: any) => {
    setSettingsData(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [field]: value
      }
    }));
  };

  const renderSettingsForm = (category: string) => {
    const data = settingsData[category];
    const isLoading = loading[category];
    const isSaving = saving[category];
    const errorMsg = error[category];
    const successMsg = success[category];

    if (isLoading) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200 }}>
          <CircularProgress />
          <Typography sx={{ ml: 2 }}>Loading {category} settings...</Typography>
        </Box>
      );
    }

    if (errorMsg) {
      return (
        <Alert severity="error" sx={{ mb: 2 }}>
          {errorMsg}
          <Button 
            size="small" 
            onClick={() => loadSettings(category)}
            sx={{ ml: 1 }}
          >
            Retry
          </Button>
        </Alert>
      );
    }

    if (!data) {
      return (
        <Alert severity="info">
          Click "Load Settings" to view and edit {category} configuration
        </Alert>
      );
    }

    return (
      <Box>
        {successMsg && (
          <Alert severity="success" sx={{ mb: 2 }}>
            Settings saved successfully!
          </Alert>
        )}
        
        <Card>
          <CardHeader 
            title={`${settingsCategories.find(cat => cat.id === category)?.name} Configuration`}
            action={
              <Box>
                <Button
                  startIcon={<RefreshIcon />}
                  onClick={() => {
                    setSettingsData(prev => {
                      const updated = { ...prev };
                      delete updated[category];
                      return updated;
                    });
                    loadSettings(category);
                  }}
                  sx={{ mr: 1 }}
                >
                  Reload
                </Button>
                <Button
                  variant="contained"
                  startIcon={<SaveIcon />}
                  onClick={() => saveSettings(category, data)}
                  disabled={isSaving}
                >
                  {isSaving ? 'Saving...' : 'Save Settings'}
                </Button>
              </Box>
            }
          />
          <CardContent>
            <SettingsFormRenderer
              category={category}
              data={data}
              onChange={(field, value) => handleFieldChange(category, field, value)}
              isYamlBased={category === 'self_reflection' || category === 'agent_behaviors' || category === 'query_patterns'}
            />
          </CardContent>
        </Card>
      </Box>
    );
  };


  // Load initial category on mount
  useEffect(() => {
    loadSettings(selectedCategory);
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Jarvis AI Assistant
            </Typography>

            <IconButton onClick={toggleDarkMode} color="inherit">
              {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
            </IconButton>
          </Toolbar>
        </AppBar>

        {/* Navigation Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={3}
            onChange={handleTabChange} 
            aria-label="jarvis modes"
            centered
          >
            <Tab 
              icon={<ChatIcon />} 
              label="Standard Chat" 
              id="tab-0"
              aria-controls="tabpanel-0"
            />
            <Tab 
              icon={<GroupIcon />} 
              label="Multi-Agent" 
              id="tab-1"
              aria-controls="tabpanel-1"
            />
            <Tab 
              icon={<WorkflowIcon />} 
              label="Workflow" 
              id="tab-2"
              aria-controls="tabpanel-2"
            />
            <Tab 
              icon={<SettingsIcon />} 
              label="Settings" 
              id="tab-3"
              aria-controls="tabpanel-3"
            />
          </Tabs>
        </Box>

        {/* Main Content */}
        <Container maxWidth={false} sx={{ flex: 1, py: 2, overflow: 'hidden' }}>
          <Grid container spacing={2} sx={{ height: '100%' }}>
            {/* Settings Categories Sidebar */}
            <Grid item xs={12} md={3}>
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
                      button
                      selected={selectedCategory === category.id}
                      onClick={() => handleCategorySelect(category.id)}
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
            <Grid item xs={12} md={9}>
              <Paper sx={{ height: '100%', overflow: 'auto', p: 2 }}>
                <Box sx={{ height: '100%', overflow: 'auto' }}>
                  {renderSettingsForm(selectedCategory)}
                </Box>
              </Paper>
            </Grid>
          </Grid>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default SettingsApp;