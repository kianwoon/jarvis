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
  Grid
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
  Save as SaveIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import SettingsFormRenderer from './components/settings/SettingsFormRenderer';
import './styles/settings-theme.css';

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
    id: 'langgraph_agents',
    name: 'LangGraph Agents',
    icon: <GroupIcon />,
    description: 'Agent definitions, roles, system prompts, tools, and configurations'
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
        window.location.href = '/workflow.html';
        break;
      case 3:
        // Already on settings page
        break;
    }
  };

  const loadSettings = async (category: string, force: boolean = false) => {
    if (settingsData[category] && !force) return; // Already loaded

    setLoading(prev => ({ ...prev, [category]: true }));
    setError(prev => ({ ...prev, [category]: '' }));

    try {
      let response;
      let data;

      // Handle different endpoint patterns based on category
      if (category === 'mcp') {
        // MCP uses multiple endpoints for servers and tools
        const [serversResponse, toolsResponse] = await Promise.all([
          fetch('/api/v1/mcp/servers/'),
          fetch('/api/v1/mcp/tools/')
        ]);
        
        let servers = [];
        let tools = [];
        
        if (serversResponse.ok) {
          servers = await serversResponse.json();
        } else {
          console.error(`Failed to load MCP servers: ${serversResponse.status} ${serversResponse.statusText}`);
        }
        
        if (toolsResponse.ok) {
          tools = await toolsResponse.json();
        } else {
          console.error(`Failed to load MCP tools: ${toolsResponse.status} ${toolsResponse.statusText}`);
        }
        
        data = {
          servers: Array.isArray(servers) ? servers : servers.data || [],
          tools: Array.isArray(tools) ? tools : tools.data || []
        };
      } else if (category === 'langgraph_agents') {
        // LangGraph agents use a specific endpoint
        response = await fetch('/api/v1/langgraph/agents');
        if (response.ok) {
          data = await response.json();
        } else {
          data = { agents: [] };
        }
      } else if (category === 'collection_registry') {
        // Collection registry might use a different endpoint
        response = await fetch('/api/v1/collections/');
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
    const errorMsg = error[category];
    const successMsg = success[category];

    if (isLoading) {
      return (
        <div className="jarvis-loading">
          <div className="jarvis-spinner"></div>
          <Typography>Loading {category} settings...</Typography>
        </div>
      );
    }

    if (errorMsg) {
      return (
        <div className="jarvis-alert jarvis-alert-error">
          {errorMsg}
          <button 
            className="jarvis-btn jarvis-btn-secondary"
            onClick={() => loadSettings(category)}
            style={{ marginLeft: '12px', padding: '6px 12px', fontSize: '12px' }}
          >
            Retry
          </button>
        </div>
      );
    }

    if (!data) {
      return (
        <div className="jarvis-alert jarvis-alert-info">
          Click "Reload" in the header to load and edit {category} configuration
        </div>
      );
    }

    return (
      <div>
        {successMsg && (
          <div className="jarvis-alert jarvis-alert-success">
            Settings saved successfully!
          </div>
        )}
        
        <div className="settings-section-header">
          <div className="settings-section-title">Configuration Details</div>
          <div className="settings-section-subtitle">
            Configure your {settingsCategories.find(cat => cat.id === category)?.name.toLowerCase()} settings below
          </div>
        </div>
        
        <SettingsFormRenderer
          category={category}
          data={data}
          onChange={(field, value) => handleFieldChange(category, field, value)}
          onRefresh={() => {
            setSettingsData(prev => {
              const updated = { ...prev };
              delete updated[category];
              return updated;
            });
            loadSettings(category, true);
          }}
          isYamlBased={category === 'self_reflection' || category === 'query_patterns'}
        />
      </div>
    );
  };


  // Load initial category on mount
  useEffect(() => {
    loadSettings(selectedCategory);
  }, []);

  // Set theme data attribute for CSS
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

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

        {/* Main Content with Left Nav + Modern Design */}
        <Container maxWidth={false} sx={{ flex: 1, py: 2, overflow: 'hidden' }}>
          <Grid container spacing={2} sx={{ height: '100%' }}>
            {/* Settings Categories Sidebar */}
            <Grid item xs={12} md={3} sx={{ height: '100%' }}>
              <Paper sx={{ height: '100%', overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
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
            <Grid item xs={12} md={9} sx={{ height: '100%' }}>
              <Paper sx={{ height: '100%', overflow: 'auto', display: 'flex', flexDirection: 'column', p: 2 }}>
                {/* Modern Header */}
                <div className="settings-header">
                  <div className="settings-header-content">
                    <h1>{settingsCategories.find(cat => cat.id === selectedCategory)?.name || 'Settings'}</h1>
                    <p>{settingsCategories.find(cat => cat.id === selectedCategory)?.description || 'Configure your settings'}</p>
                  </div>
                  <div className="settings-header-actions">
                    <button 
                      className="jarvis-btn jarvis-btn-secondary"
                      onClick={() => {
                        setSettingsData(prev => {
                          const updated = { ...prev };
                          delete updated[selectedCategory];
                          return updated;
                        });
                        loadSettings(selectedCategory, true);
                      }}
                      disabled={loading[selectedCategory]}
                    >
                      <RefreshIcon sx={{ fontSize: 16 }} />
                      Reload
                    </button>
                    <button 
                      className="jarvis-btn jarvis-btn-primary"
                      onClick={() => saveSettings(selectedCategory, settingsData[selectedCategory])}
                      disabled={saving[selectedCategory] || !settingsData[selectedCategory]}
                    >
                      <SaveIcon sx={{ fontSize: 16 }} />
                      {saving[selectedCategory] ? 'Saving...' : 'Save Settings'}
                    </button>
                  </div>
                </div>

                {/* Settings Content */}
                <Box sx={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
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