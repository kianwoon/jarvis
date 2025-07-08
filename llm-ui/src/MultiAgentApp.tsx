import React, { useState, useEffect } from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Paper,
  Grid,
  Container,
  ThemeProvider,
  createTheme,
  CssBaseline,
  Button
} from '@mui/material';
import {
  LightMode as LightModeIcon,
  DarkMode as DarkModeIcon,
  ArrowBack as ArrowBackIcon,
  Group as GroupIcon
} from '@mui/icons-material';
import { Agent, MultiAgentMessage, AgentStatus, CollaborationPhase } from './types/MultiAgent';
import AgentSelector from './components/multiagent/AgentSelector';
import CollaborationWorkspace from './components/multiagent/CollaborationWorkspace';
import MultiAgentChat from './components/multiagent/MultiAgentChat';

function MultiAgentApp() {
  // Theme management
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('jarvis-dark-mode');
    return saved ? JSON.parse(saved) : false;
  });

  // Multi-agent state
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedAgents, setSelectedAgents] = useState<Agent[]>([]);
  const [messages, setMessages] = useState<MultiAgentMessage[]>([]);
  const [agentStatuses, setAgentStatuses] = useState<Record<string, AgentStatus>>({});
  const [collaborationPhase, setCollaborationPhase] = useState<CollaborationPhase>({
    phase: 'selection',
    status: 'pending',
    progress: 0,
    description: 'Select agents for collaboration',
    agents_involved: []
  });
  const [loading, setLoading] = useState(false);

  // Session management
  const [sessionId] = useState(() => `multi-agent-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`);
  const storageKey = 'jarvis-multi-agent-conversations';

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

  // Load agents on component mount
  useEffect(() => {
    loadAgents();
    loadConversationHistory();
  }, []);

  // Save conversations to localStorage
  useEffect(() => {
    if (messages.length > 0) {
      const conversations = JSON.parse(localStorage.getItem(storageKey) || '{}');
      conversations[sessionId] = messages;
      localStorage.setItem(storageKey, JSON.stringify(conversations));
    }
  }, [messages, sessionId, storageKey]);

  const loadAgents = async () => {
    try {
      const response = await fetch('/api/v1/langchain/agents');
      if (response.ok) {
        const agentsData = await response.json();
        setAgents(agentsData.agents || []);
      } else {
        console.error('Failed to load agents');
      }
    } catch (error) {
      console.error('Error loading agents:', error);
    }
  };

  const loadConversationHistory = () => {
    try {
      const conversations = JSON.parse(localStorage.getItem(storageKey) || '{}');
      if (conversations[sessionId]) {
        const savedMessages = conversations[sessionId].map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }));
        setMessages(savedMessages);
      }
    } catch (error) {
      console.error('Error loading conversation history:', error);
    }
  };

  const toggleDarkMode = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem('jarvis-dark-mode', JSON.stringify(newDarkMode));
  };

  const handleAgentSelection = (agents: Agent[]) => {
    setSelectedAgents(agents);
    
    // Update agent statuses
    const newStatuses: Record<string, AgentStatus> = {};
    agents.forEach(agent => {
      newStatuses[agent.name] = {
        name: agent.name,
        status: 'selected',
        current_task: 'Waiting for query'
      };
    });
    setAgentStatuses(newStatuses);

    // Update collaboration phase
    setCollaborationPhase({
      phase: 'selection',
      status: 'complete',
      progress: 100,
      description: `Selected ${agents.length} agent${agents.length > 1 ? 's' : ''} for collaboration`,
      agents_involved: agents.map(a => a.name)
    });
  };

  const clearConversation = () => {
    setMessages([]);
    setAgentStatuses({});
    setSelectedAgents([]);
    setCollaborationPhase({
      phase: 'selection',
      status: 'pending',
      progress: 0,
      description: 'Select agents for collaboration',
      agents_involved: []
    });
    
    // Clear from localStorage
    const conversations = JSON.parse(localStorage.getItem(storageKey) || '{}');
    delete conversations[sessionId];
    localStorage.setItem(storageKey, JSON.stringify(conversations));
  };

  const goToStandardChat = () => {
    window.location.href = '/';
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <AppBar position="static">
          <Toolbar>
            <IconButton onClick={goToStandardChat} color="inherit" sx={{ mr: 2 }}>
              <ArrowBackIcon />
            </IconButton>
            
            <GroupIcon sx={{ mr: 1 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Jarvis Multi-Agent Collaboration
            </Typography>
            
            <Button
              variant="outlined"
              onClick={clearConversation}
              sx={{ mr: 2, color: 'white', borderColor: 'white' }}
            >
              New Session
            </Button>

            <IconButton onClick={toggleDarkMode} color="inherit">
              {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
            </IconButton>
          </Toolbar>
        </AppBar>

        {/* Main Content */}
        <Container maxWidth={false} sx={{ flex: 1, py: 2, overflow: 'hidden' }}>
          <Grid container spacing={2} sx={{ height: '100%' }}>
            {/* Agent Selection Panel */}
            <Grid item xs={12} md={3}>
              <Paper sx={{ height: '100%', p: 2, overflow: 'auto' }}>
                <Typography variant="h6" gutterBottom>
                  Agent Selection
                </Typography>
                <AgentSelector
                  agents={agents}
                  selectedAgents={selectedAgents}
                  onAgentSelection={handleAgentSelection}
                  collaborationPhase={collaborationPhase}
                />
              </Paper>
            </Grid>

            {/* Collaboration Workspace */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
                  <Typography variant="h6">
                    Collaboration Workspace
                  </Typography>
                </Box>
                <Box sx={{ flex: 1, overflow: 'hidden' }}>
                  <CollaborationWorkspace
                    agentStatuses={agentStatuses}
                    collaborationPhase={collaborationPhase}
                    selectedAgents={selectedAgents}
                  />
                </Box>
              </Paper>
            </Grid>

            {/* Chat Interface */}
            <Grid item xs={12} md={3}>
              <Paper sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
                  <Typography variant="h6">
                    Multi-Agent Chat
                  </Typography>
                </Box>
                <Box sx={{ flex: 1, overflow: 'hidden' }}>
                  <MultiAgentChat
                    messages={messages}
                    setMessages={setMessages}
                    selectedAgents={selectedAgents}
                    sessionId={sessionId}
                    loading={loading}
                    setLoading={setLoading}
                    setAgentStatuses={setAgentStatuses}
                    setCollaborationPhase={setCollaborationPhase}
                  />
                </Box>
              </Paper>
            </Grid>
          </Grid>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default MultiAgentApp;