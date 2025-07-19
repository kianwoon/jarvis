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
  Button,
  Tabs,
  Tab
} from '@mui/material';
import {
  LightMode as LightModeIcon,
  DarkMode as DarkModeIcon,
  Group as GroupIcon,
  Chat as ChatIcon,
  AccountTree as WorkflowIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';
import { Agent, MultiAgentMessage, AgentStatus, CollaborationPhase } from './types/MultiAgent';
import CollaborationWorkspace from './components/multiagent/CollaborationWorkspace';
import MultiAgentChat from './components/multiagent/MultiAgentChat';
import AgentResponseWindow from './components/multiagent/AgentResponseWindow';

function MultiAgentApp() {
  // Theme management
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('jarvis-dark-mode');
    return saved ? JSON.parse(saved) : false;
  });

  // Multi-agent state
  const [messages, setMessages] = useState<MultiAgentMessage[]>([]);
  const [agentStatuses, setAgentStatuses] = useState<Record<string, AgentStatus>>({});
  const [agentStreamingContent, setAgentStreamingContent] = useState<Record<string, string>>({});
  const [activeAgents, setActiveAgents] = useState<Agent[]>([]);
  const [collaborationPhase, setCollaborationPhase] = useState<CollaborationPhase>({
    phase: 'ready',
    status: 'pending',
    progress: 0,
    description: 'Ready to start multi-agent collaboration',
    agents_involved: [],
    completed_agents: []
  });
  const [loading, setLoading] = useState(false);

  // Session management
  const [sessionId] = useState(() => `multi-agent-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`);
  const storageKey = 'jarvis-multi-agent-chat'; // Static key like standard chat

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

  // Load conversation history on component mount
  useEffect(() => {
    loadConversationHistory();
  }, []);

  // Set theme data attribute for CSS
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  // Save conversations to localStorage
  useEffect(() => {
    if (messages.length > 0) {
      //console.log('Saving messages to localStorage:', messages.length, 'messages');
      localStorage.setItem(storageKey, JSON.stringify(messages));
      //console.log('Messages saved to key:', storageKey);
    }
  }, [messages, storageKey]);


  const loadConversationHistory = () => {
    try {
      //console.log('Loading conversation history with key:', storageKey);
      const savedMessages = localStorage.getItem(storageKey);
      //console.log('Saved messages from localStorage:', savedMessages);
      if (savedMessages) {
        const parsed = JSON.parse(savedMessages);
        //console.log('Parsed messages:', parsed);
        const restoredMessages = parsed.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }));
        setMessages(restoredMessages);
        
        // Restore activeAgents from assistant messages with agent_contributions
        const assistantMessages = restoredMessages.filter((msg: any) => msg.role === 'assistant');
        const allAgentContributions = assistantMessages.flatMap((msg: any) => msg.agent_contributions || []);
        
        if (allAgentContributions.length > 0) {
          const uniqueAgents = Array.from(new Set(allAgentContributions.map((contrib: any) => contrib.agent_name)))
            .map(agentName => ({
              id: agentName.toLowerCase().replace(/\s+/g, '-'),
              name: agentName,
              role: allAgentContributions.find((contrib: any) => contrib.agent_name === agentName)?.agent_role || 'AI Agent',
              system_prompt: '',
              tools: [],
              description: '',
              is_active: true,
              config: {}
            }));
          
          //console.log('Restoring active agents:', uniqueAgents);
          setActiveAgents(uniqueAgents);
          
          // Set collaboration phase to complete
          setCollaborationPhase({
            phase: 'complete',
            status: 'complete',
            progress: 100,
            description: 'Previous collaboration session',
            agents_involved: uniqueAgents.map(agent => agent.name),
            completed_agents: uniqueAgents.map(agent => agent.name) // All agents are complete in loaded session
          });
        }
        
        //console.log('Messages and agents loaded successfully');
      } else {
        //console.log('No saved messages found');
      }
    } catch (error) {
      //console.error('Error loading conversation history:', error);
    }
  };

  const toggleDarkMode = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem('jarvis-dark-mode', JSON.stringify(newDarkMode));
  };


  const clearConversation = () => {
    setMessages([]);
    setAgentStatuses({});
    setAgentStreamingContent({});
    setActiveAgents([]);
    setCollaborationPhase({
      phase: 'ready',
      status: 'pending',
      progress: 0,
      description: 'Ready to start multi-agent collaboration',
      agents_involved: [],
      completed_agents: []
    });
    
    // Clear from localStorage
    localStorage.removeItem(storageKey);
  };

  const goToStandardChat = () => {
    window.location.href = '/';
  };

  const goToWorkflow = () => {
    // Navigate to workflow page
    window.location.href = '/workflow.html';
  };

  const goToSettings = () => {
    // Navigate to settings page
    window.location.href = '/settings.html';
  };

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    switch (newValue) {
      case 0:
        goToStandardChat();
        break;
      case 1:
        // Already on multi-agent page, do nothing
        break;
      case 2:
        goToWorkflow();
        break;
      case 3:
        goToSettings();
        break;
    }
  };

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

        {/* Navigation Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={1}
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
            <Tab 
              label="Standard Chat" 
              id="tab-0"
              aria-controls="tabpanel-0"
            />
            <Tab 
              label="Multi-Agent" 
              id="tab-1"
              aria-controls="tabpanel-1"
            />
            <Tab 
              label="Workflow" 
              id="tab-2"
              aria-controls="tabpanel-2"
            />
            <Tab 
              label="Settings" 
              id="tab-3"
              aria-controls="tabpanel-3"
            />
          </Tabs>
        </Box>

        {/* Main Content */}
        <Container maxWidth={false} sx={{ flex: 1, py: 2, overflow: 'hidden' }}>
          <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Top Section: Input and Collaboration Status */}
            <Box sx={{ height: 'auto', maxHeight: '300px', mb: 2 }}>
              <Paper sx={{ height: '100%', p: 2, display: 'flex', flexDirection: 'column', overflow: 'auto' }}>
                <Typography variant="h6" gutterBottom>
                  Multi-Agent Collaboration
                </Typography>
                
                {/* Collaboration Phase Status */}
                {activeAgents.length > 0 && (
                  <Box sx={{ mb: 1 }}>
                    <CollaborationWorkspace
                      agentStatuses={agentStatuses}
                      collaborationPhase={collaborationPhase}
                      selectedAgents={activeAgents}
                    />
                  </Box>
                )}
                
                {/* Input Area */}
                <Box sx={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
                  <MultiAgentChat
                    messages={messages}
                    setMessages={setMessages}
                    sessionId={sessionId}
                    loading={loading}
                    setLoading={setLoading}
                    setAgentStatuses={setAgentStatuses}
                    setCollaborationPhase={setCollaborationPhase}
                    agentStatuses={agentStatuses}
                    setAgentStreamingContent={setAgentStreamingContent}
                    setActiveAgents={setActiveAgents}
                  />
                </Box>
              </Paper>
            </Box>

            {/* Bottom Section: Agent Response Windows */}
            <Box sx={{ flex: 1, overflow: 'auto' }}>
              {activeAgents.length === 0 ? (
                <Paper sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Typography variant="h6" color="text.secondary">
                    Ask a question and agents will be automatically selected to help you
                  </Typography>
                </Paper>
              ) : (
                <Grid container spacing={2} sx={{ height: '100%' }}>
                  {activeAgents.map((agent) => (
                    <Grid item xs={12} md={activeAgents.length <= 2 ? 6 : activeAgents.length <= 4 ? 3 : 2.4} key={agent.id}>
                      <Paper sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                        <Box sx={{ 
                          p: 2, 
                          borderBottom: 1, 
                          borderColor: 'divider', 
                          backgroundColor: 'primary.main',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          height: '80px',
                          minHeight: '80px'
                        }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <GroupIcon sx={{ color: 'white' }} />
                            <Box>
                              <Typography variant="h6" sx={{ color: 'white', fontWeight: 'bold', lineHeight: 1.2 }}>
                                {agent.name}
                              </Typography>
                              <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.9)', lineHeight: 1 }}>
                                {agent.role}
                              </Typography>
                            </Box>
                          </Box>
                        </Box>
                        <Box sx={{ flex: 1, overflow: 'auto', p: 1 }}>
                          {/* Agent-specific response area */}
                          <AgentResponseWindow
                            agent={agent}
                            agentStatus={agentStatuses[agent.name]}
                            messages={messages}
                            streamingContent={agentStreamingContent[agent.name] || ''}
                          />
                        </Box>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              )}
            </Box>
          </Box>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default MultiAgentApp;