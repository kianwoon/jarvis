import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Box,
  Paper,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Avatar,
  Typography,
  CircularProgress,
  Chip
} from '@mui/material';

interface Agent {
  name: string;
  role: string;
  description: string;
  avatar: string;
}

interface AgentAutocompleteProps {
  searchTerm: string;
  position: { top: number; left: number };
  onSelect: (agent: Agent) => void;
  onClose: () => void;
  visible: boolean;
}

// Helper function to get emoji avatar based on agent role
const getAgentAvatar = (role: string): string => {
  const roleAvatars: Record<string, string> = {
    'document_researcher': '🔍',
    'sales': '💼',
    'technical': '💻',
    'financial': '💰',
    'planning': '📋',
    'security': '🔒',
    'data': '📊',
    'communication': '💬',
    'research': '🔬',
    'analyst': '📈',
    'ceo': '👔',
    'cfo': '💵',
    'cto': '🖥️',
    'engineer': '🔧',
    'manager': '📌'
  };
  
  const roleLower = role.toLowerCase();
  // Check for partial matches
  for (const [key, avatar] of Object.entries(roleAvatars)) {
    if (roleLower.includes(key)) {
      return avatar;
    }
  }
  return '🤖'; // Default avatar
};

const AgentAutocomplete: React.FC<AgentAutocompleteProps> = ({
  searchTerm,
  position,
  onSelect,
  onClose,
  visible
}) => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [filteredAgents, setFilteredAgents] = useState<Agent[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);

  // Fetch agents from the API
  const fetchAgents = useCallback(async () => {
    if (!visible) return;
    
    console.log('AgentAutocomplete: Fetching agents...');
    setLoading(true);
    try {
      const response = await fetch('/api/v1/intelligent-chat/agents/autocomplete');
      
      if (!response.ok) {
        console.warn('Failed to fetch agents for autocomplete');
        return;
      }

      const data = await response.json();
      // Transform the data to match our Agent interface
      const agentList = data.agents || data || [];
      const transformedAgents = (Array.isArray(agentList) ? agentList : []).map((agent: any) => ({
        name: agent.name,
        role: agent.role,
        description: agent.description || agent.role,
        avatar: agent.avatar || getAgentAvatar(agent.role)
      }));
      console.log('AgentAutocomplete: Fetched', transformedAgents.length, 'agents');
      setAgents(transformedAgents);
    } catch (error) {
      console.error('Error fetching agents:', error);
      setAgents([]);
    } finally {
      setLoading(false);
    }
  }, [visible]);

  // Filter agents based on search term
  useEffect(() => {
    if (!searchTerm.trim()) {
      setFilteredAgents(agents);
    } else {
      const filtered = agents.filter(agent =>
        agent.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        agent.role.toLowerCase().includes(searchTerm.toLowerCase()) ||
        agent.description.toLowerCase().includes(searchTerm.toLowerCase())
      );
      setFilteredAgents(filtered);
    }
    setSelectedIndex(0);
  }, [searchTerm, agents]);

  // Fetch agents when component becomes visible
  useEffect(() => {
    if (visible) {
      fetchAgents();
    }
  }, [visible, fetchAgents]);

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!visible || filteredAgents.length === 0) return;

      console.log('AgentAutocomplete keydown:', e.key, 'visible:', visible, 'agents:', filteredAgents.length);
      
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          e.stopPropagation();
          console.log('Arrow down pressed, current index:', selectedIndex);
          setSelectedIndex(prev => {
            const newIndex = prev < filteredAgents.length - 1 ? prev + 1 : 0;
            console.log('New index:', newIndex);
            return newIndex;
          });
          break;
        case 'ArrowUp':
          e.preventDefault();
          e.stopPropagation();
          console.log('Arrow up pressed, current index:', selectedIndex);
          setSelectedIndex(prev => {
            const newIndex = prev > 0 ? prev - 1 : filteredAgents.length - 1;
            console.log('New index:', newIndex);
            return newIndex;
          });
          break;
        case 'Enter':
          e.preventDefault();
          e.stopPropagation();
          console.log('Enter pressed, selecting agent at index:', selectedIndex);
          if (filteredAgents[selectedIndex]) {
            console.log('Selecting agent:', filteredAgents[selectedIndex].name);
            onSelect(filteredAgents[selectedIndex]);
          }
          break;
        case 'Escape':
          e.preventDefault();
          e.stopPropagation();
          console.log('Escape pressed, closing autocomplete');
          onClose();
          break;
      }
    };

    // Use capture phase to intercept before textarea
    if (visible) {
      document.addEventListener('keydown', handleKeyDown, true);
    }
    
    return () => {
      document.removeEventListener('keydown', handleKeyDown, true);
    };
  }, [visible, filteredAgents, selectedIndex, onSelect, onClose]);

  // Scroll selected item into view
  useEffect(() => {
    if (listRef.current && filteredAgents.length > 0) {
      const selectedElement = listRef.current.children[selectedIndex] as HTMLElement;
      if (selectedElement) {
        selectedElement.scrollIntoView({
          block: 'nearest',
          behavior: 'smooth'
        });
      }
    }
  }, [selectedIndex, filteredAgents]);

  if (!visible) {
    return null;
  }

  return (
    <Paper
      sx={{
        position: 'fixed',
        top: Math.max(position.top - 350, 10), // Show above input, with minimum distance from top
        left: position.left,
        zIndex: 9999,
        maxWidth: 400,
        minWidth: 300,
        maxHeight: 300,
        overflow: 'hidden',
        boxShadow: 3,
        border: 2,
        borderColor: 'primary.main',
        backgroundColor: 'background.paper',
        pointerEvents: 'auto'  // Ensure clicks are not blocked
      }}
      elevation={8}
      onMouseDown={(e) => {
        e.preventDefault();
        e.stopPropagation();
        console.log('Paper mouseDown prevented');
      }}
      onClick={(e) => {
        e.stopPropagation();  // Prevent event bubbling
        console.log('Paper clicked');
      }}
    >
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
          <CircularProgress size={20} />
          <Typography variant="body2" sx={{ ml: 1 }}>
            Loading agents...
          </Typography>
        </Box>
      ) : filteredAgents.length > 0 ? (
        <List
          ref={listRef}
          sx={{
            py: 0,
            maxHeight: 300,
            overflow: 'auto'
          }}
        >
          {filteredAgents.map((agent, index) => (
            <ListItemButton
              key={agent.name}
              selected={index === selectedIndex}
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                console.log('Agent clicked:', agent.name);
                // Use setTimeout to ensure the event is handled after any blur events
                setTimeout(() => onSelect(agent), 0);
              }}
              onMouseDown={(e) => {
                e.preventDefault();
                e.stopPropagation();
                console.log('Agent mouse down:', agent.name);
              }}
              onMouseEnter={() => setSelectedIndex(index)}
              sx={{ 
                py: 1,
                cursor: 'pointer',
                '&:hover': {
                  backgroundColor: 'action.hover'
                }
              }}
            >
              <ListItemIcon>
                <Avatar
                  sx={{
                    width: 32,
                    height: 32,
                    fontSize: '1rem',
                    bgcolor: 'primary.main'
                  }}
                >
                  {agent.avatar}
                </Avatar>
              </ListItemIcon>
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="subtitle2" component="span">
                      {agent.name}
                    </Typography>
                    <Chip
                      label={agent.role}
                      size="small"
                      variant="outlined"
                      sx={{ fontSize: '0.75rem', height: 20 }}
                    />
                  </Box>
                }
                secondary={
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{
                      display: '-webkit-box',
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: 'vertical',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis'
                    }}
                  >
                    {agent.description}
                  </Typography>
                }
              />
            </ListItemButton>
          ))}
        </List>
      ) : (
        <Box sx={{ p: 2, textAlign: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            {searchTerm.trim() 
              ? `No agents found matching "${searchTerm}"`
              : 'No agents available'
            }
          </Typography>
        </Box>
      )}
      
      {/* Helper text */}
      {filteredAgents.length > 0 && (
        <Box
          sx={{
            px: 2,
            py: 1,
            borderTop: 1,
            borderColor: 'divider',
            backgroundColor: 'background.default'
          }}
        >
          <Typography variant="caption" color="text.secondary">
            Use ↑↓ to navigate, Enter to select, Esc to close
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default AgentAutocomplete;