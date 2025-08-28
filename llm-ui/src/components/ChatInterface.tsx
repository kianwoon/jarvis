import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Button,
  Paper,
  Typography,
  IconButton,
  CircularProgress,
  Chip,
  Avatar,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tooltip
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import ClearIcon from '@mui/icons-material/Clear';
import CloseIcon from '@mui/icons-material/Close';
import DownloadIcon from '@mui/icons-material/Download';
import CopyIcon from '@mui/icons-material/ContentCopy';
import { 
  ExpandMore as ExpandMoreIcon, 
  Description as DocumentIcon,
  Search as SearchIcon,
  Public as WebIcon,
  AccessTime as TimeIcon,
  Build as ToolIcon,
  Hub as RadiatingIcon,
  Settings as SettingsIcon,
  Timeline as GraphIcon
} from '@mui/icons-material';
import TempDocumentPanel from './temp-documents/TempDocumentPanel';
import FileUploadComponent from './shared/FileUploadComponent';
import { MessageContent } from './shared/MessageContent';
import AgentAutocomplete from './AgentAutocomplete';
import RadiatingToggle from './radiating/RadiatingToggle';
import RadiatingDepthControl from './radiating/RadiatingDepthControl';
import RadiatingProgress from './radiating/RadiatingProgress';
import RadiatingResultsViewer from './radiating/RadiatingResultsViewer';
import RadiatingVisualization from './radiating/RadiatingVisualization';
import { 
  RadiatingConfig, 
  RadiatingProgress as RadiatingProgressType, 
  RadiatingResults,
  RadiatingVisualizationData
} from '../types/radiating';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  source?: string;
  metadata?: any;
  toolsUsed?: string[];
  context?: Array<{
    content: string;
    source: string;
    score?: number;
  }>;
}

interface Agent {
  name: string;
  role: string;
  description: string;
  avatar: string;
}

interface ChatInterfaceProps {
  endpoint?: string;
  title?: string;
  enableTemporaryDocuments?: boolean;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ 
  endpoint = '/api/v1/intelligent-chat/intelligent-chat',
  title = 'Jarvis Chat',
  enableTemporaryDocuments = true
}) => {
  console.log('🚀 ChatInterface component loaded with title:', title);
  
  const [messages, setMessages] = useState<Message[]>([]);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [conversationId] = useState(() => {
    const storageKey = `jarvis-conversation-id-${title.toLowerCase().replace(/\s+/g, '-')}`;
    const saved = localStorage.getItem(storageKey);
    if (saved) {
      return saved;
    }
    
    // Generate a more robust unique ID using crypto.randomUUID() if available
    let uniqueId;
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
      uniqueId = crypto.randomUUID();
    } else {
      // Fallback: enhanced timestamp with random component
      const timestamp = Date.now();
      const randomComponent = Math.random().toString(36).substring(2, 8);
      uniqueId = `${timestamp}-${randomComponent}`;
    }
    
    const newId = `chat-${title.toLowerCase().replace(/\s+/g, '-')}-${uniqueId}`;
    localStorage.setItem(storageKey, newId);
    return newId;
  });
  const [documentCount, setDocumentCount] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const storageKey = `jarvis-chat-${title.toLowerCase().replace(/\s+/g, '-')}`;

  // Agent selection state
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [showAutocomplete, setShowAutocomplete] = useState(false);
  const [autocompletePosition, setAutocompletePosition] = useState({ top: 0, left: 0 });
  const [agentSearchTerm, setAgentSearchTerm] = useState('');
  
  // Radiating Coverage state
  const [radiatingEnabled, setRadiatingEnabled] = useState(false);
  const [radiatingConfig, setRadiatingConfig] = useState<RadiatingConfig | null>(null);
  const [radiatingProgress, setRadiatingProgress] = useState<RadiatingProgressType | null>(null);
  const [radiatingResults, setRadiatingResults] = useState<RadiatingResults | null>(null);
  const [radiatingJobId, setRadiatingJobId] = useState<string | null>(null);
  const [showRadiatingSettings, setShowRadiatingSettings] = useState(false);
  const [showRadiatingVisualization, setShowRadiatingVisualization] = useState(false);

  console.log('📝 Storage key generated:', storageKey);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Load conversation from localStorage on component mount
  useEffect(() => {
    console.log('🔍 Loading conversation from localStorage with key:', storageKey);
    const savedMessages = localStorage.getItem(storageKey);
    console.log('📦 Saved messages found:', savedMessages ? 'Yes' : 'No');
    if (savedMessages) {
      try {
        const parsed = JSON.parse(savedMessages);
        console.log('✅ Parsed messages:', parsed.length, 'messages');
        setMessages(parsed.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        })));
      } catch (e) {
        console.warn('Failed to load saved conversation:', e);
      }
    }
  }, [storageKey]);

  // Save conversation to localStorage when messages change
  useEffect(() => {
    if (messages.length > 0) {
      console.log('💾 Saving', messages.length, 'messages to localStorage with key:', storageKey);
      localStorage.setItem(storageKey, JSON.stringify(messages));
    }
  }, [messages, storageKey]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Monitor input changes for agent clearing
  useEffect(() => {
    const handleInputMonitoring = () => {
      const input = inputRef.current?.value || '';
      if (selectedAgent && !input.includes(`@${selectedAgent.name}`)) {
        console.log(`🧹 CLEARING selectedAgent: ${selectedAgent.name} not found in input: "${input}"`);
        setSelectedAgent(null);
      }
    };

    const interval = setInterval(handleInputMonitoring, 500);
    return () => clearInterval(interval);
  }, [selectedAgent]);

  // Handle click outside to close autocomplete
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (showAutocomplete && inputRef.current && !inputRef.current.contains(event.target as Node)) {
        setShowAutocomplete(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showAutocomplete]);

  const sendMessage = async () => {
    const currentInput = inputRef.current?.value.trim() || '';
    if (!currentInput || loading) return;
    
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: currentInput,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    if (inputRef.current) inputRef.current.value = '';
    
    // Clear selected agent after sending message if the input no longer contains the @ mention
    if (selectedAgent && !currentInput.includes(`@${selectedAgent.name}`)) {
      console.log(`🧹 CLEARING selectedAgent after send: ${selectedAgent.name} not in message`);
      setSelectedAgent(null);
    }
    
    // Auto-detect agent mentions before sending if no agent is selected
    let effectiveSelectedAgent = selectedAgent;
    if (!selectedAgent && currentInput.includes('@')) {
      // Look for @agent_name patterns in the input
      const atMentionMatch = currentInput.match(/@([\w\s]+?)(?:\s|$)/);
      if (atMentionMatch) {
        const mentionedAgentName = atMentionMatch[1].trim();
        console.log(`🔍 Auto-detecting agent mention: "${mentionedAgentName}"`);
        
        // Fetch available agents and find a match
        try {
          const agentsResponse = await fetch('/api/v1/intelligent-chat/agents/autocomplete');
          if (agentsResponse.ok) {
            const { agents } = await agentsResponse.json();
            const matchedAgent = agents.find((agent: Agent) => 
              agent.name.toLowerCase() === mentionedAgentName.toLowerCase()
            );
            
            if (matchedAgent) {
              effectiveSelectedAgent = matchedAgent;
              console.log(`✅ Auto-selected agent: ${matchedAgent.name}`);
              setSelectedAgent(matchedAgent); // Update state for UI
            } else {
              console.log(`⚠️ Agent "${mentionedAgentName}" not found in available agents`);
            }
          }
        } catch (error) {
          console.warn('Failed to fetch agents for auto-detection:', error);
        }
      }
    }
    
    setLoading(true);

    // Add timeout protection for loading state
    const loadingTimeout = setTimeout(() => {
      console.warn('⚠️ Request taking longer than expected, but keeping loading state...');
    }, 30000); // 30 second warning

    const emergencyTimeout = setTimeout(() => {
      console.error('❌ Emergency timeout - releasing loading state after 2 minutes');
      setLoading(false);
    }, 120000); // 2 minute emergency release

    let assistantMessage: Message | undefined;

    try {
      const requestBody = {
        question: currentInput,
        conversation_id: conversationId,
        use_langgraph: false,
        // Include selected agent (either manually selected or auto-detected)
        ...(effectiveSelectedAgent && {
          selected_agent: effectiveSelectedAgent.name
        }),
        // Include temporary document preferences if enabled
        ...(enableTemporaryDocuments && {
          use_hybrid_rag: documentCount > 0,
          hybrid_strategy: "temp_priority",
          fallback_to_persistent: true,
          temp_results_weight: 0.7
        }),
        // Include radiating coverage configuration if enabled
        ...(radiatingEnabled && {
          use_radiating: true,
          radiating_config: radiatingConfig || {
            maxDepth: 3,
            strategy: 'breadth-first',
            relevanceThreshold: 0.5,
            maxEntitiesPerLevel: 20,
            includeRelationships: true
          }
        })
      };
      
      console.log('🚀 SENDING REQUEST:', requestBody);
      console.log('🤖 SELECTED AGENT STATE:', selectedAgent);
      console.log('🤖 EFFECTIVE SELECTED AGENT:', effectiveSelectedAgent);
      console.log('🔍 INPUT TEXT:', currentInput);
      if (effectiveSelectedAgent) {
        console.log(`✅ Agent Selected: ${effectiveSelectedAgent.name} - Should be included in request`);
        console.log(`🎯 Agent details:`, effectiveSelectedAgent);
        if (effectiveSelectedAgent !== selectedAgent) {
          console.log(`🔧 Agent was auto-detected from @mention`);
        }
      } else {
        console.log('❌ No agent selected and no valid @mention found');
      }
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      assistantMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: '',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = new TextDecoder().decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (!line.trim()) continue;
          
          try {
            const data = JSON.parse(line);
            
            // Handle radiating-specific events
            if (data.type === 'radiating_start' && data.job_id) {
              setRadiatingJobId(data.job_id);
              setRadiatingProgress({
                isActive: true,
                currentDepth: 0,
                totalDepth: data.max_depth || 3,
                entitiesDiscovered: 0,
                relationshipsFound: 0,
                processedEntities: 0,
                queueSize: 0,
                elapsedTime: 0,
                status: 'initializing'
              });
            } else if (data.type === 'radiating_progress' && data.progress) {
              setRadiatingProgress(data.progress);
            } else if (data.type === 'radiating_complete' && data.results) {
              setRadiatingResults(data.results);
              setRadiatingProgress({
                ...data.progress,
                isActive: false,
                status: 'completed'
              });
            } else if (data.type === 'status' && data.message) {
              setStatusMessage(data.message);
              // Clear status message after 3 seconds unless it's replaced
              setTimeout(() => {
                setStatusMessage(prev => prev === data.message ? '' : prev);
              }, 3000);
            } else if (data.type === 'classification' && data.routing) {
              // Show classification info briefly
              const classificationMsg = `Query classified as: ${data.routing.primary_type} (${(data.routing.confidence * 100).toFixed(0)}% confidence)`;
              setStatusMessage(classificationMsg);
              setTimeout(() => {
                setStatusMessage(prev => prev === classificationMsg ? '' : prev);
              }, 2000);
            } else if (data.token) {
              // Clear status message when actual content starts
              setStatusMessage('');
              assistantMessage.content += data.token;
              if (data.source) assistantMessage.source = data.source;
              if (data.metadata) {
                assistantMessage.metadata = data.metadata;
                // Extract tools used from metadata
                if (data.metadata.tools_executed) {
                  assistantMessage.toolsUsed = data.metadata.tools_executed;
                }
              }
              setMessages(prev => 
                prev.map(msg => 
                  msg.id === assistantMessage.id ? assistantMessage : msg
                )
              );
            } else if (data.answer) {
              assistantMessage.content = data.answer;
              assistantMessage.source = data.source;
              assistantMessage.metadata = data.metadata;
              
              // Extract tools used from metadata
              if (data.metadata && data.metadata.tools_executed) {
                assistantMessage.toolsUsed = data.metadata.tools_executed;
              }
              
              if (data.context_documents || data.retrieved_docs || data.documents) {
                assistantMessage.context = (data.context_documents || data.retrieved_docs || data.documents).map((doc: any) => ({
                  content: doc.content || doc.text || '',
                  source: doc.source || doc.metadata?.source || 'Unknown',
                  score: doc.relevance_score || doc.score
                }));
              }
              setMessages(prev => 
                prev.map(msg => 
                  msg.id === assistantMessage.id ? assistantMessage : msg
                )
              );
              break;
            }
          } catch (e) {
            // Skip invalid JSON lines
          }
        }
      }
    } catch (error) {
      console.error('Chat request failed:', error);
      
      // Check if we have a partial assistant message to preserve
      const existingAssistantMessage = messages.find(msg => msg.id === assistantMessage?.id);
      
      if (existingAssistantMessage && existingAssistantMessage.content.trim()) {
        // Update existing message with error indicator but preserve content
        setMessages(prev => 
          prev.map(msg => 
            msg.id === existingAssistantMessage.id 
              ? { 
                  ...msg, 
                  content: msg.content + '\n\n⚠️ *Response interrupted due to connection error*',
                  source: 'PARTIAL_RESPONSE'
                }
              : msg
          )
        );
      } else {
        // No partial content, show full error message
        const errorMessage: Message = {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: `❌ **Connection Error**\n\n${error instanceof Error ? error.message : 'Unknown error occurred'}\n\nPlease try your request again.`,
          timestamp: new Date(),
          source: 'ERROR'
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } finally {
      // Clear the timeout protection
      clearTimeout(loadingTimeout);
      clearTimeout(emergencyTimeout);
      setLoading(false);
    }
  };

  const clearChat = async () => {
    try {
      console.log('🧹 Starting chat clear...', { conversationId });
      
      // Clear conversation history from Redis if conversation ID exists
      if (conversationId && conversationId.trim()) {
        try {
          const response = await fetch(`/api/v1/langchain/conversation/${conversationId}`, {
            method: 'DELETE',
            headers: {
              'Content-Type': 'application/json'
            }
          });
          
          if (response.ok) {
            console.log('✅ Redis conversation history cleared');
          } else {
            console.warn('⚠️ Redis clear failed, but continuing with frontend clear');
          }
        } catch (apiError) {
          console.error('❌ API call failed:', apiError);
          // Continue with frontend clearing even if API fails
        }
      }
      
      // Clear frontend state and localStorage
      setMessages([]);
      localStorage.removeItem(storageKey);
      // Note: We don't remove conversation ID as it's persistent for the session
      
      console.log('✅ Chat cleared successfully');
      
    } catch (error) {
      console.error('❌ Error during chat clear:', error);
      // Fallback: Still clear frontend even if something fails
      setMessages([]);
      localStorage.removeItem(storageKey);
    }
  };

  // Handle input changes to detect @ mentions
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const input = e.target.value;
    const cursorPosition = e.target.selectionStart;
    
    console.log('Input changed:', input, 'Cursor at:', cursorPosition);
    
    // Find the last @ character before cursor position
    const textBeforeCursor = input.substring(0, cursorPosition);
    const lastAtIndex = textBeforeCursor.lastIndexOf('@');
    
    console.log('Last @ index:', lastAtIndex, 'Text before cursor:', textBeforeCursor);
    
    if (lastAtIndex !== -1) {
      // Check if there's a space after the @ but before cursor
      const textAfterAt = textBeforeCursor.substring(lastAtIndex + 1);
      
      // Only show autocomplete if there's no space after @ and we're right after @ or typing the agent name
      if (!textAfterAt.includes(' ') && textAfterAt.length <= 50) {
        // Calculate position for autocomplete dropdown
        const textArea = e.target;
        const rect = textArea.getBoundingClientRect();
        
        // Simple approximation of cursor position - in real implementation you'd want more precise positioning
        const lineHeight = parseInt(getComputedStyle(textArea).lineHeight, 10) || 20;
        const lines = textBeforeCursor.split('\n').length;
        
        setAutocompletePosition({
          top: rect.top + (lines * lineHeight) + 25,
          left: rect.left + 50
        });
        
        setAgentSearchTerm(textAfterAt);
        setShowAutocomplete(true);
        console.log('Showing autocomplete with search term:', textAfterAt);
        console.log('Position:', { top: rect.top + (lines * lineHeight) + 25, left: rect.left + 50 });
        console.log('State will be - showAutocomplete: true, agentSearchTerm:', textAfterAt);
        return;
      }
    }
    
    // Hide autocomplete if conditions aren't met
    setShowAutocomplete(false);
  };

  // Handle agent selection from autocomplete
  const handleAgentSelect = (agent: Agent) => {
    console.log('handleAgentSelect called with agent:', agent);
    if (!inputRef.current) {
      console.log('No input ref found');
      return;
    }
    
    try {
      const input = inputRef.current.value;
      const cursorPosition = inputRef.current.selectionStart || 0;
      const textBeforeCursor = input.substring(0, cursorPosition);
      const lastAtIndex = textBeforeCursor.lastIndexOf('@');
      
      console.log('Current input:', input);
      console.log('Cursor position:', cursorPosition);
      console.log('Text before cursor:', textBeforeCursor);
      console.log('Last @ index:', lastAtIndex);
      
      if (lastAtIndex !== -1) {
        // Replace @searchterm with @agent_name
        const beforeAt = input.substring(0, lastAtIndex);
        const afterCursor = input.substring(cursorPosition);
        const newValue = `${beforeAt}@${agent.name} ${afterCursor}`;
        
        console.log('New value:', newValue);
        inputRef.current.value = newValue;
        
        // Set cursor position after the agent name
        const newCursorPos = lastAtIndex + agent.name.length + 2;
        console.log('Setting cursor to position:', newCursorPos);
        inputRef.current.setSelectionRange(newCursorPos, newCursorPos);
        
        // Trigger onChange to update any state if needed
        const event = new Event('input', { bubbles: true });
        inputRef.current.dispatchEvent(event);
      }
      
      setSelectedAgent(agent);
      setShowAutocomplete(false);
      
      console.log('🎯 AGENT SELECTED SUCCESSFULLY:', agent.name);
      console.log('🤖 Full agent data:', agent);
      console.log('✅ selectedAgent state should now be set');
      
      // Focus input after a small delay to ensure autocomplete closes first
      setTimeout(() => {
        if (inputRef.current) {
          inputRef.current.focus();
        }
      }, 50);
    } catch (error) {
      console.error('Error in handleAgentSelect:', error);
      setShowAutocomplete(false);
    }
  };

  // Handle closing autocomplete
  const handleAutocompleteClose = () => {
    setShowAutocomplete(false);
  };

  // Clear selected agent when input is cleared or @ mention is removed
  const handleInputClear = () => {
    const input = inputRef.current?.value || '';
    if (selectedAgent && !input.includes(`@${selectedAgent.name}`)) {
      setSelectedAgent(null);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    // Don't handle Enter if autocomplete is visible - let autocomplete handle it
    if (showAutocomplete && (e.key === 'Enter' || e.key === 'ArrowUp' || e.key === 'ArrowDown' || e.key === 'Escape')) {
      return;
    }
    
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Convert single message to Markdown format
  const convertMessageToMarkdown = (message: Message): string => {
    const timestamp = message.timestamp.toLocaleString();
    const role = message.role === 'user' ? '**You**' : '**Assistant**';
    
    let content = `# ${role} Response - ${timestamp}\n\n${message.content}\n\n`;
    
    // Add tools used if available
    if (message.toolsUsed && message.toolsUsed.length > 0) {
      content += `### Tools Used\n\n`;
      message.toolsUsed.forEach((tool, idx) => {
        content += `- ${tool}\n`;
      });
      content += '\n';
    }
    
    // Add context sources if available
    if (message.context && message.context.length > 0) {
      content += `### Sources (${message.context.length})\n\n`;
      message.context.forEach((doc: any, idx: number) => {
        content += `${idx + 1}. **${doc.source}** ${doc.score ? `(${(doc.score * 100).toFixed(1)}% match)` : ''}\n`;
        content += `   ${doc.content.substring(0, 200)}${doc.content.length > 200 ? '...' : ''}\n\n`;
      });
    }
    
    return content;
  };

  // Convert messages to Markdown format (kept for compatibility)
  const convertToMarkdown = (): string => {
    const header = `# Chat Conversation: ${title}\n\n*Downloaded on ${new Date().toLocaleString()}*\n\n---\n\n`;
    
    const conversationMarkdown = messages.map((message, index) => {
      const timestamp = message.timestamp.toLocaleString();
      const role = message.role === 'user' ? '**You**' : '**Assistant**';
      
      let content = `## ${role} - ${timestamp}\n\n${message.content}\n\n`;
      
      // Add tools used if available
      if (message.toolsUsed && message.toolsUsed.length > 0) {
        content += `### Tools Used\n\n`;
        message.toolsUsed.forEach((tool, idx) => {
          content += `- ${tool}\n`;
        });
        content += '\n';
      }
      
      // Add context sources if available
      if (message.context && message.context.length > 0) {
        content += `### Sources (${message.context.length})\n\n`;
        message.context.forEach((doc: any, idx: number) => {
          content += `${idx + 1}. **${doc.source}** ${doc.score ? `(${(doc.score * 100).toFixed(1)}% match)` : ''}\n`;
          content += `   ${doc.content.substring(0, 200)}${doc.content.length > 200 ? '...' : ''}\n\n`;
        });
      }
      
      content += '---\n\n';
      return content;
    }).join('');
    
    return header + conversationMarkdown;
  };

  // Download single message as Markdown file
  const downloadMessage = (message: Message) => {
    const markdown = convertMessageToMarkdown(message);
    const blob = new Blob([markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('T')[0];
    const filename = `jarvis-response-${message.id}-${timestamp}.md`;
    
    link.href = url;
    link.download = filename;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Download conversation as Markdown file (kept for compatibility)
  const downloadConversation = () => {
    if (messages.length === 0) return;
    
    const markdown = convertToMarkdown();
    const blob = new Blob([markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('T')[0];
    const filename = `chat-${title.replace(/[^a-zA-Z0-9]/g, '-')}-${timestamp}.md`;
    
    link.href = url;
    link.download = filename;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Copy single message to clipboard
  const copyMessageToClipboard = async (message: Message) => {
    try {
      const markdown = convertMessageToMarkdown(message);
      await navigator.clipboard.writeText(markdown);
      
      // Show success feedback
      setStatusMessage('Response copied to clipboard!');
      setTimeout(() => setStatusMessage(''), 2000);
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
      setStatusMessage('Failed to copy response to clipboard');
    }
  };

  // Copy conversation to clipboard (kept for compatibility)
  const copyToClipboard = async () => {
    if (messages.length === 0) return;
    
    try {
      const markdown = convertToMarkdown();
      await navigator.clipboard.writeText(markdown);
      
      // Show success feedback
      setStatusMessage('Conversation copied to clipboard!');
      setTimeout(() => setStatusMessage(''), 2000);
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
      setStatusMessage('Failed to copy conversation to clipboard');
    }
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', p: 2 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h5">{title}</Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {/* Radiating Coverage Toggle */}
          <RadiatingToggle
            conversationId={conversationId}
            onToggle={(enabled) => {
              setRadiatingEnabled(enabled);
              if (!enabled) {
                setRadiatingProgress(null);
                setRadiatingResults(null);
              }
            }}
            disabled={loading}
            size="medium"
            showLabel={true}
            showStatus={false}
          />
          
          {/* Radiating Settings Button */}
          {radiatingEnabled && (
            <IconButton 
              onClick={() => setShowRadiatingSettings(!showRadiatingSettings)}
              color={showRadiatingSettings ? 'primary' : 'default'}
            >
              <SettingsIcon />
            </IconButton>
          )}
          
          {/* Radiating Visualization Button */}
          {radiatingResults && (
            <IconButton 
              onClick={() => setShowRadiatingVisualization(true)}
              color="primary"
            >
              <GraphIcon />
            </IconButton>
          )}
          
          
          <Tooltip title="Clear conversation">
            <IconButton 
              onClick={clearChat} 
              disabled={loading}
              color="default"
              aria-label="Clear conversation"
            >
              <ClearIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Temporary Documents Panel */}
      {enableTemporaryDocuments && (
        <TempDocumentPanel
          conversationId={conversationId}
          disabled={loading}
          variant="compact"
          defaultExpanded={false}
          onDocumentChange={(activeCount) => {
            setDocumentCount(activeCount);
          }}
        />
      )}
      
      {/* Radiating Settings Panel */}
      {radiatingEnabled && showRadiatingSettings && (
        <Box sx={{ mb: 2 }}>
          <RadiatingDepthControl
            conversationId={conversationId}
            onConfigChange={(config) => setRadiatingConfig(config)}
            disabled={loading}
            compact={true}
          />
        </Box>
      )}
      
      {/* Radiating Progress */}
      {radiatingProgress && radiatingProgress.isActive && (
        <Box sx={{ mb: 2 }}>
          <RadiatingProgress
            jobId={radiatingJobId || undefined}
            progress={radiatingProgress}
            onCancel={() => {
              setRadiatingProgress(null);
              setRadiatingJobId(null);
            }}
            compact={true}
            showDetails={false}
          />
        </Box>
      )}
      
      {/* Radiating Results Summary */}
      {radiatingResults && !radiatingProgress?.isActive && (
        <Box sx={{ mb: 2 }}>
          <RadiatingResultsViewer
            results={radiatingResults}
            onEntityClick={(entity) => {
              console.log('Entity clicked:', entity);
            }}
            onExploreEntity={(entity) => {
              // Could trigger a new search based on the entity
              if (inputRef.current) {
                inputRef.current.value = `Tell me more about ${entity.name}`;
                sendMessage();
              }
            }}
            compact={true}
            maxHeight={200}
          />
        </Box>
      )}

      {/* Status Message */}
      {statusMessage && (
        <Alert 
          severity="info" 
          sx={{ 
            mb: 1, 
            backgroundColor: 'action.hover',
            '& .MuiAlert-icon': {
              color: 'primary.main'
            }
          }}
        >
          {statusMessage}
        </Alert>
      )}

      {/* Selected Agent Badge */}
      {selectedAgent && (
        <Box sx={{ mb: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="body2" color="text.secondary">
              Selected Agent:
            </Typography>
            <Chip
              avatar={
                <Avatar sx={{ bgcolor: 'primary.main', fontSize: '0.875rem' }}>
                  {selectedAgent.avatar}
                </Avatar>
              }
              label={selectedAgent.name}
              variant="filled"
              color="primary"
              onDelete={() => setSelectedAgent(null)}
              deleteIcon={<CloseIcon />}
              sx={{ fontSize: '0.875rem' }}
            />
          </Box>
        </Box>
      )}

      {/* Messages */}
      <Paper sx={{ flex: 1, p: 2, overflow: 'auto', mb: 2 }}>
        {messages.length === 0 && (
          <Alert severity="info">
            Welcome to Jarvis! Ask me anything about your documents or use my tools.
          </Alert>
        )}
        
        {messages.map((message) => (
          <Box
            key={message.id}
            sx={{
              mb: 2,
              display: 'flex',
              flexDirection: 'column',
              alignItems: message.role === 'user' ? 'flex-end' : 'flex-start',
              width: '100%'  // Ensure parent takes full width
            }}
          >
            <Box
              sx={{
                width: message.role === 'user' ? '80%' : '95%',  // Use width instead of maxWidth for assistant
                maxWidth: message.role === 'user' ? '80%' : '100%',  // Still set maxWidth for user messages
                p: 2,
                borderRadius: 2,
                backgroundColor: message.role === 'user' ? 'primary.light' : 'background.paper',
                color: message.role === 'user' ? 'white' : 'text.primary',
                position: 'relative'
              }}
            >
              {/* Copy/Download buttons for assistant messages only */}
              {message.role === 'assistant' ? (
                <Box
                  sx={{
                    position: 'absolute',
                    top: 8,
                    right: 8,
                    display: 'flex',
                    gap: 0.5,
                    opacity: 0.7,
                    '&:hover': { opacity: 1 },
                    zIndex: 1
                  }}
                >
                  <Tooltip title="Copy this response to clipboard">
                    <IconButton
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        copyMessageToClipboard(message);
                      }}
                      sx={{
                        backgroundColor: 'rgba(0,0,0,0.1)',
                        color: 'text.secondary',
                        '&:hover': {
                          backgroundColor: 'rgba(0,0,0,0.2)'
                        },
                        width: 24,
                        height: 24
                      }}
                      aria-label="Copy response to clipboard"
                    >
                      <CopyIcon sx={{ fontSize: 14 }} />
                    </IconButton>
                  </Tooltip>
                  
                  <Tooltip title="Download this response as Markdown">
                    <IconButton
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        downloadMessage(message);
                      }}
                      sx={{
                        backgroundColor: 'rgba(0,0,0,0.1)',
                        color: 'text.secondary',
                        '&:hover': {
                          backgroundColor: 'rgba(0,0,0,0.2)'
                        },
                        width: 24,
                        height: 24
                      }}
                      aria-label="Download response as Markdown file"
                    >
                      <DownloadIcon sx={{ fontSize: 14 }} />
                    </IconButton>
                  </Tooltip>
                </Box>
              ) : null}
              
              <MessageContent content={message.content} />
              
              {message.source && (
                <Box sx={{ mt: 1 }}>
                  <Chip label={message.source} size="small" variant="outlined" />
                </Box>
              )}

              {/* Tools used */}
              {message.toolsUsed && message.toolsUsed.length > 0 && (
                <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {message.toolsUsed.map((tool, index) => {
                    // Get appropriate icon for each tool
                    const getToolIcon = (toolName: string) => {
                      if (toolName.includes('google_search') || toolName.includes('search')) {
                        return <WebIcon fontSize="small" />;
                      } else if (toolName.includes('rag_knowledge_search') || toolName.includes('knowledge')) {
                        return <SearchIcon fontSize="small" />;
                      } else if (toolName.includes('datetime') || toolName.includes('time')) {
                        return <TimeIcon fontSize="small" />;
                      } else {
                        return <ToolIcon fontSize="small" />;
                      }
                    };

                    // Clean tool name for display
                    const getToolDisplayName = (toolName: string) => {
                      if (toolName.includes('google_search')) return 'Web Search';
                      if (toolName.includes('rag_knowledge_search')) return 'Knowledge Base';
                      if (toolName.includes('get_datetime')) return 'Current Time';
                      return toolName.replace(/_/g, ' ').replace(/([A-Z])/g, ' $1').trim();
                    };

                    return (
                      <Chip
                        key={index}
                        icon={getToolIcon(tool)}
                        label={getToolDisplayName(tool)}
                        size="small"
                        variant="outlined"
                        sx={{
                          fontSize: '0.75rem',
                          height: '24px',
                          backgroundColor: message.role === 'user' ? 'rgba(255,255,255,0.1)' : 'action.hover',
                          color: message.role === 'user' ? 'inherit' : 'text.secondary'
                        }}
                      />
                    );
                  })}
                </Box>
              )}
            </Box>
            
            <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
              {message.timestamp.toLocaleTimeString()}
            </Typography>
            
            {/* Context documents */}
            {message.context && message.context.length > 0 && (
              <Box sx={{ mt: 1, width: '100%', alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start' }}>
                <Accordion sx={{ backgroundColor: 'background.default' }}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <DocumentIcon fontSize="small" />
                      <Typography variant="body2">
                        {message.context.length} source document{message.context.length > 1 ? 's' : ''}
                      </Typography>
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <List dense>
                      {message.context.map((doc: any, index: number) => (
                        <ListItem key={index} sx={{ px: 0 }}>
                          <ListItemIcon>
                            <DocumentIcon fontSize="small" />
                          </ListItemIcon>
                          <ListItemText
                            primary={doc.source || `Document ${index + 1}`}
                            secondary={
                              <Typography variant="caption" sx={{ 
                                display: '-webkit-box',
                                WebkitLineClamp: 2,
                                WebkitBoxOrient: 'vertical',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis'
                              }}>
                                {doc.content}
                              </Typography>
                            }
                          />
                          {doc.score && (
                            <Chip 
                              label={`${(doc.score * 100).toFixed(1)}%`}
                              size="small"
                              variant="outlined"
                              sx={{ ml: 1 }}
                            />
                          )}
                        </ListItem>
                      ))}
                    </List>
                  </AccordionDetails>
                </Accordion>
              </Box>
            )}
          </Box>
        ))}
        
        {loading && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 2 }}>
            <CircularProgress size={20} />
            <Typography variant="body2" color="text.secondary">
              Jarvis is thinking...
            </Typography>
          </Box>
        )}
        
        <div ref={messagesEndRef} />
      </Paper>

      {/* Input */}
      <Box sx={{ display: 'flex', gap: 1, position: 'relative' }}>
        <Box
          component="textarea"
          ref={inputRef}
          onChange={handleInputChange}
          onKeyDown={handleKeyPress}
          placeholder="Ask Jarvis anything... Type @ to mention an agent"
          disabled={loading}
          sx={{
            flex: 1,
            minHeight: '56px',
            maxHeight: '120px',
            padding: '16px',
            border: 1,
            borderColor: 'divider',
            borderRadius: 1,
            fontSize: '16px',
            fontFamily: 'inherit',
            resize: 'vertical',
            outline: 'none',
            backgroundColor: 'background.paper',
            color: 'text.primary',
            '&:focus': {
              borderColor: 'primary.main'
            },
            '&:disabled': {
              backgroundColor: 'action.disabled',
              color: 'text.disabled'
            }
          }}
        />
        <FileUploadComponent
          onUploadSuccess={(result) => {
            console.log('Upload success:', result);
          }}
          disabled={loading}
          autoClassify={true}
        />
        <Button
          variant="contained"
          onClick={sendMessage}
          disabled={loading}
          sx={{ minWidth: 60 }}
        >
          {loading ? <CircularProgress size={24} /> : <SendIcon />}
        </Button>

        {/* Agent Autocomplete */}
        {showAutocomplete && (
          <AgentAutocomplete
            searchTerm={agentSearchTerm}
            position={autocompletePosition}
            onSelect={handleAgentSelect}
            onClose={handleAutocompleteClose}
            visible={showAutocomplete}
          />
        )}
      </Box>
      
      {/* Radiating Visualization Dialog */}
      <Dialog
        open={showRadiatingVisualization}
        onClose={() => setShowRadiatingVisualization(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="h6">Radiating Coverage Visualization</Typography>
            <IconButton onClick={() => setShowRadiatingVisualization(false)}>
              <CloseIcon />
            </IconButton>
          </Box>
        </DialogTitle>
        
        <DialogContent>
          {radiatingResults && (
            <RadiatingVisualization
              data={{
                nodes: radiatingResults.entities.map(entity => ({
                  id: entity.id,
                  name: entity.name,
                  type: entity.type,
                  group: entity.depth,
                  radius: entity.relevanceScore,
                  color: ''
                })),
                links: radiatingResults.relationships.map(rel => ({
                  source: rel.sourceId,
                  target: rel.targetId,
                  value: rel.weight,
                  type: rel.type
                }))
              }}
              width={900}
              height={600}
              onNodeClick={(node) => {
                console.log('Node clicked in visualization:', node);
              }}
            />
          )}
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setShowRadiatingVisualization(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ChatInterface;