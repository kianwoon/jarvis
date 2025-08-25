import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Button,
  Paper,
  Typography,
  IconButton,
  CircularProgress,
  Chip,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider
} from '@mui/material';
import {
  Send as SendIcon,
  Clear as ClearIcon,
  ExpandMore as ExpandMoreIcon,
  Description as DocumentIcon,
  Search as SearchIcon,
  Chat as ChatIcon,
  Source as SourceIcon
} from '@mui/icons-material';
import { MessageContent } from '../shared/MessageContent';
import { 
  notebookAPI, 
  NotebookWithDocuments, 
  NotebookChatMessage, 
  getErrorMessage
} from './NotebookAPI';

interface NotebookChatProps {
  notebook: NotebookWithDocuments;
  onDocumentChange?: () => void;
}

const NotebookChat: React.FC<NotebookChatProps> = ({ 
  notebook,
  onDocumentChange 
}) => {
  const [messages, setMessages] = useState<NotebookChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [statusMessage, setStatusMessage] = useState<string>('');
  
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  
  // Conversation ID for the notebook chat
  const conversationId = `notebook-${notebook.id}`;
  const storageKey = `jarvis-notebook-chat-${notebook.id}`;

  const scrollToBottom = (force = false) => {
    if (!messagesContainerRef.current) return;
    
    // Check if user is near the bottom (within 100px) or force scroll
    const container = messagesContainerRef.current;
    const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 100;
    
    if (force || isNearBottom) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  };

  // Load conversation from localStorage on component mount
  useEffect(() => {
    const savedMessages = localStorage.getItem(storageKey);
    if (savedMessages) {
      try {
        const parsed = JSON.parse(savedMessages);
        setMessages(parsed.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        })));
      } catch (e) {
        console.warn('Failed to load saved notebook conversation:', e);
      }
    }
  }, [storageKey]);

  // Save conversation to localStorage when messages change
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem(storageKey, JSON.stringify(messages));
    }
  }, [messages, storageKey]);

  useEffect(() => {
    // Auto-scroll only for new assistant messages or when explicitly needed
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      const isNewUserMessage = lastMessage.role === 'user';
      scrollToBottom(isNewUserMessage);
    }
  }, [messages]);

  const sendMessage = async () => {
    const currentInput = inputRef.current?.value.trim() || '';
    if (!currentInput || loading) return;
    
    const userMessage: NotebookChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: currentInput,
      timestamp: new Date(),
      notebook_id: notebook.id
    };

    setMessages(prev => [...prev, userMessage]);
    if (inputRef.current) inputRef.current.value = '';
    
    setLoading(true);
    setError('');

    // Add timeout protection for loading state
    const loadingTimeout = setTimeout(() => {
      console.warn('⚠️ Notebook chat request taking longer than expected...');
    }, 30000);

    const emergencyTimeout = setTimeout(() => {
      console.error('❌ Emergency timeout - releasing loading state after 2 minutes');
      setLoading(false);
    }, 120000);

    let assistantMessage: NotebookChatMessage | undefined;

    try {
      const response = await notebookAPI.startNotebookChat(notebook.id, {
        message: currentInput,
        conversation_id: conversationId,
        include_context: true
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
        timestamp: new Date(),
        notebook_id: notebook.id
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
            
            // Debug: Log all response data to understand format
            console.log('📨 Received response data:', data);
            
            if (data.type === 'status' && data.message) {
              setStatusMessage(data.message);
              // Clear status message after 3 seconds unless it's replaced
              setTimeout(() => {
                setStatusMessage(prev => prev === data.message ? '' : prev);
              }, 3000);
            } else if (data.type === 'classification' && data.routing) {
              // Show classification info briefly
              const classificationMsg = `Searching notebook documents (${(data.routing.confidence * 100).toFixed(0)}% confidence)`;
              setStatusMessage(classificationMsg);
              setTimeout(() => {
                setStatusMessage(prev => prev === classificationMsg ? '' : prev);
              }, 2000);
            } else if (data.token || data.chunk) {
              // Clear status message when actual content starts
              setStatusMessage('');
              const tokenText = data.token || data.chunk;
              assistantMessage.content += tokenText;
              if (data.metadata) {
                assistantMessage.metadata = data.metadata;
              }
              setMessages(prev => 
                prev.map(msg => 
                  msg.id === assistantMessage.id ? assistantMessage : msg
                )
              );
            } else if (data.answer) {
              assistantMessage.content = data.answer;
              assistantMessage.metadata = data.metadata;
              
              if (data.sources || data.context_documents || data.retrieved_docs || data.documents) {
                const sources = data.sources || data.context_documents || data.retrieved_docs || data.documents;
                assistantMessage.context = sources.map((doc: any) => ({
                  content: doc.content || doc.text || '',
                  source: doc.source || doc.document_name || doc.metadata?.source || 'Unknown',
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
            // Log parsing errors to help debug response format issues
            console.warn('⚠️ Failed to parse response line:', line, 'Error:', e);
          }
        }
      }
      
      // If we have partial content from streaming but never got a final response, keep it
      if (assistantMessage && assistantMessage.content.trim()) {
        console.log('💾 Preserving partial streaming response:', assistantMessage.content.length, 'characters');
        setMessages(prev => 
          prev.map(msg => 
            msg.id === assistantMessage.id ? assistantMessage : msg
          )
        );
      }
      
    } catch (error) {
      console.error('Notebook chat request failed:', error);
      setError(getErrorMessage(error));
      
      // Check if we have a partial assistant message to preserve
      const existingAssistantMessage = messages.find(msg => msg.id === assistantMessage?.id);
      
      if (existingAssistantMessage && existingAssistantMessage.content.trim()) {
        // Update existing message with error indicator but preserve content
        setMessages(prev => 
          prev.map(msg => 
            msg.id === existingAssistantMessage.id 
              ? { 
                  ...msg, 
                  content: msg.content + '\n\n⚠️ *Response interrupted due to connection error*'
                }
              : msg
          )
        );
      } else {
        // No partial content, show full error message
        const errorMessage: NotebookChatMessage = {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: `❌ **Connection Error**\n\n${error instanceof Error ? error.message : 'Unknown error occurred'}\n\nPlease try your request again.`,
          timestamp: new Date(),
          notebook_id: notebook.id
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } finally {
      // Clear the timeout protection
      clearTimeout(loadingTimeout);
      clearTimeout(emergencyTimeout);
      setLoading(false);
      setStatusMessage('');
    }
  };

  const clearChat = async () => {
    if (!window.confirm('Clear this notebook chat? This action cannot be undone.')) {
      return;
    }
    
    setMessages([]);
    localStorage.removeItem(storageKey);
    setError('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ChatIcon color="primary" />
          <Typography variant="h6">
            Chat with "{notebook.name}"
          </Typography>
          <Chip
            icon={<DocumentIcon />}
            label={`${notebook.document_count} document${notebook.document_count !== 1 ? 's' : ''}`}
            size="small"
            variant="outlined"
            color="primary"
          />
        </Box>
        
        <IconButton onClick={clearChat} disabled={loading}>
          <ClearIcon />
        </IconButton>
      </Box>

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

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      {/* Messages */}
      <Paper sx={{ 
        flex: 1, 
        mb: 2,
        display: 'flex',
        flexDirection: 'column',
        minHeight: 400, // Minimum height to ensure scrollable area
        maxHeight: 'calc(100vh - 300px)' // Dynamic max height based on viewport
      }}>
        <Box 
          ref={messagesContainerRef}
          sx={{
            flex: 1,
            overflow: 'auto',
            overflowX: 'hidden', // Prevent horizontal scroll
            overflowY: 'scroll', // Force vertical scroll
            p: 2,
            minHeight: 0,
            height: '100%', // Ensure full height usage
            scrollBehavior: 'smooth',
            // Force scroll container behavior
            '&::-webkit-scrollbar': {
              width: '8px',
            },
            '&::-webkit-scrollbar-track': {
              background: 'rgba(0,0,0,0.1)',
              borderRadius: '4px'
            },
            '&::-webkit-scrollbar-thumb': {
              background: 'rgba(0,0,0,0.3)',
              borderRadius: '4px',
              '&:hover': {
                background: 'rgba(0,0,0,0.5)'
              }
            }
          }}
        >
        {messages.length === 0 && (
          <Alert severity="info" icon={<ChatIcon />}>
            <Typography variant="subtitle2" gutterBottom>
              Welcome to Notebook Chat!
            </Typography>
            <Typography variant="body2">
              Ask questions about the documents in "{notebook.name}". I'll search through your notebook's content to provide context-aware answers with source citations.
            </Typography>
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
              width: '100%'
            }}
          >
            <Box
              sx={{
                width: message.role === 'user' ? '80%' : '95%',
                maxWidth: message.role === 'user' ? '80%' : '100%',
                p: 2,
                borderRadius: 2,
                backgroundColor: message.role === 'user' ? 'primary.light' : 'background.paper',
                color: message.role === 'user' ? 'white' : 'text.primary'
              }}
            >
              <MessageContent content={message.content} />
            </Box>
            
            <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
              {message.timestamp.toLocaleTimeString()}
            </Typography>
            
            {/* Context documents - notebook sources */}
            {message.context && message.context.length > 0 && (
              <Box sx={{ mt: 1, width: '100%', alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start' }}>
                <Accordion sx={{ backgroundColor: 'background.default' }}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <SourceIcon fontSize="small" />
                      <Typography variant="body2">
                        {message.context.length} source{message.context.length > 1 ? 's' : ''} from notebook
                      </Typography>
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <List dense>
                      {message.context.map((doc: any, index: number) => (
                        <React.Fragment key={index}>
                          <ListItem sx={{ px: 0 }}>
                            <ListItemIcon>
                              <DocumentIcon fontSize="small" />
                            </ListItemIcon>
                            <ListItemText
                              primary={
                                <Typography variant="subtitle2" color="primary">
                                  {doc.source || `Document ${index + 1}`}
                                </Typography>
                              }
                              secondary={
                                <Typography variant="body2" sx={{ 
                                  display: '-webkit-box',
                                  WebkitLineClamp: 3,
                                  WebkitBoxOrient: 'vertical',
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                  mt: 0.5
                                }}>
                                  {doc.content}
                                </Typography>
                              }
                            />
                            {doc.score && (
                              <Chip 
                                label={`${(doc.score * 100).toFixed(1)}% match`}
                                size="small"
                                variant="outlined"
                                color="primary"
                                sx={{ ml: 1 }}
                              />
                            )}
                          </ListItem>
                          {index < message.context.length - 1 && <Divider variant="inset" component="li" />}
                        </React.Fragment>
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
              Searching notebook documents...
            </Typography>
          </Box>
        )}
        
          <div ref={messagesEndRef} />
        </Box>
      </Paper>

      {/* Input */}
      <Box sx={{ display: 'flex', gap: 1 }}>
        <Box
          component="textarea"
          ref={inputRef}
          onKeyDown={handleKeyPress}
          placeholder={`Ask questions about documents in "${notebook.name}"...`}
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
        <Button
          variant="contained"
          onClick={sendMessage}
          disabled={loading}
          sx={{ minWidth: 60 }}
        >
          {loading ? <CircularProgress size={24} /> : <SendIcon />}
        </Button>
      </Box>
    </Box>
  );
};

export default NotebookChat;