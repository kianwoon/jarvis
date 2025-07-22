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
  ListItemText
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import ClearIcon from '@mui/icons-material/Clear';
import { 
  ExpandMore as ExpandMoreIcon, 
  Description as DocumentIcon,
  Search as SearchIcon,
  Public as WebIcon,
  AccessTime as TimeIcon,
  Build as ToolIcon
} from '@mui/icons-material';
import TempDocumentPanel from './temp-documents/TempDocumentPanel';
import FileUploadComponent from './shared/FileUploadComponent';
import { MessageContent } from './shared/MessageContent';

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

interface ChatInterfaceProps {
  endpoint?: string;
  title?: string;
  enableTemporaryDocuments?: boolean;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ 
  endpoint = '/api/v1/langchain/rag',
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
    setLoading(true);

    // Add timeout protection for loading state
    const loadingTimeout = setTimeout(() => {
      console.warn('⚠️ Request taking longer than expected, but keeping loading state...');
    }, 30000); // 30 second warning

    const emergencyTimeout = setTimeout(() => {
      console.error('❌ Emergency timeout - releasing loading state after 2 minutes');
      setLoading(false);
    }, 120000); // 2 minute emergency release

    try {
      const requestBody = {
        question: currentInput,
        conversation_id: conversationId,
        use_langgraph: false,
        // Include temporary document preferences if enabled
        ...(enableTemporaryDocuments && {
          use_hybrid_rag: documentCount > 0,
          hybrid_strategy: "temp_priority",
          fallback_to_persistent: true,
          temp_results_weight: 0.7
        })
      };
      
      console.log('🚀 SENDING REQUEST:', requestBody);
      
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

      let assistantMessage: Message = {
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
            
            // Handle status messages for user feedback
            if (data.type === 'status' && data.message) {
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

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', p: 2 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h5">{title}</Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <IconButton onClick={clearChat} disabled={loading}>
            <ClearIcon />
          </IconButton>
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
              alignItems: message.role === 'user' ? 'flex-end' : 'flex-start'
            }}
          >
            <Box
              sx={{
                maxWidth: message.role === 'user' ? '80%' : '95%',
                p: 2,
                borderRadius: 2,
                backgroundColor: message.role === 'user' ? 'primary.light' : 'background.paper',
                color: message.role === 'user' ? 'white' : 'text.primary'
              }}
            >
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
              <Box sx={{ mt: 1, maxWidth: '95%', alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start' }}>
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
      <Box sx={{ display: 'flex', gap: 1 }}>
        <Box
          component="textarea"
          ref={inputRef}
          onKeyDown={handleKeyPress}
          placeholder="Ask Jarvis anything..."
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
      </Box>
    </Box>
  );
};

export default ChatInterface;