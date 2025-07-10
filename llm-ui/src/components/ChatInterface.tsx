import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  TextField,
  Button,
  Paper,
  Typography,
  IconButton,
  CircularProgress,
  Switch,
  FormControlLabel,
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
import { ExpandMore as ExpandMoreIcon, Description as DocumentIcon } from '@mui/icons-material';
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
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [conversationId] = useState(() => {
    const storageKey = `jarvis-conversation-id-${title.toLowerCase().replace(/\s+/g, '-')}`;
    const saved = localStorage.getItem(storageKey);
    if (saved) {
      return saved;
    }
    const newId = `chat-${title.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`;
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
    if (!input.trim() || loading) return;

    const currentInput = input.trim();
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: currentInput,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const requestBody = {
        question: currentInput,
        thinking,
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
            
            if (data.token) {
              assistantMessage.content += data.token;
              if (data.source) assistantMessage.source = data.source;
              if (data.metadata) assistantMessage.metadata = data.metadata;
              setMessages(prev => 
                prev.map(msg => 
                  msg.id === assistantMessage.id ? assistantMessage : msg
                )
              );
            } else if (data.answer) {
              assistantMessage.content = data.answer;
              assistantMessage.source = data.source;
              assistantMessage.metadata = data.metadata;
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
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    localStorage.removeItem(storageKey);
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
          <FormControlLabel
            control={
              <Switch
                checked={thinking}
                onChange={(e) => setThinking(e.target.checked)}
              />
            }
            label="Thinking Mode"
          />
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
        <TextField
          fullWidth
          multiline
          maxRows={4}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask Jarvis anything..."
          disabled={loading}
          variant="outlined"
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
          disabled={loading || !input.trim()}
          sx={{ minWidth: 60 }}
        >
          {loading ? <CircularProgress size={24} /> : <SendIcon />}
        </Button>
      </Box>
    </Box>
  );
};

export default ChatInterface;