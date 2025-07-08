import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  Paper,
  TextField,
  Button,
  Switch,
  FormControlLabel,
  CircularProgress,
  Chip,
  ThemeProvider,
  createTheme,
  CssBaseline,
  IconButton,
  Tabs,
  Tab,
  Container,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import DatabaseTableManager from './components/settings/DatabaseTableManager';
import FileUploadComponent from './components/shared/FileUploadComponent';
import {
  Send as SendIcon,
  Clear as ClearIcon,
  Psychology as ThinkingIcon,
  LightMode as LightModeIcon,
  DarkMode as DarkModeIcon,
  Chat as ChatIcon,
  Group as GroupIcon,
  AccountTree as WorkflowIcon,
  Settings as SettingsIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Schedule as ScheduleIcon,
  Stream as StreamIcon,
  Description as DocumentIcon,
  Build as ToolIcon,
  ExpandMore as ExpandMoreIcon,
  Psychology as ReasoningIcon
} from '@mui/icons-material';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  source?: string;
  status?: 'sending' | 'sent' | 'streaming' | 'complete' | 'error';
  error?: string;
  context?: Array<{
    content: string;
    source: string;
    score?: number;
  }>;
  toolsUsed?: string[];
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
      style={{ height: '100%' }}
    >
      {value === index && children}
    </div>
  );
}

// Helper function to parse content with <think> tags
interface ContentPart {
  type: 'content' | 'thinking';
  text: string;
}

function parseContentWithThinking(content: string): ContentPart[] {
  const thinkRegex = /<think>([\s\S]*?)<\/think>/g;
  const parts: ContentPart[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = thinkRegex.exec(content)) !== null) {
    // Add content before the <think> tag
    if (match.index > lastIndex) {
      parts.push({
        type: 'content',
        text: content.substring(lastIndex, match.index)
      });
    }
    
    // Add the thinking content
    parts.push({
      type: 'thinking',
      text: match[1].trim()
    });
    
    lastIndex = match.index + match[0].length;
  }
  
  // Add remaining content after the last <think> tag
  if (lastIndex < content.length) {
    parts.push({
      type: 'content',
      text: content.substring(lastIndex)
    });
  }
  
  return parts;
}

// Component to render message content with markdown and thinking sections
function MessageContent({ content }: { content: string }) {
  const [expandedThinking, setExpandedThinking] = useState<Set<number>>(new Set());
  const parts = parseContentWithThinking(content);
  
  const toggleThinking = (index: number) => {
    const newExpanded = new Set(expandedThinking);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedThinking(newExpanded);
  };

  return (
    <Box>
      {parts.map((part, index) => {
        if (part.type === 'content' && part.text.trim()) {
          return (
            <ReactMarkdown
              key={index}
              remarkPlugins={[remarkGfm]}
              components={{
                p: ({ children }) => (
                  <Typography variant="body1" component="div" sx={{ mb: 1 }}>
                    {children}
                  </Typography>
                ),
                h1: ({ children }) => (
                  <Typography variant="h4" component="h1" sx={{ mb: 2, mt: 2 }}>
                    {children}
                  </Typography>
                ),
                h2: ({ children }) => (
                  <Typography variant="h5" component="h2" sx={{ mb: 1.5, mt: 1.5 }}>
                    {children}
                  </Typography>
                ),
                h3: ({ children }) => (
                  <Typography variant="h6" component="h3" sx={{ mb: 1, mt: 1 }}>
                    {children}
                  </Typography>
                ),
                code: ({ className, children }) => {
                  const inline = !className?.includes('language-');
                  return inline ? (
                    <Box
                      component="code"
                      sx={{
                        fontFamily: 'monospace',
                        backgroundColor: 'action.hover',
                        color: 'text.primary',
                        padding: '2px 4px',
                        borderRadius: 1,
                        fontSize: 'inherit'
                      }}
                    >
                      {children}
                    </Box>
                  ) : (
                    <Box
                      component="pre"
                      sx={{
                        fontFamily: 'monospace',
                        backgroundColor: 'action.hover',
                        color: 'text.primary',
                        padding: '8px',
                        borderRadius: 1,
                        fontSize: '0.875rem',
                        overflow: 'auto',
                        margin: '8px 0'
                      }}
                    >
                      <code>{children}</code>
                    </Box>
                  );
                },
                ul: ({ children }) => (
                  <Box component="ul" sx={{ ml: 2, mb: 1 }}>
                    {children}
                  </Box>
                ),
                ol: ({ children }) => (
                  <Box component="ol" sx={{ ml: 2, mb: 1 }}>
                    {children}
                  </Box>
                ),
                li: ({ children }) => (
                  <Typography component="li" variant="body1" sx={{ mb: 0.5 }}>
                    {children}
                  </Typography>
                ),
                blockquote: ({ children }) => (
                  <Box
                    sx={{
                      borderLeft: '4px solid',
                      borderLeftColor: 'primary.main',
                      pl: 2,
                      ml: 1,
                      fontStyle: 'italic',
                      color: 'text.secondary'
                    }}
                  >
                    {children}
                  </Box>
                )
              }}
            >
              {part.text}
            </ReactMarkdown>
          );
        } else if (part.type === 'thinking') {
          const isExpanded = expandedThinking.has(index);
          return (
            <Box key={index} sx={{ my: 1 }}>
              <Accordion 
                expanded={isExpanded}
                onChange={() => toggleThinking(index)}
                sx={{ backgroundColor: 'background.default' }}
              >
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <ReasoningIcon fontSize="small" color="primary" />
                    <Typography variant="body2" color="primary">
                      Reasoning
                    </Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      p: ({ children }) => (
                        <Typography variant="body2" component="div" sx={{ mb: 1, color: 'text.secondary' }}>
                          {children}
                        </Typography>
                      ),
                      code: ({ className, children }) => {
                        const inline = !className?.includes('language-');
                        return inline ? (
                          <Box
                            component="code"
                            sx={{
                              fontFamily: 'monospace',
                              backgroundColor: 'action.selected',
                              color: 'text.secondary',
                              padding: '2px 4px',
                              borderRadius: 1,
                              fontSize: '0.8rem'
                            }}
                          >
                            {children}
                          </Box>
                        ) : (
                          <Box
                            component="pre"
                            sx={{
                              fontFamily: 'monospace',
                              backgroundColor: 'action.selected',
                              color: 'text.secondary',
                              padding: '8px',
                              borderRadius: 1,
                              fontSize: '0.8rem',
                              overflow: 'auto',
                              margin: '8px 0'
                            }}
                          >
                            <code>{children}</code>
                          </Box>
                        );
                      }
                    }}
                  >
                    {part.text}
                  </ReactMarkdown>
                </AccordionDetails>
              </Accordion>
            </Box>
          );
        }
        return null;
      })}
    </Box>
  );
}

// Chat Component
function ChatInterface({ endpoint, title }: { endpoint: string, title: string }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [thinking, setThinking] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [sessionId] = useState(() => `session-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`);
  const storageKey = `jarvis-chat-${title.toLowerCase().replace(/\s+/g, '-')}`;

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
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
        console.warn('Failed to load saved conversation:', e);
      }
    }
  }, [storageKey]);

  // Save conversation to localStorage whenever messages change
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem(storageKey, JSON.stringify(messages));
    }
  }, [messages, storageKey]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: input,
      timestamp: new Date(),
      status: 'sending'
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: input,
          thinking: thinking,
          conversation_id: sessionId
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      // Mark user message as sent
      setMessages(prev => 
        prev.map(msg => 
          msg.id === userMessage.id ? { ...msg, status: 'sent' } : msg
        )
      );

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      let assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        status: 'streaming'
      };

      setMessages(prev => [...prev, assistantMessage]);

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        
        // Keep the last incomplete line in buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;
          
          console.log('Raw line received:', line);
          
          // Handle direct tool execution responses differently
          if (line.includes('direct_tool_execution')) {
            console.log('Direct tool execution:', line);
            // Don't skip - try to parse this as it might contain the actual response
          }
          
          // Try to parse as JSON first, regardless of starting character
          let data: any = null;
          try {
            data = JSON.parse(line);
            console.log('Parsed JSON data:', data);
          } catch (e) {
            // If it's not valid JSON, treat it as text content
            console.log('Non-JSON line, treating as text:', line);
            assistantMessage.content += line;
            setMessages(prev => 
              prev.map(msg => 
                msg.id === assistantMessage.id ? { ...msg, content: assistantMessage.content } : msg
              )
            );
            continue;
          }

          // Now handle the parsed JSON data
          // Handle different response types from backend
            if (data.type === 'token' && data.content) {
              assistantMessage.content += data.content;
              setMessages(prev => 
                prev.map(msg => 
                  msg.id === assistantMessage.id ? { ...msg, content: assistantMessage.content } : msg
                )
              );
            } else if (data.type === 'final' && data.response) {
              assistantMessage.content = data.response;
              assistantMessage.source = data.source || data.context;
              assistantMessage.status = 'complete';
              assistantMessage.context = data.context_documents || data.retrieved_docs;
              assistantMessage.toolsUsed = data.tools_used || data.mcp_tools;
              setMessages(prev => 
                prev.map(msg => 
                  msg.id === assistantMessage.id ? assistantMessage : msg
                )
              );
              break;
            } else if (data.type === 'status') {
              // Handle status updates (tool usage, processing steps, etc.)
              console.log('Status update:', data.message);
            } else if (data.type === 'error') {
              throw new Error(data.message || 'Backend error');
            }
            // Handle JSON-RPC responses (MCP tools)
            else if (data.jsonrpc && data.result) {
              // This is a tool execution result - log it but don't display as chat
              console.log('Tool execution result:', data.result);
              // Extract any relevant context for display
              if (data.result.documents && data.result.documents.length > 0) {
                assistantMessage.context = data.result.documents.map((doc: any) => ({
                  content: doc.content || doc.text || '',
                  source: doc.source || doc.metadata?.source || 'Unknown',
                  score: doc.relevance_score || doc.score
                }));
              }
            }
            // Handle tool execution result objects (like datetime responses)
            else if (data.content && Array.isArray(data.content)) {
              console.log('Processing tool response content:', data.content);
              // Extract text from tool response content
              const textContent = data.content
                .filter((item: any) => item.type === 'text')
                .map((item: any) => item.text)
                .join(' ');
              
              console.log('Extracted text content:', textContent);
              
              if (textContent) {
                // This looks like a datetime response, format it nicely
                if (textContent.match(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/)) {
                  const date = new Date(textContent);
                  const formatted = date.toLocaleString('en-US', {
                    weekday: 'long',
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                    timeZoneName: 'short'
                  });
                  assistantMessage.content = `The current date and time is: ${formatted}`;
                  console.log('Formatted datetime response:', assistantMessage.content);
                } else {
                  assistantMessage.content = textContent;
                  console.log('Set text content:', assistantMessage.content);
                }
                assistantMessage.status = 'complete';
                setMessages(prev => 
                  prev.map(msg => 
                    msg.id === assistantMessage.id ? assistantMessage : msg
                  )
                );
                console.log('Updated messages with tool response');
                break;
              } else {
                console.log('No text content found in tool response');
              }
            }
            // Handle direct tool execution responses
            else if (typeof data === 'string' && data.includes('direct_tool_execution')) {
              console.log('Direct tool execution:', data);
              // Skip displaying raw tool execution strings
            }
            // Handle streaming tokens (from intelligent-chat)
            else if (data.token) {
              assistantMessage.content += data.token;
              setMessages(prev => 
                prev.map(msg => 
                  msg.id === assistantMessage.id ? { ...msg, content: assistantMessage.content } : msg
                )
              );
            } 
            // Handle final answer (from intelligent-chat)
            else if (data.answer) {
              console.log('Processing answer field:', data.answer);
              
              // Check if answer is a JSON string that needs parsing
              let answerContent = data.answer;
              try {
                const parsedAnswer = JSON.parse(data.answer);
                console.log('Parsed answer JSON:', parsedAnswer);
                
                // Handle the parsed answer content (like tool responses)
                if (parsedAnswer.content && Array.isArray(parsedAnswer.content)) {
                  const textContent = parsedAnswer.content
                    .filter((item: any) => item.type === 'text')
                    .map((item: any) => item.text)
                    .join(' ');
                  
                  console.log('Extracted text from parsed answer:', textContent);
                  
                  if (textContent) {
                    // Format datetime responses
                    if (textContent.match(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/)) {
                      const date = new Date(textContent);
                      answerContent = `The current date and time is: ${date.toLocaleString('en-US', {
                        weekday: 'long',
                        year: 'numeric',
                        month: 'long',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                        timeZoneName: 'short'
                      })}`;
                    } else {
                      answerContent = textContent;
                    }
                  }
                }
              } catch (e) {
                // If not JSON, use the answer as-is
                console.log('Answer is not JSON, using as-is');
              }
              
              assistantMessage.content = answerContent;
              assistantMessage.source = data.source;
              assistantMessage.status = 'complete';
              
              // Handle documents from intelligent-chat response
              if (data.documents && data.documents.length > 0) {
                assistantMessage.context = data.documents.map((doc: any) => ({
                  content: doc.content,
                  source: doc.source,
                  score: doc.relevance_score
                }));
              }
              
              setMessages(prev => 
                prev.map(msg => 
                  msg.id === assistantMessage.id ? assistantMessage : msg
                )
              );
              break;
            }
            // Handle various streaming event types from intelligent-chat
            else if (data.event) {
              switch(data.event) {
                case 'chat_start':
                case 'tools_start':
                case 'rag_start':
                case 'synthesis_start':
                  console.log(`${data.event}:`, data.data);
                  break;
                case 'tool_result':
                case 'rag_result':
                  console.log(`${data.event}:`, data.data);
                  // Extract context from tool/rag results
                  if (data.data && data.data.documents) {
                    assistantMessage.context = data.data.documents.map((doc: any) => ({
                      content: doc.content,
                      source: doc.source,
                      score: doc.relevance_score
                    }));
                  }
                  break;
                default:
                  console.log('Unknown event:', data.event, data.data);
              }
            }
            // Catch-all for unhandled response formats
            else {
              console.log('Unhandled response format:', data);
              // If we don't know what this is but it might be a complete response, use it
              if (data && typeof data === 'object' && !data.type && !data.event && !data.jsonrpc) {
                console.log('Treating as unknown complete response');
                assistantMessage.content = JSON.stringify(data, null, 2);
                assistantMessage.status = 'complete';
                setMessages(prev => 
                  prev.map(msg => 
                    msg.id === assistantMessage.id ? assistantMessage : msg
                  )
                );
                break;
              }
            }
        }
      }
    } catch (error) {
      // Mark user message as error
      setMessages(prev => 
        prev.map(msg => 
          msg.id === userMessage.id ? { 
            ...msg, 
            status: 'error',
            error: error instanceof Error ? error.message : 'Unknown error'
          } : msg
        )
      );
      
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date(),
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    localStorage.removeItem(storageKey);
  };

  // File upload handlers
  const handleFileUploadStart = (file: File) => {
    const uploadMessage: Message = {
      id: `upload-${Date.now()}`,
      role: 'user',
      content: `📎 Uploading ${file.name}...`,
      timestamp: new Date(),
      status: 'sending'
    };
    setMessages(prev => [...prev, uploadMessage]);
  };

  const handleFileUploadSuccess = (result: any) => {
    const successMessage: Message = {
      id: `upload-success-${Date.now()}`,
      role: 'assistant',
      content: `✅ Successfully processed **${result.filename}**\n\n` +
               `• **${result.unique_chunks}** chunks added to knowledge base\n` +
               `• **Collection:** ${result.collection}\n` +
               `• **File type:** ${result.file_type}\n` +
               (result.classified_type ? `• **Document type:** ${result.classified_type}\n` : '') +
               (result.duplicates_filtered ? `• **Duplicates filtered:** ${result.duplicates_filtered}\n` : '') +
               `\nYou can now ask questions about this document!`,
      timestamp: new Date(),
      status: 'complete',
      source: result.collection
    };
    setMessages(prev => [...prev, successMessage]);
  };

  const handleFileUploadError = (error: string) => {
    const errorMessage: Message = {
      id: `upload-error-${Date.now()}`,
      role: 'assistant',
      content: `❌ **Upload failed:** ${error}`,
      timestamp: new Date(),
      status: 'error',
      error
    };
    setMessages(prev => [...prev, errorMessage]);
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Chat Controls */}
      <Box sx={{ p: 2, backgroundColor: 'background.paper', borderBottom: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6">{title}</Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {title !== 'Standard Chat' && (
              <FormControlLabel
                control={
                  <Switch
                    checked={thinking}
                    onChange={(e) => setThinking(e.target.checked)}
                    size="small"
                  />
                }
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <ThinkingIcon fontSize="small" />
                    <Typography variant="body2">Thinking</Typography>
                  </Box>
                }
              />
            )}
          </Box>
        </Box>
      </Box>

      {/* Messages Area */}
      <Box sx={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <Paper 
          sx={{ 
            flex: 1, 
            m: 2, 
            p: 2, 
            overflow: 'auto',
            backgroundColor: 'background.default'
          }}
          elevation={1}
        >
          {messages.length === 0 && (
            <Box 
              sx={{ 
                textAlign: 'center', 
                color: 'text.secondary',
                p: 4,
                border: '2px dashed',
                borderColor: 'divider',
                borderRadius: 2,
                mt: 4
              }}
            >
              <Typography variant="h6" gutterBottom>
                {title} Ready!
              </Typography>
              <Typography variant="body2">
                {title === 'Standard Chat' && 'Ask me anything about your documents using RAG and MCP tools.'}
                {title === 'Multi-Agent' && 'Complex queries will be handled by specialized AI agents working together.'}
                {title === 'Workflow' && 'Design and execute custom AI workflows with visual nodes.'}
              </Typography>
            </Box>
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
              <Paper
                sx={{
                  p: 2,
                  maxWidth: '90%',
                  backgroundColor: message.role === 'user' ? 'primary.main' : 'background.paper',
                  color: message.role === 'user' ? 'white' : 'text.primary'
                }}
                elevation={2}
              >
                <MessageContent content={message.content} />
                
                {message.source && (
                  <Box sx={{ mt: 1 }}>
                    <Chip 
                      label={message.source} 
                      size="small" 
                      variant="outlined"
                      sx={{ 
                        backgroundColor: message.role === 'user' ? 'rgba(255,255,255,0.2)' : 'transparent',
                        color: message.role === 'user' ? 'white' : 'inherit'
                      }}
                    />
                  </Box>
                )}
                
                {/* Tools used */}
                {message.toolsUsed && message.toolsUsed.length > 0 && (
                  <Box sx={{ mt: 1 }}>
                    {message.toolsUsed.map((tool, index) => (
                      <Chip 
                        key={index}
                        icon={<ToolIcon />}
                        label={tool} 
                        size="small" 
                        variant="outlined"
                        sx={{ 
                          mr: 0.5,
                          backgroundColor: message.role === 'user' ? 'rgba(255,255,255,0.2)' : 'transparent',
                          color: message.role === 'user' ? 'white' : 'inherit'
                        }}
                      />
                    ))}
                  </Box>
                )}
              </Paper>
              
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5, px: 1 }}>
                <Typography 
                  variant="caption" 
                  color="text.secondary"
                >
                  {message.timestamp.toLocaleTimeString()}
                </Typography>
                
                {/* Status indicator */}
                {message.status === 'sending' && (
                  <ScheduleIcon sx={{ fontSize: 12, color: 'text.secondary' }} />
                )}
                {message.status === 'sent' && (
                  <CheckCircleIcon sx={{ fontSize: 12, color: 'success.main' }} />
                )}
                {message.status === 'streaming' && (
                  <StreamIcon sx={{ fontSize: 12, color: 'primary.main' }} />
                )}
                {message.status === 'complete' && (
                  <CheckCircleIcon sx={{ fontSize: 12, color: 'success.main' }} />
                )}
                {message.status === 'error' && (
                  <ErrorIcon sx={{ fontSize: 12, color: 'error.main' }} />
                )}
              </Box>
              
              {/* Context documents */}
              {message.context && message.context.length > 0 && (
                <Box sx={{ mt: 1, maxWidth: '90%', alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start' }}>
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
                        {message.context.map((doc, index) => (
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
                {title === 'Multi-Agent' ? 'Agents are collaborating...' : 'Jarvis is thinking...'}
              </Typography>
            </Box>
          )}
          
          <div ref={messagesEndRef} />
        </Paper>
      </Box>

      {/* Input Area */}
      <Box sx={{ p: 2, backgroundColor: 'background.paper' }}>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
          <TextField
            fullWidth
            multiline
            minRows={2}
            maxRows={4}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={`Ask ${title}...`}
            disabled={loading}
            variant="outlined"
          />
          {title === 'Standard Chat' && (
            <Box sx={{ position: 'relative' }}>
              <FileUploadComponent
                onUploadStart={handleFileUploadStart}
                onUploadSuccess={handleFileUploadSuccess}
                onUploadError={handleFileUploadError}
                disabled={loading}
                autoClassify={true}
              />
            </Box>
          )}
          <Button
            variant="contained"
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            sx={{ minWidth: 60, height: 40 }}
          >
            {loading ? <CircularProgress size={24} color="inherit" /> : <SendIcon />}
          </Button>
        </Box>
      </Box>
    </Box>
  );
}

// Workflow Interface with Automation Management
function WorkflowInterface() {
  const [workflowTab, setWorkflowTab] = useState(0);
  const [automationData, setAutomationData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const loadAutomationWorkflows = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch('/api/v1/automation/workflows');
      if (response.ok) {
        const data = await response.json();
        setAutomationData(Array.isArray(data) ? data : data.workflows || []);
      } else {
        setAutomationData([]);
      }
    } catch (err) {
      setError('Failed to load automation workflows');
      setAutomationData([]);
    } finally {
      setLoading(false);
    }
  };

  const handleWorkflowTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setWorkflowTab(newValue);
    if (newValue === 1) {
      loadAutomationWorkflows();
    }
  };

  useEffect(() => {
    if (workflowTab === 1) {
      loadAutomationWorkflows();
    }
  }, []);

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs 
          value={workflowTab}
          onChange={handleWorkflowTabChange} 
          aria-label="workflow tabs"
          sx={{ px: 2 }}
        >
          <Tab label="Visual Designer" id="workflow-tab-0" />
          <Tab label="Automation & Workflows" id="workflow-tab-1" />
        </Tabs>
      </Box>

      {workflowTab === 0 && (
        <Box sx={{ flex: 1, p: 3 }}>
          <Typography variant="h4" gutterBottom>Workflow Designer</Typography>
          <Paper sx={{ flex: 1, p: 3, display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '60vh' }}>
            <Box sx={{ textAlign: 'center' }}>
              <WorkflowIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" gutterBottom>Visual Workflow Editor</Typography>
              <Typography variant="body2" color="text.secondary">
                Visual workflow designer will be restored here.
                <br />
                Create custom AI workflows with drag-and-drop nodes.
              </Typography>
            </Box>
          </Paper>
        </Box>
      )}

      {workflowTab === 1 && (
        <Box sx={{ flex: 1, p: 2 }}>
          <DatabaseTableManager
            category="automation"
            data={automationData}
            onChange={setAutomationData}
            onRefresh={loadAutomationWorkflows}
          />
        </Box>
      )}
    </Box>
  );
}

function App() {
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('jarvis-dark-mode');
    return saved ? JSON.parse(saved) : false;
  });
  const [tabValue, setTabValue] = useState(() => {
    // Check URL parameters for initial tab
    const urlParams = new URLSearchParams(window.location.search);
    const tab = urlParams.get('tab');
    return tab ? parseInt(tab, 10) : 0;
  });

  // Create theme based on dark mode state
  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#2196f3',
      },
      background: {
        default: darkMode ? '#121212' : '#f5f5f5',
        paper: darkMode ? '#1e1e1e' : '#ffffff',
      },
    },
  });

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    if (newValue === 1) {
      // Redirect to multi-agent page
      window.location.href = '/multi-agent.html';
      return;
    }
    if (newValue === 2) {
      // Redirect to workflow page
      window.location.href = '/workflow.html';
      return;
    }
    if (newValue === 3) {
      // Redirect to settings page
      window.location.href = '/settings.html';
      return;
    }
    setTabValue(newValue);
    // Update URL parameter
    const url = new URL(window.location.href);
    if (newValue === 0) {
      url.searchParams.delete('tab');
    } else {
      url.searchParams.set('tab', newValue.toString());
    }
    window.history.pushState({}, '', url.pathname + url.search);
  };

  const toggleDarkMode = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem('jarvis-dark-mode', JSON.stringify(newDarkMode));
  };

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
            
            {/* New Session Button - only for Standard Chat */}
            {tabValue === 0 && (
              <Button
                variant="outlined"
                onClick={() => {
                  // Clear standard chat messages - use same key pattern as ChatInterface
                  const storageKey = 'jarvis-chat-standard-chat';
                  localStorage.removeItem(storageKey);
                  window.location.reload();
                }}
                sx={{ mr: 2, color: 'white', borderColor: 'white' }}
              >
                New Session
              </Button>
            )}

            {/* Dark Mode Toggle */}
            <IconButton onClick={toggleDarkMode} color="inherit">
              {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
            </IconButton>
          </Toolbar>
        </AppBar>

        {/* Navigation Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={tabValue} 
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

        {/* Tab Content */}
        <Box sx={{ flex: 1, overflow: 'hidden' }}>
          <TabPanel value={tabValue} index={0}>
            <ChatInterface 
              endpoint="/api/v1/langchain/rag" 
              title="Standard Chat"
            />
          </TabPanel>
          <TabPanel value={tabValue} index={1}>
            <ChatInterface 
              endpoint="/api/v1/langchain/multi-agent" 
              title="Multi-Agent"
            />
          </TabPanel>
          <TabPanel value={tabValue} index={2}>
            <WorkflowInterface />
          </TabPanel>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;