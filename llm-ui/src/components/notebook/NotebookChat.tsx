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
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tooltip
} from '@mui/material';
import {
  Send as SendIcon,
  Clear as ClearIcon,
  ExpandMore as ExpandMoreIcon,
  Description as DocumentIcon,
  Chat as ChatIcon,
  Source as SourceIcon,
  Download as DownloadIcon,
  ContentCopy as CopyIcon
} from '@mui/icons-material';
import { MessageContent } from '../shared/MessageContent';
import { 
  notebookAPI, 
  NotebookWithDocuments, 
  NotebookChatMessage, 
  getErrorMessage,
  CacheStatus
} from './NotebookAPI';
import CacheStatusTag from './CacheStatusTag';

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
  const [clearDialogOpen, setClearDialogOpen] = useState(false);
  const [showClearSuccess, setShowClearSuccess] = useState(false);
  
  // Cache management state
  const [cacheStatuses, setCacheStatuses] = useState<Map<string, CacheStatus & { original_query?: string; cache_age_human?: string; }>>(new Map());
  const [cacheLoading, setCacheLoading] = useState(false);
  
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
        const loadedMessages = parsed.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }));
        setMessages(loadedMessages);
        
        // Check cache status for any assistant messages with cache metadata
        loadedMessages.forEach((msg: NotebookChatMessage) => {
          if (msg.role === 'assistant' && msg.metadata && msg.metadata.cache_hit) {
            // Don't await this as it's not critical for initial load
            checkCacheStatusForMessage(msg.id, 'Previous query').catch(err => 
              console.warn('Failed to load cache status for existing message:', err)
            );
          }
        });
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
      // Smart source detection to prevent truncation for comprehensive queries
      const isComprehensiveQuery = /\b(list|all|projects|show|display|entire|complete|full)\b/i.test(currentInput);
      const maxSources = isComprehensiveQuery ? 100 : 50;
      
      const response = await notebookAPI.startNotebookChat(notebook.id, {
        message: currentInput,
        conversation_id: conversationId,
        include_context: true,
        max_sources: maxSources
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

      let hasStreamedContent = false;
      let finalAssistantMessage = { ...assistantMessage };

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          console.log('🏁 Stream completed. Final message content length:', finalAssistantMessage.content.length);
          break;
        }

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
              finalAssistantMessage.content += tokenText;
              hasStreamedContent = true;
              if (data.metadata) {
                finalAssistantMessage.metadata = data.metadata;
                // Check for cache hit information
                if (data.metadata.cache_hit) {
                  checkCacheStatusForMessage(finalAssistantMessage.id, currentInput);
                }
              }
              // Update the message with the latest content immediately
              setMessages(prev => 
                prev.map(msg => 
                  msg.id === finalAssistantMessage.id ? { ...finalAssistantMessage } : msg
                )
              );
            } else if (data.answer) {
              // Complete response received - use this as the final content
              finalAssistantMessage.content = data.answer;
              finalAssistantMessage.metadata = data.metadata;
              hasStreamedContent = true;
              
              // Check for cache hit information in final response
              if (data.metadata && data.metadata.cache_hit) {
                checkCacheStatusForMessage(finalAssistantMessage.id, currentInput);
              }
              
              if (data.sources || data.context_documents || data.retrieved_docs || data.documents) {
                const sources = data.sources || data.context_documents || data.retrieved_docs || data.documents;
                finalAssistantMessage.context = sources.map((doc: any) => ({
                  content: doc.content || doc.text || '',
                  source: doc.source || doc.document_name || doc.metadata?.source || 'Unknown',
                  score: doc.relevance_score || doc.score
                }));
              }
              
              // Update the message with final content and break
              setMessages(prev => 
                prev.map(msg => 
                  msg.id === finalAssistantMessage.id ? { ...finalAssistantMessage } : msg
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
      
      // Final preservation step - ensure the message is saved with all content
      if (hasStreamedContent && finalAssistantMessage.content.trim()) {
        console.log('✅ Final preservation of streamed response:', finalAssistantMessage.content.length, 'characters');
        // Use functional update to ensure we get the latest state and properly preserve content
        setMessages(prev => {
          const updatedMessages = prev.map(msg => 
            msg.id === finalAssistantMessage.id ? { ...finalAssistantMessage } : msg
          );
          console.log('📝 Final message state update completed');
          return updatedMessages;
        });
        
        // Also save to localStorage immediately to prevent any loss
        setTimeout(() => {
          const currentMessages = JSON.parse(localStorage.getItem(storageKey) || '[]');
          const updatedStorageMessages = currentMessages.map((msg: any) => 
            msg.id === finalAssistantMessage.id ? finalAssistantMessage : msg
          );
          localStorage.setItem(storageKey, JSON.stringify(updatedStorageMessages));
          console.log('💾 Streamed message preserved to localStorage');
        }, 10);
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
      // Don't clear status message immediately to avoid interfering with final content
      setTimeout(() => setStatusMessage(''), 100);
    }
  };

  const openClearDialog = () => {
    setClearDialogOpen(true);
  };

  const handleClearChat = () => {
    setMessages([]);
    localStorage.removeItem(storageKey);
    setError('');
    setStatusMessage('');
    setClearDialogOpen(false);
    
    // Clear cache statuses
    setCacheStatuses(new Map());
    
    // Show success message
    setShowClearSuccess(true);
    setTimeout(() => setShowClearSuccess(false), 3000);
  };

  const handleCancelClear = () => {
    setClearDialogOpen(false);
  };

  // Cache status checking for messages
  const checkCacheStatusForMessage = async (messageId: string, originalQuery: string) => {
    try {
      const cacheStatus = await notebookAPI.getCacheStatus(notebook.id, conversationId);
      if (cacheStatus.cache_exists) {
        setCacheStatuses(prev => new Map(prev.set(messageId, {
          ...cacheStatus,
          original_query: originalQuery,
          cache_age_human: calculateCacheAge(cacheStatus.created_at)
        })));
      }
    } catch (error) {
      console.warn('Failed to check cache status:', error);
    }
  };

  // Calculate human-readable cache age
  const calculateCacheAge = (createdAt?: string): string => {
    if (!createdAt) return 'cached';
    
    const created = new Date(createdAt);
    const now = new Date();
    const diffMinutes = Math.floor((now.getTime() - created.getTime()) / (1000 * 60));
    
    if (diffMinutes < 1) return 'just now';
    if (diffMinutes < 60) return `${diffMinutes} min ago`;
    
    const diffHours = Math.floor(diffMinutes / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  // Handle cache clearing
  const handleClearCache = async (notebookId: string, conversationId: string) => {
    setCacheLoading(true);
    try {
      const response = await notebookAPI.clearConversationCache(notebookId, conversationId);
      
      // Clear all cache statuses for this conversation
      setCacheStatuses(new Map());
      
      // Show success message
      setStatusMessage(`Cache cleared successfully: ${response.message}`);
      setTimeout(() => setStatusMessage(''), 3000);
      
    } catch (error) {
      console.error('Failed to clear cache:', error);
      setError('Failed to clear conversation cache');
    } finally {
      setCacheLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Convert single message to Markdown format
  const convertMessageToMarkdown = (message: NotebookChatMessage): string => {
    const timestamp = message.timestamp.toLocaleString();
    const role = message.role === 'user' ? '**You**' : '**Assistant**';
    
    let content = `# ${role} Response - ${timestamp}\n\n${message.content}\n\n`;
    
    // Add context sources if available
    if (message.context && message.context.length > 0) {
      content += `### Sources (${message.context.length})\n\n`;
      message.context.forEach((doc: any, idx: number) => {
        content += `${idx + 1}. **${doc.source}** (${doc.score ? `${(doc.score * 100).toFixed(1)}% match` : 'relevance unknown'})\n`;
        content += `   ${doc.content.substring(0, 200)}${doc.content.length > 200 ? '...' : ''}\n\n`;
      });
    }
    
    return content;
  };

  // Convert messages to Markdown format (kept for compatibility)
  const convertToMarkdown = (): string => {
    const header = `# Chat Conversation: ${notebook.name}\n\n*Downloaded on ${new Date().toLocaleString()}*\n\n---\n\n`;
    
    const conversationMarkdown = messages.map((message, index) => {
      const timestamp = message.timestamp.toLocaleString();
      const role = message.role === 'user' ? '**You**' : '**Assistant**';
      
      let content = `## ${role} - ${timestamp}\n\n${message.content}\n\n`;
      
      // Add context sources if available
      if (message.context && message.context.length > 0) {
        content += `### Sources (${message.context.length})\n\n`;
        message.context.forEach((doc: any, idx: number) => {
          content += `${idx + 1}. **${doc.source}** (${doc.score ? `${(doc.score * 100).toFixed(1)}% match` : 'relevance unknown'})\n`;
          content += `   ${doc.content.substring(0, 200)}${doc.content.length > 200 ? '...' : ''}\n\n`;
        });
      }
      
      content += '---\n\n';
      return content;
    }).join('');
    
    return header + conversationMarkdown;
  };

  // Download single message as Markdown file
  const downloadMessage = (message: NotebookChatMessage) => {
    const markdown = convertMessageToMarkdown(message);
    const blob = new Blob([markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('T')[0];
    const filename = `notebook-response-${message.id}-${timestamp}.md`;
    
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
    const filename = `notebook-chat-${notebook.name.replace(/[^a-zA-Z0-9]/g, '-')}-${timestamp}.md`;
    
    link.href = url;
    link.download = filename;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Copy single message to clipboard
  const copyMessageToClipboard = async (message: NotebookChatMessage) => {
    try {
      const markdown = convertMessageToMarkdown(message);
      await navigator.clipboard.writeText(markdown);
      
      // Show success feedback
      setStatusMessage('Response copied to clipboard!');
      setTimeout(() => setStatusMessage(''), 2000);
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
      setError('Failed to copy response to clipboard');
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
      setError('Failed to copy conversation to clipboard');
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
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          
          <Tooltip title="Clear conversation history">
            <IconButton 
              onClick={openClearDialog} 
              disabled={loading || messages.length === 0}
              color="default"
              aria-label="Clear conversation history"
            >
              <ClearIcon />
            </IconButton>
          </Tooltip>
        </Box>
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

      {/* Success Message */}
      {showClearSuccess && (
        <Alert 
          severity="success" 
          sx={{ mb: 1 }}
          onClose={() => setShowClearSuccess(false)}
        >
          Conversation history cleared successfully
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
            {/* Cache Status Tag - show above assistant messages that used cache */}
            {message.role === 'assistant' && cacheStatuses.has(message.id) && (
              <Box sx={{ mb: 1, alignSelf: 'flex-start' }}>
                <CacheStatusTag
                  cacheStatus={cacheStatuses.get(message.id)!}
                  onClearCache={handleClearCache}
                  loading={cacheLoading}
                  notebookId={notebook.id}
                  conversationId={conversationId}
                />
              </Box>
            )}
            
            <Box
              sx={{
                width: message.role === 'user' ? '80%' : '95%',
                maxWidth: message.role === 'user' ? '80%' : '100%',
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

      {/* Clear Confirmation Dialog */}
      <Dialog
        open={clearDialogOpen}
        onClose={handleCancelClear}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Clear Conversation History</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to clear all chat messages in this notebook conversation?
            This action cannot be undone and will remove {messages.length} message{messages.length !== 1 ? 's' : ''} from "{notebook.name}".
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCancelClear}>
            Cancel
          </Button>
          <Button 
            onClick={handleClearChat}
            color="error"
            variant="contained"
          >
            Clear History
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default NotebookChat;