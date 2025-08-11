import React, { useState, useEffect, useMemo } from 'react';
import { Handle, Position } from 'reactflow';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import '../../styles/workflow-animations.css';
import '../../styles/accordion-scroll.css';
import '../../styles/resizable-field.css';
import { useResizableTextField } from '../../hooks/useResizableTextField';
import { useNodeExecutionStatus } from '../../hooks/useNodeExecutionStatus';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box, 
  Select, 
  MenuItem, 
  FormControl, 
  InputLabel,
  TextField,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  FormControlLabel,
  Checkbox,
  FormLabel,
  FormGroup,
  Grid,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Button,
  Badge
} from '@mui/material';
import { 
  Psychology as PsychologyIcon,
  ExpandMore as ExpandMoreIcon,
  Build as ToolIcon,
  Input as InputIcon,
  Output as OutputIcon,
  PlayCircle as RunningIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  Schedule as IdleIcon,
  Close as CloseIcon,
  Fullscreen as ExpandIcon,
  ContentCopy as CopyIcon
} from '@mui/icons-material';

interface ExecutionData {
  input?: string;
  input_sections?: any;
  output?: string;
  tools_used?: Array<{
    tool: string;
    input: any;
    output: any;
    duration: number;
  }>;
  status?: 'idle' | 'running' | 'success' | 'error';
  timestamp?: string;
  error?: string;
}

interface AgentNodeProps {
  data: {
    label?: string;
    agentName?: string;
    customPrompt?: string;
    tools?: string[];
    timeout?: number;
    temperature?: number;
    model?: string;
    max_tokens?: number;
    query?: string;
    context?: Record<string, any>;
    executionData?: ExecutionData;
    // Enhanced state management fields
    stateEnabled?: boolean;
    stateOperation?: 'merge' | 'replace' | 'append' | 'passthrough';
    outputFormat?: 'text' | 'structured' | 'context' | 'full';
    chainKey?: string;
    // Sequence information for badge display
    agentIndex?: number;
    totalAgents?: number;
    sequenceDisplay?: string;
  };
  id: string;
  updateNodeData?: (nodeId: string, newData: any) => void;
  showIO?: boolean;
  onAccordionStateChange?: (isOpen: boolean) => void;
  onDropdownOpen?: () => void;
  onDropdownClose?: () => void;
}

interface Agent {
  name: string;
  role: string;
  system_prompt: string;
  description: string;
  tools: string[];
  config: {
    model?: string;
    temperature?: number;
    max_tokens?: number;
    timeout?: number;
  };
  is_active: boolean;
}

interface MCPTool {
  name: string;
  description: string;
  endpoint: string;
  method: string;
  parameters: any;
}

// Helper function to parse thinking content from output
const parseOutputContent = (content: string) => {
  if (!content) return { thinking: '', output: content };
  
  // Regular expression to match <think>...</think> tags (including multiline)
  const thinkingRegex = /<think>([\s\S]*?)<\/think>/gi;
  const thinkingMatches = content.match(thinkingRegex);
  
  // Extract thinking content
  const thinking = thinkingMatches 
    ? thinkingMatches.map(match => match.replace(/<\/?think>/gi, '')).join('\n\n')
    : '';
  
  // Remove thinking tags from output
  const output = content.replace(thinkingRegex, '').trim();
  
  return { thinking, output };
};

// Helper function to format content as markdown
const MarkdownContent: React.FC<{ content: string; sx?: any }> = ({ content, sx }) => (
  <ReactMarkdown
    remarkPlugins={[remarkGfm, remarkBreaks]}
    rehypePlugins={[rehypeHighlight, rehypeRaw]}
    components={{
      // Customize code blocks
      code: ({ inline, className, children, ...props }) => {
        const match = /language-(\w+)/.exec(className || '');
        return !inline && match ? (
          <pre style={{ 
            backgroundColor: 'rgba(0, 0, 0, 0.05)', 
            padding: '12px', 
            borderRadius: '4px',
            overflow: 'auto',
            fontSize: '0.875rem'
          }}>
            <code className={className} {...props}>
              {children}
            </code>
          </pre>
        ) : (
          <code 
            style={{ 
              backgroundColor: 'rgba(0, 0, 0, 0.05)', 
              padding: '2px 4px', 
              borderRadius: '2px',
              fontSize: '0.875rem'
            }} 
            {...props}
          >
            {children}
          </code>
        );
      },
      // Customize paragraphs
      p: ({ children }) => (
        <Typography variant="body2" sx={{ mb: 1, lineHeight: 1.6 }}>
          {children}
        </Typography>
      ),
      // Customize headings
      h1: ({ children }) => (
        <Typography variant="h5" sx={{ mt: 2, mb: 1, fontWeight: 600 }}>
          {children}
        </Typography>
      ),
      h2: ({ children }) => (
        <Typography variant="h6" sx={{ mt: 2, mb: 1, fontWeight: 600 }}>
          {children}
        </Typography>
      ),
      h3: ({ children }) => (
        <Typography variant="subtitle1" sx={{ mt: 1.5, mb: 1, fontWeight: 600 }}>
          {children}
        </Typography>
      ),
      // Customize lists
      ul: ({ children }) => (
        <ul style={{ marginLeft: '20px', marginBottom: '8px' }}>
          {children}
        </ul>
      ),
      ol: ({ children }) => (
        <ol style={{ marginLeft: '20px', marginBottom: '8px' }}>
          {children}
        </ol>
      ),
      // Customize links to open in new window
      a: ({ href, children }) => (
        <Box
          component="a"
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          sx={{
            color: 'primary.main',
            textDecoration: 'underline',
            '&:hover': {
              textDecoration: 'none'
            }
          }}
        >
          {children}
        </Box>
      ),
    }}
    style={sx}
  >
    {content}
  </ReactMarkdown>
);

const AgentNode: React.FC<AgentNodeProps> = ({ data, id, updateNodeData, showIO = false, onAccordionStateChange, onDropdownOpen, onDropdownClose }) => {
  const [availableAgents, setAvailableAgents] = useState<Agent[]>([]);
  const [availableTools, setAvailableTools] = useState<MCPTool[]>([]);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedAgent, setSelectedAgent] = useState('');
  const [selectedAgentData, setSelectedAgentData] = useState<Agent | null>(null);
  const [selectedTools, setSelectedTools] = useState<string[]>(data.tools || []);
  const [query, setQuery] = useState(data.query || '');
  const [context, setContext] = useState(data.context || {});
  const [bringToFront, setBringToFront] = useState(false);
  const [advancedExpanded, setAdvancedExpanded] = useState(false);
  const [toolsExpanded, setToolsExpanded] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const [ioExpanded, setIoExpanded] = useState(showIO);
  const [executionData, setExecutionData] = useState<ExecutionData>(data.executionData || { status: 'idle' });
  const [inputDialogOpen, setInputDialogOpen] = useState(false);
  const [outputDialogOpen, setOutputDialogOpen] = useState(false);
  const [executionDetailsExpanded, setExecutionDetailsExpanded] = useState(false);
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [titleValue, setTitleValue] = useState(data.label || '');
  
  // Enhanced state management fields
  const [stateEnabled, setStateEnabled] = useState(data.stateEnabled || false);
  const [stateOperation, setStateOperation] = useState(data.stateOperation || 'passthrough');
  const [outputFormat, setOutputFormat] = useState(data.outputFormat || 'text');
  const [chainKey, setChainKey] = useState(data.chainKey || '');
  
  // Sequence information for badge display
  const [sequenceDisplay, setSequenceDisplay] = useState(data.sequenceDisplay || '');
  
  // Resizable field configurations
  const queryFieldConfig = useResizableTextField({
    minHeight: 4,
    maxHeight: 12,
    resizable: true,
    autoResize: true,
    defaultHeight: 4
  }, `${id}-query`);
  
  const systemPromptFieldConfig = useResizableTextField({
    minHeight: 6,
    maxHeight: 20,
    resizable: true,
    autoResize: true,
    defaultHeight: 6
  }, `${id}-system-prompt`);

  // Sync showIO prop changes
  useEffect(() => {
    setIoExpanded(showIO);
  }, [showIO]);

  // Sync execution data from props
  useEffect(() => {
    if (data.executionData) {
      setExecutionData(data.executionData);
    }
  }, [data.executionData]);

  // Sync sequence display from props
  useEffect(() => {
    if (data.sequenceDisplay) {
      setSequenceDisplay(data.sequenceDisplay);
    }
  }, [data.sequenceDisplay]);

  // Sync title value from props
  useEffect(() => {
    setTitleValue(data.label || '');
  }, [data.label]);

  // Debug effect to monitor all data changes
  useEffect(() => {
    // Monitor data changes for debugging
  }, [data]);

  // Sync state management fields from props
  useEffect(() => {
    if (data.stateEnabled !== undefined) {
      setStateEnabled(data.stateEnabled);
    }
    if (data.stateOperation !== undefined) {
      setStateOperation(data.stateOperation);
    }
    if (data.outputFormat !== undefined) {
      setOutputFormat(data.outputFormat);
    }
    if (data.chainKey !== undefined) {
      setChainKey(data.chainKey);
    }
  }, [data.stateEnabled, data.stateOperation, data.outputFormat, data.chainKey]);

  // Notify parent when accordion state changes
  useEffect(() => {
    const isAnyAccordionOpen = advancedExpanded || toolsExpanded;
    onAccordionStateChange?.(isAnyAccordionOpen);
  }, [advancedExpanded, toolsExpanded, onAccordionStateChange]);

  // Use the universal execution status hook
  const { status, isExecuting } = useNodeExecutionStatus({
    nodeId: id,
    autoReset: false
  });
  
  // Create statusInfo based on execution data
  const statusInfo = useMemo(() => {
    const currentStatus = executionData?.status || 'idle';
    switch (currentStatus) {
      case 'running':
        return {
          nodeClass: 'workflow-node--running-agent',
          color: '#ff9800',
          icon: <RunningIcon />,
          label: 'Running'
        };
      case 'success':
        return {
          nodeClass: 'workflow-node--success',
          color: '#4caf50',
          icon: <SuccessIcon />,
          label: 'Success'
        };
      case 'error':
        return {
          nodeClass: 'workflow-node--error',
          color: '#f44336',
          icon: <ErrorIcon />,
          label: 'Error'
        };
      default:
        return {
          nodeClass: 'workflow-node--idle',
          color: '#9e9e9e',
          icon: <IdleIcon />,
          label: 'Idle'
        };
    }
  }, [executionData?.status]);
  
  // Removed animation state - now using CSS animations like ParallelNode
  
  // Debug logging
  useEffect(() => {
    // Monitor execution status for debugging
  }, [executionData?.status, isExecuting, id]);
  
  // Animation now handled entirely by CSS - removed JavaScript animation code

  // Helper function to get badge styling based on execution status
  const getBadgeColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'warning'; // Orange/Yellow for executing
      case 'success':
        return 'success'; // Green for completed
      case 'error':
        return 'error'; // Red for failed
      default:
        return 'default'; // Gray for pending/idle
    }
  };

  // Helper function to render typing indicator for running status
  const TypingIndicator = () => (
    <div className="typing-indicator">
      <div className="typing-dot"></div>
      <div className="typing-dot"></div>
      <div className="typing-dot"></div>
    </div>
  );

  // Helper function to render progress bar for running status
  const ProgressBar = () => <div className="progress-bar"></div>;

  // Helper function to copy content to clipboard
  const copyToClipboard = (content: string) => {
    navigator.clipboard.writeText(content).then(() => {
      // Content copied to clipboard
    });
  };

  // Helper function to truncate text
  const truncateText = (text: string, maxLength: number = 100) => {
    if (!text) return '';
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
  };

  // Parse agent input to extract meaningful sections for state management display
  const parseAgentInput = (input: string, inputSections?: any) => {
    // Use structured data from backend if available (new format)
    if (inputSections) {
      const hasWorkflowDependencies = inputSections.has_dependencies;
      const dependencyResults = inputSections.dependency_results || '';
      const role = inputSections.role || 'Agent';
      const userRequest = inputSections.user_request || '';
      
      let summary = '';
      if (hasWorkflowDependencies) {
        const depSummary = dependencyResults.substring(0, 150);
        summary = `📥 From previous agents: ${depSummary}${depSummary.length >= 150 ? '...' : ''}`;
      } else if (role) {
        summary = `👤 ${role}`;
      } else {
        summary = truncateText(input, 80);
      }
      
      return {
        hasWorkflowDependencies,
        role,
        dependencyResults,
        userRequest,
        summary
      };
    }
    
    // Fallback to parsing (legacy format)
    if (!input) return { hasWorkflowDependencies: false, role: '', dependencyResults: '', userRequest: '', summary: '' };
    
    const lines = input.split('\n');
    let role = '';
    let dependencyResults = '';
    let userRequest = '';
    let inDependencySection = false;
    let inUserRequestSection = false;
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      
      // Extract role
      if (line.startsWith('ROLE:')) {
        role = line.replace('ROLE:', '').trim();
      }
      
      // Detect workflow dependencies section
      if (line.includes('WORKFLOW DEPENDENCIES') || line.includes('DEPENDENCY RESULTS')) {
        inDependencySection = true;
        inUserRequestSection = false;
        continue;
      }
      
      // Detect user request section
      if (line.includes('USER REQUEST') || line.includes('ORIGINAL QUERY')) {
        inUserRequestSection = true;
        inDependencySection = false;
        continue;
      }
      
      // Stop capturing at task section
      if (line.includes('TASK:') || line.includes('INSTRUCTIONS:')) {
        inDependencySection = false;
        inUserRequestSection = false;
        continue;
      }
      
      // Capture dependency results
      if (inDependencySection && line.trim()) {
        dependencyResults += line + '\n';
      }
      
      // Capture user request
      if (inUserRequestSection && line.trim()) {
        userRequest += line + '\n';
      }
    }
    
    const hasWorkflowDependencies = dependencyResults.trim().length > 0;
    
    // Create summary for compact view
    let summary = '';
    if (hasWorkflowDependencies) {
      const depSummary = dependencyResults.trim().substring(0, 150);
      summary = `📥 From previous agents: ${depSummary}${depSummary.length >= 150 ? '...' : ''}`;
    } else if (role) {
      summary = `👤 ${role}`;
    } else {
      summary = truncateText(input, 80);
    }
    
    return {
      hasWorkflowDependencies,
      role: role || 'Agent',
      dependencyResults: dependencyResults.trim(),
      userRequest: userRequest.trim(),
      summary
    };
  };

  // Fetch available agents and tools from the API
  useEffect(() => {
    const fetchAgentsAndTools = async () => {
      try {
        // Fetch agents
        const agentsResponse = await fetch('http://127.0.0.1:8000/api/v1/langgraph/agents');
        
        if (agentsResponse.ok) {
          const agentsData = await agentsResponse.json();
          
          if (Array.isArray(agentsData)) {
            // Filter only active agents with complete data
            const activeAgents = agentsData
              .filter((agent: any) => agent.is_active)
              .map((agent: any) => ({
                name: agent.name,
                role: agent.role || agent.name,
                system_prompt: agent.system_prompt || '',
                description: agent.description || '',
                tools: agent.tools || [],
                config: {
                  model: agent.config?.model || 'claude-3-sonnet-20240229',
                  temperature: agent.config?.temperature || 0.7,
                  max_tokens: agent.config?.max_tokens || 4000,
                  timeout: agent.config?.timeout || 45
                },
                is_active: agent.is_active
              }));
            
            setAvailableAgents(activeAgents);
          }
        } else {
          // Failed to fetch agents
        }

        // Fetch MCP tools
        const toolsResponse = await fetch('http://127.0.0.1:8000/api/v1/mcp/tools');
        
        if (toolsResponse.ok) {
          const toolsData = await toolsResponse.json();
          
          if (Array.isArray(toolsData)) {
            const mcpTools = toolsData.map((tool: any) => ({
              name: tool.name,
              description: tool.description || '',
              endpoint: tool.endpoint || '',
              method: tool.method || 'GET',
              parameters: tool.parameters || {}
            }));
            
            setAvailableTools(mcpTools);
          }
        } else {
          // Failed to fetch tools
        }

        // Fetch available models from Ollama API
        const modelsResponse = await fetch('http://127.0.0.1:8000/api/v1/ollama/models');
        
        if (modelsResponse.ok) {
          const modelsData = await modelsResponse.json();
          
          let modelNames: string[] = [];
          if (modelsData.success && modelsData.models) {
            // Ollama is available - use actual models
            modelNames = modelsData.models.map((model: any) => model.name).filter(Boolean);
          } else if (modelsData.fallback_models) {
            // Ollama not available - use fallback models from API
            modelNames = modelsData.fallback_models.map((model: any) => model.name).filter(Boolean);
          } else {
            // Last resort fallback
            modelNames = ['llama3.1:8b', 'deepseek-r1:8b', 'qwen2.5:32b'];
          }
          
          setAvailableModels(modelNames);
        } else {
          // Failed to fetch models - use fallback
          setAvailableModels(['llama3.1:8b', 'deepseek-r1:8b', 'qwen2.5:32b']);
        }
      } catch (error) {
        // Failed to fetch data
        // Fallback agents
        const fallbackAgents = [
          { 
            name: 'Researcher Agent', 
            role: 'Researcher Agent', 
            system_prompt: 'You are a helpful research assistant.',
            description: 'Conducts research and analysis',
            tools: [],
            config: { temperature: 0.7, timeout: 45 },
            is_active: true 
          },
          { 
            name: 'CEO Agent', 
            role: 'CEO Agent', 
            system_prompt: 'You are a strategic business advisor.',
            description: 'Provides strategic business guidance',
            tools: [],
            config: { temperature: 0.7, timeout: 45 },
            is_active: true 
          }
        ];
        setAvailableAgents(fallbackAgents);
        
        // Fallback models
        setAvailableModels(['qwen3:30b-a3b', 'llama3.1:70b', 'llama3.1:8b']);
      }
    };

    fetchAgentsAndTools();
  }, []);

  const handleAgentChange = (agentName: string) => {
    setSelectedAgent(agentName);
    
    // Find the selected agent data
    const agentData = availableAgents.find(agent => agent.name === agentName);
    setSelectedAgentData(agentData || null);
    
    // Pre-fill context with agent's system prompt and config
    if (agentData) {
      // Don't pollute context with agent template data - only use for UI state
      const newContext = {
        ...context
        // DO NOT copy agent template data to context - causes duplicate data pollution
      };
      setContext(newContext);
      
      // Set default tools from agent configuration
      const defaultTools = agentData.tools || [];
      setSelectedTools(defaultTools);
      
      updateNodeData?.(id, { 
        agentName, 
        context: newContext,
        tools: defaultTools,
        customPrompt: agentData.system_prompt
      });
    } else {
      updateNodeData?.(id, { agentName });
    }
  };

  // Set initial agent once options are loaded
  useEffect(() => {
    // Only initialize once when agents are loaded and we haven't initialized yet
    if (availableAgents.length > 0 && !isInitialized) {
      // Initial setup - checking for agent restore
      
      if (data.agentName && availableAgents.some(a => a.name === data.agentName)) {
        setSelectedAgent(data.agentName);
        
        // Also restore the agent data when loading existing workflow
        const agentData = availableAgents.find(agent => agent.name === data.agentName);
        if (agentData) {
          setSelectedAgentData(agentData);
          
          // If we have stored tools, use them, otherwise use agent defaults
          if (data.tools && Array.isArray(data.tools) && data.tools.length >= 0) {
            setSelectedTools(data.tools);
          } else {
            setSelectedTools(agentData.tools || []);
          }
          
          // If we have stored context, use it, otherwise use agent defaults
          if (data.context && Object.keys(data.context).length > 0) {
            setContext(data.context);
          } else {
            const defaultContext = {
              system_prompt: agentData.system_prompt,
              temperature: agentData.config.temperature,
              timeout: agentData.config.timeout,
              model: agentData.config.model
            };
            setContext(defaultContext);
          }
        }
      } else {
        // No agent to restore or agent not found
      }

      // Restore enhanced state management fields - always check all fields
      
      // Set each field explicitly to ensure proper restoration
      if (data.stateEnabled !== undefined) {
        setStateEnabled(data.stateEnabled);
      }
      if (data.stateOperation !== undefined) {
        setStateOperation(data.stateOperation);
      }
      if (data.outputFormat !== undefined) {
        setOutputFormat(data.outputFormat);
      }
      if (data.chainKey !== undefined) {
        setChainKey(data.chainKey);
      }
      
      setIsInitialized(true);
    }
  }, [availableAgents, data.agentName, isInitialized]); // Only run when agents load or agentName changes, not on tool changes

  // Debug: Log current state
  useEffect(() => {
    // Monitor current state for debugging
  }, [availableAgents, selectedAgent]);

  const handleQueryChange = (newQuery: string) => {
    setQuery(newQuery);
    updateNodeData?.(id, { query: newQuery });
  };


  const handleToolsChange = (toolName: string, isSelected: boolean) => {
    const newSelectedTools = isSelected
      ? [...selectedTools, toolName]
      : selectedTools.filter(t => t !== toolName);
    
    setSelectedTools(newSelectedTools);
    updateNodeData?.(id, { tools: newSelectedTools });
  };

  const handleRemoveTool = (toolToRemove: string) => {
    setSelectedTools(prevTools => {
      const newTools = prevTools.filter(t => t !== toolToRemove);
      
      // Use setTimeout to ensure the state update happens before calling updateNodeData
      setTimeout(() => {
        updateNodeData?.(id, { tools: newTools });
      }, 0);
      
      return newTools;
    });
  };

  // Enhanced state management handlers
  const handleStateEnabledChange = (enabled: boolean) => {
    setStateEnabled(enabled);
    updateNodeData?.(id, { stateEnabled: enabled });
  };

  const handleStateOperationChange = (operation: string) => {
    setStateOperation(operation as 'merge' | 'replace' | 'append' | 'passthrough');
    updateNodeData?.(id, { stateOperation: operation });
  };

  const handleOutputFormatChange = (format: string) => {
    setOutputFormat(format as 'text' | 'structured' | 'context' | 'full');
    updateNodeData?.(id, { outputFormat: format });
  };

  const handleChainKeyChange = (key: string) => {
    setChainKey(key);
    updateNodeData?.(id, { chainKey: key });
  };

  // Determine if we should show I/O panels
  const hasInput = executionData.input && executionData.status !== 'idle';
  const hasOutput = executionData.output && executionData.status !== 'idle';
  const hasExecutionData = hasInput || hasOutput || isExecuting;

  // Fixed dimensions to prevent inconsistent node sizes
  const calculateNodeDimensions = () => {
    // Use consistent fixed dimensions for all AI agent nodes
    return { 
      minWidth: 320, 
      maxWidth: 400 
    };
  };
  
  const dimensions = calculateNodeDimensions();
  
  
  return (
    <>
      <Badge
        badgeContent={sequenceDisplay}
        color={getBadgeColor(executionData.status || 'idle') as 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning'}
        overlap="circular"
        anchorOrigin={{
          vertical: 'top',
          horizontal: 'left',
        }}
        invisible={!sequenceDisplay}
        sx={{
          overflow: 'visible !important',
          '& .MuiBadge-badge': {
            fontSize: '0.9rem',
            fontWeight: 600,
            minWidth: '32px',
            height: '24px',
            borderRadius: '12px',
            animation: executionData.status === 'running' ? 'pulse 2s infinite' : 'none',
            // Position the badge slightly lower
            transform: 'translate(-50%, -100%)',
            top: '3px',
            left: '20px',
            '@keyframes pulse': {
              '0%': { opacity: 1 },
              '50%': { opacity: 0.7 },
              '100%': { opacity: 1 },
            },
          }
        }}
      >
        <div 
          style={{ 
            position: 'relative', 
            overflow: 'visible'
          }}
        >
          <Card 
            className={`workflow-node node-resize-transition ${statusInfo.nodeClass}`} 
            onClick={() => {
              setBringToFront(true);
              setTimeout(() => setBringToFront(false), 5000);
            }}
            sx={(theme) => ({ 
              minWidth: dimensions.minWidth,
              maxWidth: dimensions.maxWidth,
              minHeight: 280,
              width: 'fit-content',
              // Remove all border styling - let CSS animations handle borders completely
              bgcolor: theme.palette.mode === 'dark' 
                ? 'rgba(171, 71, 188, 0.12)' 
                : 'rgba(156, 39, 176, 0.08)',
              color: theme.palette.text.primary,
              position: 'relative',
              // Bulletproof z-index solution
              zIndex: bringToFront ? 9999999 : 1,
              overflow: 'visible !important',
              // CSS-based animation (like ParallelNode) - remove border styles when executing to let CSS animations handle them
              ...(isExecuting ? {} : {
                border: `2px solid ${theme.palette.mode === 'dark' ? '#ab47bc' : '#9c27b0'}`,
                transition: 'all 0.3s ease-in-out'
              })
            })}>
      <CardContent 
        onClick={() => {
          setBringToFront(true);
          setTimeout(() => setBringToFront(false), 5000);
        }}
        sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}
      >
        {/* Header with agent name and status */}
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={1} sx={{ minHeight: 40 }}>
          <Box display="flex" alignItems="center" gap={1}>
            <PsychologyIcon sx={(theme) => ({ 
              color: theme.palette.mode === 'dark' ? '#ab47bc' : '#9c27b0' 
            })} />
            <Box display="flex" flexDirection="column" gap={0.2} sx={{ flex: 1, minWidth: 0 }}>
              {!isEditingTitle ? (
                <Box>
                  <Typography 
                    variant="subtitle2" 
                    onClick={() => setIsEditingTitle(true)}
                    sx={(theme) => ({ 
                      fontWeight: 600, 
                      color: theme.palette.mode === 'dark' ? '#ce93d8' : '#4a148c',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      cursor: 'pointer',
                      '&:hover': {
                        opacity: 0.8
                      }
                    })}>
                    {data.label || 'AI Agent'}
                  </Typography>
                  <Typography 
                    variant="caption"
                    sx={(theme) => ({
                      fontSize: '0.7rem', 
                      fontWeight: 400,
                      opacity: 0.6,
                      color: theme.palette.mode === 'dark' ? '#ce93d8' : '#4a148c'
                    })}>
                    ({id})
                  </Typography>
                </Box>
              ) : (
                <TextField
                  value={titleValue}
                  onChange={(e) => setTitleValue(e.target.value)}
                  onBlur={() => {
                    if (titleValue !== data.label) {
                      if (updateNodeData) {
                        updateNodeData(id, { label: titleValue });
                      }
                    }
                    setIsEditingTitle(false);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      e.stopPropagation();
                      if (titleValue !== data.label) {
                        if (updateNodeData) {
                          updateNodeData(id, { label: titleValue });
                        }
                      }
                      setIsEditingTitle(false);
                    } else if (e.key === 'Escape') {
                      setTitleValue(data.label || '');
                      setIsEditingTitle(false);
                    }
                  }}
                  autoFocus
                  size="small"
                  placeholder="Enter node title"
                  sx={{
                    mb: 0.5,
                    '& .MuiInputBase-input': {
                      fontSize: '0.875rem',
                      fontWeight: 600,
                      padding: '2px 4px'
                    },
                    '& .MuiOutlinedInput-root': {
                      '& fieldset': {
                        border: '1px solid rgba(206, 147, 216, 0.3)'
                      },
                      '&:hover fieldset': {
                        border: '1px solid rgba(206, 147, 216, 0.5)'
                      },
                      '&.Mui-focused fieldset': {
                        border: '1px solid #ce93d8'
                      }
                    }
                  }}
                />
              )}
            </Box>
          </Box>
          
          {/* Execution Status */}
          {hasExecutionData && (
            <Box display="flex" alignItems="center" gap={0.5}>
              <Box sx={{ color: statusInfo.color }}>
                {statusInfo.icon}
              </Box>
              <Typography variant="caption" sx={{ 
                fontWeight: 600, 
                color: statusInfo.color,
                fontSize: '0.7rem'
              }}>
                {statusInfo.label}
                {isExecuting && <TypingIndicator />}
              </Typography>
            </Box>
          )}
        </Box>

        {/* Progress Bar for Running Status */}
        {isExecuting && (
          <Box sx={{ mb: 1 }}>
            <ProgressBar />
          </Box>
        )}

        {/* Execution Details Toggle */}
        {hasExecutionData && (
          <Box 
            display="flex" 
            alignItems="center" 
            justifyContent="center" 
            sx={{ mb: 1, cursor: 'pointer' }}
            onClick={() => setExecutionDetailsExpanded(!executionDetailsExpanded)}
          >
            <Typography 
              variant="caption" 
              sx={{ 
                color: 'text.secondary',
                fontSize: '0.65rem',
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
                '&:hover': { color: 'primary.main' }
              }}
            >
              {executionDetailsExpanded ? 'Collapse Details' : 'Show Details'}
              <ExpandMoreIcon 
                sx={{ 
                  fontSize: '0.8rem',
                  transform: executionDetailsExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                  transition: 'transform 0.3s ease'
                }} 
              />
            </Typography>
          </Box>
        )}

        {/* Agent Configuration - Always Visible */}
        <Box sx={{ mb: 1 }}>
          {/* Agent Selection */}
          <div 
            className="nodrag"
            onClick={() => {
              setBringToFront(true);
              setTimeout(() => setBringToFront(false), 5000);
            }}
          >
            <FormControl 
              fullWidth 
              size="small" 
              sx={{ mb: 1 }}
              onMouseDown={(e) => e.stopPropagation()}
              onClick={(e) => e.stopPropagation()}
            >
              <InputLabel>Agent</InputLabel>
              <Select
                value={selectedAgent}
                onChange={(e) => handleAgentChange(e.target.value)}
                onClick={() => {
                  setBringToFront(true);
                  setTimeout(() => setBringToFront(false), 5000);
                }}
                displayEmpty
                renderValue={(value) => {
                  if (!value) {
                    return '';  // Return empty string when no agent is selected
                  }
                  return value;
                }}
                sx={{ fontSize: '0.875rem' }}
                MenuProps={{
                  variant: "menu",
                  disableAutoFocusItem: true,
                  disableScrollLock: true,
                  PaperProps: {
                    sx: {
                      maxHeight: 200,
                      overflow: 'auto',
                      zIndex: 99999999,
                      '&::-webkit-scrollbar': {
                        width: '8px',
                      },
                      '&::-webkit-scrollbar-track': {
                        background: '#f1f1f1',
                      },
                      '&::-webkit-scrollbar-thumb': {
                        background: '#888',
                      }
                    }
                  }
                }}
              >
                <MenuItem value="">
                  <em>Select Agent ({availableAgents.length} available)</em>
                </MenuItem>
                {availableAgents
                  .sort((a, b) => a.name.localeCompare(b.name))
                  .map((agent, index) => (
                    <MenuItem key={index} value={agent.name}>
                      {agent.name}
                    </MenuItem>
                  ))}
              </Select>
            </FormControl>
          </div>


          {/* Query Input */}
          <div className={`nodrag resizable-field-container ${queryFieldConfig.state.isResizing ? 'resizing' : ''}`}>
            <TextField
              fullWidth
              size="small"
              label="Query"
              variant="outlined"
              multiline
              {...queryFieldConfig.fieldProps}
              value={query}
              onChange={(e) => {
                handleQueryChange(e.target.value);
                queryFieldConfig.actions.autoResizeToContent(e.target.value);
              }}
              onMouseDown={(e) => e.stopPropagation()}
              onClick={(e) => e.stopPropagation()}
              sx={{ 
                mb: 1, 
                fontSize: '0.875rem',
                ...queryFieldConfig.fieldProps.sx,
                '& .MuiInputBase-root': {
                  ...queryFieldConfig.fieldProps.sx['& .MuiInputBase-root'],
                  minHeight: queryFieldConfig.state.height || 96
                }
              }}
            />
            {queryFieldConfig.state.canResize && (
              <div 
                {...queryFieldConfig.resizeHandleProps}
                className={`nodrag resize-handle ${queryFieldConfig.state.isResizing ? 'active' : ''}`}
              />
            )}
          </div>

          {/* Status Summary */}
          <Box sx={{ minHeight: 40, display: 'flex', alignItems: 'center' }}>
            <Typography variant="caption" color="text.secondary" display="block" sx={{ 
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              maxWidth: '100%'
            }}>
              {selectedAgentData 
                ? `${truncateText(selectedAgentData.role, 20)} • ${selectedTools.length} tools${stateEnabled ? ' • State Enabled' : ''}` 
                : 'AI Agent execution'
              }
            </Typography>
          </Box>
        </Box>

        {/* Execution Details Section - Expandable */}
        {executionDetailsExpanded && hasExecutionData && (
          <Box sx={{ 
            borderTop: '1px solid rgba(156, 39, 176, 0.2)',
            pt: 1,
            mb: 1,
            minHeight: 120,
            transition: 'all 0.3s ease-in-out'
          }}>
            <Grid container spacing={1}>
              {/* Input Panel */}
              {hasInput && (
                <Grid item xs={hasOutput ? 6 : 12}>
                  <Paper 
                    className="panel-expand-enter-active"
                    sx={{ 
                      p: 1, 
                      bgcolor: 'rgba(33, 150, 243, 0.08)', 
                      borderLeft: '3px solid #2196F3',
                      cursor: 'pointer',
                      minHeight: 100,
                      maxHeight: 150,
                      overflow: 'hidden',
                      transition: 'all 0.3s ease-in-out',
                      '&:hover': { 
                        bgcolor: 'rgba(33, 150, 243, 0.12)',
                        transform: 'scale(1.02)'
                      }
                    }}
                    onClick={() => setInputDialogOpen(true)}
                  >
                    <Box display="flex" alignItems="center" gap={0.5} mb={0.5}>
                      <InputIcon sx={{ fontSize: '0.75rem', color: '#2196F3' }} />
                      <Typography variant="caption" sx={{ fontWeight: 600, color: '#2196F3', fontSize: '0.65rem' }}>
                        Input
                      </Typography>
                      <ExpandIcon sx={{ fontSize: '0.6rem', color: '#2196F3', ml: 'auto' }} />
                    </Box>
                    <Typography variant="caption" sx={{ 
                      fontSize: '0.6rem', 
                      wordBreak: 'break-word',
                      display: 'block',
                      lineHeight: 1.2,
                      maxHeight: 60,
                      overflow: 'hidden'
                    }}>
                      {(() => {
                        const parsed = parseAgentInput(executionData.input || '', executionData.input_sections);
                        return parsed.hasWorkflowDependencies ? (
                          <Box>
                            <Typography variant="caption" sx={{ 
                              fontSize: '0.55rem', 
                              color: '#4CAF50', 
                              fontWeight: 600,
                              display: 'block',
                              mb: 0.3
                            }}>
                              🔗 State: {data.stateEnabled ? `${data.stateOperation || 'passthrough'} → ${data.outputFormat || 'text'}` : 'disabled'}
                            </Typography>
                            <Typography variant="caption" sx={{ 
                              fontSize: '0.6rem',
                              display: 'block'
                            }}>
                              {parsed.summary}
                            </Typography>
                          </Box>
                        ) : (
                          parsed.summary
                        );
                      })()} 
                    </Typography>
                    {executionData.timestamp && (
                      <Typography variant="caption" sx={{ 
                        fontSize: '0.55rem', 
                        color: 'text.secondary',
                        display: 'block',
                        mt: 0.5
                      }}>
                        {new Date(executionData.timestamp).toLocaleTimeString()}
                      </Typography>
                    )}
                  </Paper>
                </Grid>
              )}
              
              {/* Output Panel */}
              {hasOutput && (
                <Grid item xs={hasInput ? 6 : 12}>
                  <Paper 
                    className="panel-expand-enter-active"
                    sx={{ 
                      p: 1, 
                      bgcolor: 'rgba(76, 175, 80, 0.08)', 
                      borderLeft: '3px solid #4CAF50',
                      cursor: 'pointer',
                      minHeight: 100,
                      maxHeight: 150,
                      overflow: 'hidden',
                      transition: 'all 0.3s ease-in-out',
                      '&:hover': { 
                        bgcolor: 'rgba(76, 175, 80, 0.12)',
                        transform: 'scale(1.02)'
                      }
                    }}
                    onClick={() => setOutputDialogOpen(true)}
                  >
                    <Box display="flex" alignItems="center" gap={0.5} mb={0.5}>
                      <OutputIcon sx={{ fontSize: '0.75rem', color: '#4CAF50' }} />
                      <Typography variant="caption" sx={{ fontWeight: 600, color: '#4CAF50', fontSize: '0.65rem' }}>
                        Output
                      </Typography>
                      <ExpandIcon sx={{ fontSize: '0.6rem', color: '#4CAF50', ml: 'auto' }} />
                    </Box>
                    <Typography variant="caption" sx={{ 
                      fontSize: '0.6rem', 
                      wordBreak: 'break-word',
                      display: 'block',
                      lineHeight: 1.2,
                      maxHeight: 60,
                      overflow: 'hidden'
                    }}>
                      {truncateText(executionData.output || '', 80)}
                    </Typography>
                    {executionData.tools_used && executionData.tools_used.length > 0 && (
                      <Typography variant="caption" sx={{ 
                        fontSize: '0.55rem', 
                        color: '#FF9800',
                        display: 'block',
                        mt: 0.5
                      }}>
                        {executionData.tools_used.length} tools used
                      </Typography>
                    )}
                  </Paper>
                </Grid>
              )}
            </Grid>
          </Box>
        )}

        {/* Advanced Configuration - Collapsible */}
        {selectedAgent && (
          <Accordion 
            expanded={advancedExpanded} 
            onChange={() => setAdvancedExpanded(!advancedExpanded)}
            sx={{ 
              boxShadow: 'none', 
              '&:before': { display: 'none' },
              zIndex: 50000
            }}
          >
            <AccordionSummary
              expandIcon={<ExpandMoreIcon />}
              sx={{ 
                p: 0, 
                minHeight: 32, 
                '& .MuiAccordionSummary-content': { my: 0.5 },
                zIndex: 50001
              }}
              className="nodrag"
              onMouseDown={(e) => e.stopPropagation()}
            >
              <Typography variant="caption" color="text.secondary">
                Advanced Configuration
              </Typography>
            </AccordionSummary>
            <AccordionDetails 
              sx={{ 
                p: 0,
                height: '350px',
                maxHeight: '350px',
                overflow: 'hidden',
                position: 'relative'
              }}
            >
              <div
                style={{
                  height: '350px',
                  overflowY: 'scroll',
                  overflowX: 'hidden',
                  padding: '8px',
                  borderRadius: '4px'
                }}
                className="accordion-scroll-container"
                onWheel={(e) => {
                  // Prevent ReactFlow zoom and handle scroll manually
                  e.preventDefault();
                  e.stopPropagation();
                  
                  // Manual scroll
                  const container = e.currentTarget;
                  container.scrollTop += e.deltaY;
                }}
              >
              {/* System Prompt */}
              <div className={`nodrag resizable-field-container ${systemPromptFieldConfig.state.isResizing ? 'resizing' : ''}`}>
                <TextField
                  fullWidth
                  size="small"
                  label="System Prompt"
                  variant="outlined"
                  multiline
                  {...systemPromptFieldConfig.fieldProps}
                  value={data.customPrompt !== undefined ? data.customPrompt : (context.system_prompt || '')}
                  onChange={(e) => {
                    // Save whatever is in the field to customPrompt
                    updateNodeData?.(id, { customPrompt: e.target.value });
                    systemPromptFieldConfig.actions.autoResizeToContent(e.target.value);
                  }}
                  onMouseDown={(e) => e.stopPropagation()}
                  onClick={(e) => e.stopPropagation()}
                  sx={{ 
                    mb: 1, 
                    fontSize: '0.75rem',
                    ...systemPromptFieldConfig.fieldProps.sx,
                    '& .MuiInputBase-root': {
                      ...systemPromptFieldConfig.fieldProps.sx['& .MuiInputBase-root'],
                      minHeight: systemPromptFieldConfig.state.height || 144
                    }
                  }}
                  placeholder={selectedAgentData ? 'Using agent default prompt...' : 'Enter custom system prompt'}
                />
                {systemPromptFieldConfig.state.canResize && (
                  <div 
                    {...systemPromptFieldConfig.resizeHandleProps}
                    className={`nodrag resize-handle ${systemPromptFieldConfig.state.isResizing ? 'active' : ''}`}
                  />
                )}
              </div>

              {/* Temperature and Timeout */}
              <div className="nodrag">
                <Box display="flex" gap={1} mb={1}>
                  <TextField
                    size="small"
                    label="Temperature"
                    type="number"
                    variant="outlined"
                    slotProps={{ htmlInput: { min: 0, max: 1, step: 0.1 } }}
                    value={data.temperature || 0.7}
                    onChange={(e) => {
                      updateNodeData?.(id, { temperature: parseFloat(e.target.value) });
                    }}
                    onMouseDown={(e) => e.stopPropagation()}
                    onClick={(e) => e.stopPropagation()}
                    sx={{ flex: 1 }}
                  />
                  <TextField
                    size="small"
                    label="Timeout (s)"
                    type="number"
                    variant="outlined"
                    slotProps={{ htmlInput: { min: 10, max: 300 } }}
                    value={data.timeout || 45}
                    onChange={(e) => {
                      updateNodeData?.(id, { timeout: parseInt(e.target.value) });
                    }}
                    onMouseDown={(e) => e.stopPropagation()}
                    onClick={(e) => e.stopPropagation()}
                    sx={{ flex: 1 }}
                  />
                </Box>
              </div>

              {/* Model and Max Tokens */}
              <div className="nodrag">
                <Box display="flex" gap={1} mb={1}>
                  <FormControl size="small" sx={{ flex: 1 }}>
                      <InputLabel>Model</InputLabel>
                      <Select
                      value={data.model || (availableModels.length > 0 ? availableModels[0] : '')}
                      onChange={(e) => {
                        updateNodeData?.(id, { model: e.target.value });
                      }}
                      onOpen={() => {
                        onDropdownOpen?.();
                        setBringToFront(true);
                        setTimeout(() => setBringToFront(false), 5000);
                      }}
                      onClose={() => {
                        onDropdownClose?.();
                      }}
                      onClick={() => {
                        setBringToFront(true);
                        setTimeout(() => setBringToFront(false), 5000);
                      }}
                      label="Model"
                      MenuProps={{
                        variant: "menu",
                        disableAutoFocusItem: true,
                        disableScrollLock: true,
                        PaperProps: {
                          sx: {
                            maxHeight: 200,
                            overflow: 'auto',
                            zIndex: 99999999,
                            '&::-webkit-scrollbar': {
                              width: '8px',
                            },
                            '&::-webkit-scrollbar-track': {
                              background: '#f1f1f1',
                            },
                            '&::-webkit-scrollbar-thumb': {
                              background: '#888',
                            }
                          }
                        }
                      }}
                    >
                      {availableModels.length === 0 ? (
                        <MenuItem value="" disabled>
                          <em>Loading models...</em>
                        </MenuItem>
                      ) : (
                        availableModels.map((model) => (
                          <MenuItem key={model} value={model}>
                            {model}
                          </MenuItem>
                        ))
                      )}
                    </Select>
                  </FormControl>
                  <TextField
                    size="small"
                    label="Max Tokens"
                    type="number"
                    variant="outlined"
                    slotProps={{ htmlInput: { min: 100, max: 8000 } }}
                    value={data.max_tokens || 4000}
                    onChange={(e) => {
                      updateNodeData?.(id, { max_tokens: parseInt(e.target.value) });
                    }}
                    onMouseDown={(e) => e.stopPropagation()}
                    onClick={(e) => e.stopPropagation()}
                    sx={{ flex: 1 }}
                  />
                </Box>
              </div>

              {/* Enhanced State Management Section */}
              <div className="nodrag">
                <Box sx={{ mt: 2, p: 1, bgcolor: 'rgba(156, 39, 176, 0.05)', borderRadius: 1, border: '1px solid rgba(156, 39, 176, 0.2)' }}>
                  <Typography variant="caption" sx={{ fontWeight: 600, color: '#9c27b0', display: 'block', mb: 1 }}>
                    🔗 Enhanced State Management
                  </Typography>
                  
                  {/* State Enabled Checkbox */}
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={stateEnabled}
                        onChange={(e) => {
                          e.stopPropagation();
                          handleStateEnabledChange(e.target.checked);
                        }}
                        onMouseDown={(e) => e.stopPropagation()}
                        onClick={(e) => e.stopPropagation()}
                        size="small"
                        sx={{ py: 0.5 }}
                      />
                    }
                    label="Enable State Management"
                    sx={{ mb: 1, '& .MuiFormControlLabel-label': { fontSize: '0.75rem' } }}
                    className="nodrag"
                  />

                  {/* Conditional fields shown when state is enabled */}
                  {stateEnabled && (
                    <Box 
                      sx={{ pl: 3, borderLeft: '2px solid rgba(156, 39, 176, 0.2)' }} 
                      className="nodrag"
                      onClick={() => {
                        setBringToFront(true);
                        setTimeout(() => setBringToFront(false), 5000);
                      }}
                    >
                      {/* State Operation */}
                      <FormControl 
                        fullWidth 
                        size="small" 
                        sx={{ mb: 1 }} 
                        className="nodrag"
                        onClick={() => {
                          setBringToFront(true);
                          setTimeout(() => setBringToFront(false), 5000);
                        }}
                      >
                          <Typography variant="caption" sx={{ mb: 0.5, display: 'block', fontWeight: 500 }}>
                            State Operation
                          </Typography>
                          <Select
                            value={stateOperation}
                            onChange={(e) => {
                              e.stopPropagation();
                              handleStateOperationChange(e.target.value);
                            }}
                            onOpen={(e) => {
                              e?.stopPropagation();
                              onDropdownOpen?.();
                            }}
                            onClose={(e) => {
                              e?.stopPropagation();
                              onDropdownClose?.();
                            }}
                            onMouseDown={(e) => e.stopPropagation()}
                            onClick={(e) => e.stopPropagation()}
                            sx={{ fontSize: '0.75rem' }}
                            MenuProps={{
                              disableScrollLock: true,
                              PaperProps: {
                                sx: {
                                  maxHeight: 200,
                                  overflow: 'auto',
                                  zIndex: 9999,
                                }
                              },
                              anchorOrigin: {
                                vertical: 'bottom',
                                horizontal: 'left',
                              },
                              transformOrigin: {
                                vertical: 'top',
                                horizontal: 'left',
                              }
                            }}
                          >
                            <MenuItem value="passthrough">Pass Output Directly</MenuItem>
                            <MenuItem value="merge">Merge with Previous</MenuItem>
                            <MenuItem value="replace">Replace Previous</MenuItem>
                            <MenuItem value="append">Append to Previous</MenuItem>
                          </Select>
                        </FormControl>

                      {/* Output Format */}
                      <FormControl 
                        fullWidth 
                        size="small" 
                        sx={{ mb: 1 }} 
                        className="nodrag"
                        onClick={() => {
                          setBringToFront(true);
                          setTimeout(() => setBringToFront(false), 5000);
                        }}
                      >
                          <Typography variant="caption" sx={{ mb: 0.5, display: 'block', fontWeight: 500 }}>
                            Output Format
                          </Typography>
                          <Select
                            value={outputFormat}
                            onChange={(e) => {
                              e.stopPropagation();
                              handleOutputFormatChange(e.target.value);
                            }}
                            onOpen={(e) => {
                              e?.stopPropagation();
                              onDropdownOpen?.();
                            }}
                            onClose={(e) => {
                              e?.stopPropagation();
                              onDropdownClose?.();
                            }}
                            onMouseDown={(e) => e.stopPropagation()}
                            onClick={(e) => e.stopPropagation()}
                            sx={{ fontSize: '0.75rem' }}
                            MenuProps={{
                              disableScrollLock: true,
                              PaperProps: {
                                sx: {
                                  maxHeight: 200,
                                  overflow: 'auto',
                                  zIndex: 9999,
                                }
                              },
                              anchorOrigin: {
                                vertical: 'bottom',
                                horizontal: 'left',
                              },
                              transformOrigin: {
                                vertical: 'top',
                                horizontal: 'left',
                              }
                            }}
                          >
                            <MenuItem value="text">Plain Text</MenuItem>
                            <MenuItem value="structured">Structured Data</MenuItem>
                            <MenuItem value="context">Context Object</MenuItem>
                            <MenuItem value="full">Full Agent Response</MenuItem>
                          </Select>
                        </FormControl>

                      {/* Chain Key */}
                      <TextField
                        fullWidth
                        size="small"
                        label="Chain Key (optional)"
                        variant="outlined"
                        value={chainKey}
                        onChange={(e) => {
                          e.stopPropagation();
                          handleChainKeyChange(e.target.value);
                        }}
                        onMouseDown={(e) => e.stopPropagation()}
                        onClick={(e) => e.stopPropagation()}
                        sx={{ fontSize: '0.75rem' }}
                        placeholder="e.g., analysis_result, processed_data"
                        helperText="State key for agent chaining"
                        className="nodrag"
                      />
                    </Box>
                  )}
                </Box>
              </div>
              </div>
            </AccordionDetails>
          </Accordion>
        )}

        {/* Selected Tools Display */}
        <Box sx={{ mt: 1 }}>
          <Typography variant="caption" color="text.secondary" display="block" mb={0.5}>
            Selected Tools ({selectedTools.length})
          </Typography>
          <Box display="flex" flexWrap="wrap" gap={0.5} mb={1}>
            {selectedTools.map((tool) => {
              // Check if this tool is currently being used
              const isToolActive = executionData.tools_used?.some(usedTool => 
                usedTool.tool === tool
              ) && isExecuting;
              
              return (
                <div 
                  key={tool} 
                  className={`nodrag tool-indicator ${isToolActive ? 'tool-indicator--active' : ''}`}
                  onMouseDown={(e) => e.stopPropagation()}
                  onClick={(e) => e.stopPropagation()}
                >
                  <Chip
                    label={tool}
                    size="small"
                    variant="outlined"
                    onDelete={() => handleRemoveTool(tool)}
                    sx={{ 
                      fontSize: '0.625rem', 
                      height: 22,
                      bgcolor: isToolActive ? 'rgba(255, 152, 0, 0.1)' : 'inherit',
                      borderColor: isToolActive ? '#FF9800' : 'inherit',
                      '& .MuiChip-deleteIcon': {
                        fontSize: '0.75rem',
                        '&:hover': { color: 'error.main' }
                      }
                    }}
                  />
                </div>
              );
            })}
            {selectedTools.length === 0 && (
              <Typography variant="caption" color="text.disabled" sx={{ fontStyle: 'italic' }}>
                No tools selected
              </Typography>
            )}
          </Box>
        </Box>

        {/* Tools Selection - Collapsible */}
        {availableTools.length > 0 && (
          <Accordion 
            expanded={toolsExpanded}
            onChange={() => setToolsExpanded(!toolsExpanded)}
            sx={{ 
              boxShadow: 'none', 
              '&:before': { display: 'none' },
              zIndex: 50002,
              mt: 1,
              pointerEvents: 'auto'
            }}
          >
            <AccordionSummary
              expandIcon={<ExpandMoreIcon />}
              sx={{ 
                p: 0, 
                minHeight: 32, 
                '& .MuiAccordionSummary-content': { my: 0.5 },
                zIndex: 50003,
                pointerEvents: 'auto'
              }}
              className="nodrag"
              onMouseDown={(e) => e.stopPropagation()}
              onClick={() => {
                setBringToFront(true);
                setTimeout(() => setBringToFront(false), 5000);
              }}
            >
              <Typography variant="caption" color="text.secondary">
                Available Tools ({availableTools.length})
              </Typography>
            </AccordionSummary>
            <AccordionDetails 
              className="nodrag"
              sx={{ 
                p: 0,
                height: '250px',
                maxHeight: '250px',
                overflow: 'hidden',
                position: 'relative'
              }}
            >
              <div 
                style={{
                  height: '250px',
                  overflowY: 'scroll',
                  overflowX: 'hidden',
                  padding: '8px',
                  borderRadius: '4px',
                  border: '1px solid rgba(0,0,0,0.1)'
                }}
                className="accordion-scroll-container"
                onPointerDown={(e) => e.stopPropagation()}
                onWheel={(e) => {
                  // Stop propagation to prevent ReactFlow zoom
                  e.stopPropagation();
                  
                  // Manual scroll (avoid preventDefault for passive compatibility)
                  const container = e.currentTarget;
                  container.scrollTop += e.deltaY;
                }}
              >
                  <FormGroup>
                  {availableTools.map((tool, index) => {
                    const isSelected = selectedTools.includes(tool.name);
                    return (
                      <FormControlLabel
                        key={`${tool.name}-${index}`}
                        control={
                          <Checkbox
                            checked={isSelected}
                            onChange={(e) => {
                              if (e.target.checked && !isSelected) {
                                handleToolsChange(tool.name, true);
                              } else if (!e.target.checked && isSelected) {
                                handleToolsChange(tool.name, false);
                              }
                            }}
                            size="small"
                            sx={{ py: 0.5 }}
                          />
                        }
                        label={
                          <Box>
                            <Typography variant="caption" sx={{ fontWeight: 500, fontSize: '0.7rem' }}>
                              {tool.name}
                            </Typography>
                            {tool.description && (
                              <Typography variant="caption" color="text.secondary" display="block" sx={{ fontSize: '0.65rem' }}>
                                {tool.description.length > 40 ? `${tool.description.slice(0, 40)}...` : tool.description}
                              </Typography>
                            )}
                          </Box>
                        }
                        sx={{ alignItems: 'flex-start', m: 0, width: '100%' }}
                      />
                    );
                  })}
                  </FormGroup>
              </div>
            </AccordionDetails>
          </Accordion>
        )}
      </CardContent>
      
      {/* Target handles - can receive connections from any direction */}
      <Handle
        type="target"
        position={Position.Top}
        id="input-top"
        style={{ background: '#ab47bc', top: -4 }}
      />
      <Handle
        type="target"
        position={Position.Right}
        id="input-right"
        style={{ background: '#ab47bc', right: -4 }}
      />
      <Handle
        type="target"
        position={Position.Bottom}
        id="input-bottom"
        style={{ background: '#ab47bc', bottom: -4 }}
      />
      <Handle
        type="target"
        position={Position.Left}
        id="input-left"
        style={{ background: '#ab47bc', left: -4 }}
      />
      
      {/* Source handles - can send connections in any direction */}
      <Handle
        type="source"
        position={Position.Top}
        id="output-top"
        style={{ background: '#ab47bc', top: -4, opacity: 0.7 }}
      />
      <Handle
        type="source"
        position={Position.Right}
        id="output-right"
        style={{ background: '#ab47bc', right: -4, opacity: 0.7 }}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="output-bottom"
        style={{ background: '#ab47bc', bottom: -4, opacity: 0.7 }}
      />
      <Handle
        type="source"
        position={Position.Left}
        id="output-left"
        style={{ background: '#ab47bc', left: -4, opacity: 0.7 }}
      />
    </Card>
        </div>
      </Badge>

    {/* Input Content Expansion Dialog */}
    <Dialog 
      open={inputDialogOpen} 
      onClose={() => setInputDialogOpen(false)}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>
        <Box display="flex" alignItems="center" gap={1}>
          <InputIcon sx={{ color: '#2196F3' }} />
          <Typography variant="h6" sx={{ color: '#2196F3' }}>
            Agent Input
          </Typography>
          <IconButton 
            onClick={() => copyToClipboard(executionData.input || '')}
            size="small"
            sx={{ ml: 'auto' }}
          >
            <CopyIcon />
          </IconButton>
        </Box>
      </DialogTitle>
      <DialogContent>
        {(() => {
          const parsed = parseAgentInput(executionData.input || '', executionData.input_sections);
          
          if (!executionData.input) {
            return (
              <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                No input data available
              </Typography>
            );
          }
          
          return (
            <Box>
              {/* State Management Info */}
              {parsed.hasWorkflowDependencies && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" sx={{ 
                    color: '#4CAF50', 
                    fontWeight: 600, 
                    mb: 1,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 0.5
                  }}>
                    🔗 State Management
                  </Typography>
                  <Box sx={{ 
                    backgroundColor: 'rgba(76, 175, 80, 0.08)',
                    p: 1.5,
                    borderRadius: 1,
                    border: '1px solid rgba(76, 175, 80, 0.3)',
                    mb: 2
                  }}>
                    <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>
                      Operation: {data.stateOperation || 'passthrough'} → Format: {data.outputFormat || 'text'}
                    </Typography>
                  </Box>
                </Box>
              )}
              
              {/* Dependency Results - Most Important */}
              {parsed.dependencyResults && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" sx={{ 
                    color: '#2196F3', 
                    fontWeight: 600, 
                    mb: 1,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 0.5
                  }}>
                    📥 Input from Previous Agents
                  </Typography>
                  <Box sx={{ 
                    backgroundColor: 'rgba(33, 150, 243, 0.08)',
                    p: 2,
                    borderRadius: 1,
                    border: '1px solid rgba(33, 150, 243, 0.2)',
                    maxHeight: 300,
                    overflow: 'auto'
                  }}>
                    <MarkdownContent content={parsed.dependencyResults} />
                  </Box>
                </Box>
              )}
              
              {/* User Request */}
              {parsed.userRequest && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" sx={{ 
                    color: '#FF9800', 
                    fontWeight: 600, 
                    mb: 1,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 0.5
                  }}>
                    📝 Original Request
                  </Typography>
                  <Box sx={{ 
                    backgroundColor: 'rgba(255, 152, 0, 0.08)',
                    p: 2,
                    borderRadius: 1,
                    border: '1px solid rgba(255, 152, 0, 0.2)'
                  }}>
                    <MarkdownContent content={parsed.userRequest} />
                  </Box>
                </Box>
              )}
              
              {/* Agent Role */}
              {parsed.role && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" sx={{ 
                    color: '#9C27B0', 
                    fontWeight: 600, 
                    mb: 1,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 0.5
                  }}>
                    👤 Agent Role
                  </Typography>
                  <Box sx={{ 
                    backgroundColor: 'rgba(156, 39, 176, 0.08)',
                    p: 1.5,
                    borderRadius: 1,
                    border: '1px solid rgba(156, 39, 176, 0.2)'
                  }}>
                    <Typography variant="body2">{parsed.role}</Typography>
                  </Box>
                </Box>
              )}
              
              {/* Full Raw Content - Collapsible */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle2">Full Raw Input</Typography>
                </AccordionSummary>
                <AccordionDetails onWheel={(e) => e.stopPropagation()}>
                  <Box sx={{ 
                    backgroundColor: 'rgba(0, 0, 0, 0.03)',
                    p: 2,
                    borderRadius: 1,
                    border: '1px solid rgba(0, 0, 0, 0.1)',
                    '& .markdown-content': {
                      '& p:last-child': { mb: 0 }
                    }
                  }}>
                    <MarkdownContent content={executionData.input} />
                  </Box>
                </AccordionDetails>
              </Accordion>
            </Box>
          );
        })()}
        
        {executionData.timestamp && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
            Executed at: {new Date(executionData.timestamp).toLocaleString()}
          </Typography>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setInputDialogOpen(false)}>Close</Button>
      </DialogActions>
    </Dialog>

    {/* Output Content Expansion Dialog */}
    <Dialog 
      open={outputDialogOpen} 
      onClose={() => setOutputDialogOpen(false)}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>
        <Box display="flex" alignItems="center" gap={1}>
          <OutputIcon sx={{ color: '#4CAF50' }} />
          <Typography variant="h6" sx={{ color: '#4CAF50' }}>
            Agent Output
          </Typography>
          <IconButton 
            onClick={() => {
              const { thinking, output } = parseOutputContent(executionData.output || '');
              const fullContent = thinking ? `## Thinking Process\n\n${thinking}\n\n## Response\n\n${output}` : output;
              copyToClipboard(fullContent);
            }}
            size="small"
            sx={{ ml: 'auto' }}
          >
            <CopyIcon />
          </IconButton>
        </Box>
      </DialogTitle>
      <DialogContent>
        {(() => {
          const { thinking, output } = parseOutputContent(executionData.output || '');
          
          return (
            <>
              {/* Thinking Section */}
              {thinking && (
                <Box sx={{ mb: 3 }}>
                  <Typography variant="h6" sx={{ 
                    mb: 2, 
                    color: '#9C27B0',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1
                  }}>
                    🤔 Agent Thinking Process
                  </Typography>
                  <Box sx={{ 
                    backgroundColor: 'rgba(156, 39, 176, 0.08)',
                    p: 2,
                    borderRadius: 1,
                    border: '1px solid rgba(156, 39, 176, 0.2)',
                    '& .markdown-content': {
                      '& p:last-child': { mb: 0 }
                    }
                  }}>
                    <MarkdownContent content={thinking} />
                  </Box>
                </Box>
              )}

              {/* Output Section */}
              <Box sx={{ mb: 2 }}>
                <Typography variant="h6" sx={{ 
                  mb: 2, 
                  color: '#4CAF50',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1
                }}>
                  💬 Agent Response
                </Typography>
                <Box sx={{ 
                  backgroundColor: 'rgba(76, 175, 80, 0.08)',
                  p: 2,
                  borderRadius: 1,
                  border: '1px solid rgba(76, 175, 80, 0.2)',
                  '& .markdown-content': {
                    '& p:last-child': { mb: 0 }
                  }
                }}>
                  {output ? (
                    <MarkdownContent content={output} />
                  ) : (
                    <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                      No output data available
                    </Typography>
                  )}
                </Box>
              </Box>
            </>
          );
        })()}

        {/* Tools Used Section */}
        {executionData.tools_used && executionData.tools_used.length > 0 && (
          <Box>
            <Typography variant="subtitle2" sx={{ mb: 1, color: '#FF9800' }}>
              Tools Used ({executionData.tools_used.length})
            </Typography>
            {executionData.tools_used.map((tool, index) => (
              <Box key={index} sx={{ 
                mb: 1, 
                p: 1, 
                backgroundColor: 'rgba(255, 152, 0, 0.08)',
                borderRadius: 1,
                border: '1px solid rgba(255, 152, 0, 0.2)'
              }}>
                <Typography variant="body2" sx={{ fontWeight: 600, color: '#FF9800' }}>
                  {tool.tool} ({tool.duration}ms)
                </Typography>
                {tool.input && (
                  <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                    <strong>Input:</strong> {JSON.stringify(tool.input)}
                  </Typography>
                )}
                {tool.output && (
                  <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                    <strong>Output:</strong> {JSON.stringify(tool.output)}
                  </Typography>
                )}
              </Box>
            ))}
          </Box>
        )}

        {/* Error Section */}
        {executionData.error && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" sx={{ mb: 1, color: '#F44336' }}>
              Error Details
            </Typography>
            <Typography variant="body2" sx={{ 
              whiteSpace: 'pre-wrap', 
              wordBreak: 'break-word',
              fontFamily: 'monospace',
              backgroundColor: 'rgba(244, 67, 54, 0.08)',
              p: 2,
              borderRadius: 1,
              border: '1px solid rgba(244, 67, 54, 0.2)',
              color: '#F44336'
            }}>
              {executionData.error}
            </Typography>
          </Box>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setOutputDialogOpen(false)}>Close</Button>
      </DialogActions>
    </Dialog>
    </>
  );
};

export default AgentNode;