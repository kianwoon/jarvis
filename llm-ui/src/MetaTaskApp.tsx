import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Paper,
  Container,
  ThemeProvider,
  createTheme,
  CssBaseline,
  Tabs,
  Tab,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Chip,
  Alert,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextareaAutosize,
  CircularProgress,
  Skeleton,
  Snackbar,
  LinearProgress,
  Backdrop
} from '@mui/material';
import {
  LightMode as LightModeIcon,
  DarkMode as DarkModeIcon,
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  PlayArrow as PlayIcon,
  Description as TemplateIcon,
  AccountTree as WorkflowIcon,
  ContentCopy as DuplicateIcon,
  Refresh as RefreshIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { metaTaskAPI, MetaTaskTemplate, MetaTaskWorkflow, ExecutionEvent } from './services/metaTaskApi';
import NavigationBar from './components/shared/NavigationBar';


interface CreateTemplateData {
  name: string;
  description: string;
  template_type: string;
  template_config: {
    phases: Array<{
      name: string;
      type: string;
      prompt?: string;
      model?: string;
      config?: Record<string, any>;
    }>;
    variables?: string[];
    output_format?: string;
  };
  input_schema?: Record<string, any>;
  output_schema?: Record<string, any>;
  default_settings?: Record<string, any>;
}

interface WorkflowExecution {
  workflowId: string;
  status: 'idle' | 'running' | 'completed' | 'failed';
  progress: number;
  currentPhase?: string;
  events: ExecutionEvent[];
  eventSource?: EventSource;
}


function MetaTaskApp() {
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('jarvis-dark-mode');
    return saved ? JSON.parse(saved) : false;
  });

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState<string>('');
  
  // Templates and Workflows management
  const [templates, setTemplates] = useState<MetaTaskTemplate[]>([]);
  const [workflows, setWorkflows] = useState<MetaTaskWorkflow[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<MetaTaskTemplate | null>(null);
  const [selectedWorkflow, setSelectedWorkflow] = useState<MetaTaskWorkflow | null>(null);
  const [executions, setExecutions] = useState<Map<string, WorkflowExecution>>(new Map());
  const [refreshKey, setRefreshKey] = useState(0);
  
  // Dialog states
  const [createWorkflowOpen, setCreateWorkflowOpen] = useState(false);
  const [editWorkflowOpen, setEditWorkflowOpen] = useState(false);
  const [editTemplateOpen, setEditTemplateOpen] = useState(false);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);
  const [selectedTemplateForEdit, setSelectedTemplateForEdit] = useState<MetaTaskTemplate | null>(null);
  const [selectedWorkflowForEdit, setSelectedWorkflowForEdit] = useState<MetaTaskWorkflow | null>(null);
  const [itemToDelete, setItemToDelete] = useState<{ type: 'template' | 'workflow'; id: string; name: string } | null>(null);
  const [executionDialogOpen, setExecutionDialogOpen] = useState(false);
  const [currentExecution, setCurrentExecution] = useState<WorkflowExecution | null>(null);
  
  const [newTemplate, setNewTemplate] = useState<CreateTemplateData>({
    name: '',
    description: '',
    template_type: 'document_generation',
    template_config: {
      phases: [],
      variables: [],
      output_format: 'markdown'
    },
    default_settings: {
      enabled: true,
      analyzer_model: {
        model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
        temperature: 0.7,
        max_tokens: 4096,
        top_p: 0.9,
        model_server: 'http://localhost:11434',
        system_prompt: 'You are an expert document analyzer. Analyze the given content and extract key information, requirements, and context for document generation.'
      },
      generator_model: {
        model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
        temperature: 0.7,
        max_tokens: 4096,
        top_p: 0.9,
        model_server: 'http://localhost:11434',
        system_prompt: 'You are an expert document generator. Create high-quality, well-structured documents based on the provided requirements and context.'
      },
      reviewer_model: {
        model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
        temperature: 0.3,
        max_tokens: 4096,
        top_p: 0.9,
        model_server: 'http://localhost:11434',
        system_prompt: 'You are an expert document reviewer. Review the generated content for quality, accuracy, completeness, and adherence to requirements.'
      },
      assembler_model: {
        model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
        temperature: 0.5,
        max_tokens: 4096,
        top_p: 0.9,
        model_server: 'http://localhost:11434',
        system_prompt: 'You are an expert document assembler. Combine and structure the reviewed content into a final, polished document.'
      },
      execution: {
        max_phases: 10,
        phase_timeout_minutes: 30,
        retry_attempts: 3,
        enable_streaming: true,
        parallel_phases: false,
        checkpoint_interval: 5,
        error_handling: 'retry'
      },
      quality_control: {
        quality_check_model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
        factuality_threshold: 0.8,
        coherence_threshold: 0.8,
        completeness_threshold: 0.8,
        auto_retry_on_low_quality: true,
        max_refinement_iterations: 3,
        enable_cross_validation: false
      },
      output: {
        default_format: 'markdown',
        include_metadata: true,
        max_output_size_mb: 10,
        output_validation: true,
        output_deduplication: false,
        include_phase_outputs: false,
        compress_large_outputs: true,
        enable_structured_output: true
      },
      caching: {
        cache_templates: true,
        cache_workflows: true,
        cache_ttl_hours: 72,
        cache_max_size_mb: 500,
        enable_distributed_cache: false,
        cache_eviction_policy: 'LRU',
        cache_compression: true
      },
      advanced: {
        log_level: 'INFO',
        enable_metrics: true,
        enable_tracing: false,
        telemetry_endpoint: '',
        performance_profiling: false,
        debug_mode: false,
        retry_backoff_strategy: 'exponential'
      }
    }
  });
  
  const [newWorkflow, setNewWorkflow] = useState<{
    templateId: string;
    name: string;
    description: string;
    inputData: Record<string, any>;
  }>({
    templateId: '',
    name: '',
    description: '',
    inputData: {}
  });
  
  // Active tab for Templates/Workflows
  const [activeTab, setActiveTab] = useState(0);

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

  const toggleDarkMode = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem('jarvis-dark-mode', JSON.stringify(newDarkMode));
  };

  useEffect(() => {
    loadTemplates();
    loadWorkflows();
  }, [refreshKey]);

  // Load templates
  const loadTemplates = useCallback(async () => {
    try {
      setLoading(true);
      setError('');
      const templatesData = await metaTaskAPI.getTemplates();
      setTemplates(templatesData);
    } catch (err) {
      console.error('Failed to load templates:', err);
      setError('Failed to load templates. Please check your connection and try again.');
      // Set empty array on error to prevent undefined issues
      setTemplates([]);
    } finally {
      setLoading(false);
    }
  }, []);

  // Load workflows
  const loadWorkflows = useCallback(async () => {
    try {
      const workflowsData = await metaTaskAPI.getWorkflows();
      setWorkflows(workflowsData);
    } catch (err) {
      console.error('Error loading workflows:', err);
      setWorkflows([]);
    }
  }, []);

  // Create new workflow
  const createWorkflow = async () => {
    try {
      if (!newWorkflow.templateId || !newWorkflow.name) {
        setError('Please select a template and provide a name');
        return;
      }
      
      setLoading(true);
      const workflow = await metaTaskAPI.createWorkflow(
        newWorkflow.templateId,
        newWorkflow.name,
        newWorkflow.description,
        newWorkflow.inputData
      );
      
      setSuccess('Workflow created successfully!');
      setCreateWorkflowOpen(false);
      setNewWorkflow({
        templateId: '',
        name: '',
        description: '',
        inputData: {}
      });
      
      // Add the new workflow to the list
      setWorkflows(prev => [...prev, workflow]);
      
      // Optionally, start execution immediately
      await runWorkflow(workflow.id);
    } catch (err: any) {
      setError(err.message || 'Failed to create workflow');
    } finally {
      setLoading(false);
    }
  };

  // Handle edit workflow
  const handleEditWorkflow = (workflow: MetaTaskWorkflow) => {
    if (workflow.status !== 'pending') {
      setError('Only pending workflows can be edited');
      return;
    }

    setSelectedWorkflowForEdit(workflow);
    setNewWorkflow({
      templateId: workflow.template_id,
      name: workflow.name,
      description: workflow.description || '',
      inputData: workflow.input_data || {}
    });
    setEditWorkflowOpen(true);
  };

  // Update workflow
  const updateWorkflow = async () => {
    if (!selectedWorkflowForEdit) return;
    
    try {
      setLoading(true);
      setError('');
      
      const updatedWorkflow = await metaTaskAPI.updateWorkflow(selectedWorkflowForEdit.id, {
        name: newWorkflow.name,
        description: newWorkflow.description,
        input_data: newWorkflow.inputData
      });
      
      // Update the workflow in the local state
      setWorkflows(prev => 
        prev.map(workflow => 
          workflow.id === selectedWorkflowForEdit.id ? updatedWorkflow : workflow
        )
      );
      
      setSuccess('Workflow updated successfully!');
      setEditWorkflowOpen(false);
      setSelectedWorkflowForEdit(null);
      setNewWorkflow({
        templateId: '',
        name: '',
        description: '',
        inputData: {}
      });
      
    } catch (err: any) {
      setError(err.message || 'Failed to update workflow');
    } finally {
      setLoading(false);
    }
  };

  // Run workflow
  const runWorkflow = async (workflowId: string) => {
    try {
      // Initialize execution state
      const execution: WorkflowExecution = {
        workflowId,
        status: 'running',
        progress: 0,
        events: []
      };
      
      setExecutions(prev => new Map(prev).set(workflowId, execution));
      setCurrentExecution(execution);
      setExecutionDialogOpen(true);
      
      // Start SSE connection for real-time updates
      const eventSource = metaTaskAPI.executeWorkflow(workflowId, (event: ExecutionEvent) => {
        setExecutions(prev => {
          const updated = new Map(prev);
          const exec = updated.get(workflowId);
          if (exec) {
            exec.events.push(event);
            
            // Update status based on event type
            switch (event.type) {
              case 'phase_start':
                exec.currentPhase = event.phase;
                break;
              case 'phase_complete':
                exec.progress = event.progress || exec.progress;
                break;
              case 'workflow_complete':
                exec.status = 'completed';
                exec.progress = 100;
                break;
              case 'workflow_error':
              case 'phase_error':
                exec.status = 'failed';
                break;
              case 'progress':
                exec.progress = event.progress || exec.progress;
                break;
            }
            
            // Update current execution if it's the active one
            if (currentExecution?.workflowId === workflowId) {
              setCurrentExecution({ ...exec });
            }
          }
          return updated;
        });
      });
      
      execution.eventSource = eventSource;
      setSuccess('Workflow started successfully!');
    } catch (err: any) {
      setError(err.message || 'Failed to run workflow');
      
      // Update execution status to failed
      setExecutions(prev => {
        const updated = new Map(prev);
        const exec = updated.get(workflowId);
        if (exec) {
          exec.status = 'failed';
        }
        return updated;
      });
    }
  };

  // Handle edit template
  const handleEditTemplate = (template: MetaTaskTemplate) => {
    setSelectedTemplateForEdit(template);
    setIsEditMode(true);
    
    // Safely populate form with existing template data
    const templateConfig = template.template_config as any;
    // Transform phases from backend (description) to UI (prompt) for editing
    const phases = (templateConfig?.phases || []).map(phase => ({
      ...phase,
      prompt: phase.prompt || phase.description || '',  // Map prompt back to UI, fallback to description
    }));
    
    // Merge existing default_settings with defaults to ensure all required fields are present
    const defaultSettingsBase = {
      enabled: true,
      analyzer_model: {
        model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
        temperature: 0.7,
        max_tokens: 4096,
        top_p: 0.9,
        model_server: 'http://localhost:11434',
        system_prompt: 'You are an expert document analyzer. Analyze the given content and extract key information, requirements, and context for document generation.'
      },
      generator_model: {
        model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
        temperature: 0.7,
        max_tokens: 4096,
        top_p: 0.9,
        model_server: 'http://localhost:11434',
        system_prompt: 'You are an expert document generator. Create high-quality, well-structured documents based on the provided requirements and context.'
      },
      reviewer_model: {
        model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
        temperature: 0.3,
        max_tokens: 4096,
        top_p: 0.9,
        model_server: 'http://localhost:11434',
        system_prompt: 'You are an expert document reviewer. Review the generated content for quality, accuracy, completeness, and adherence to requirements.'
      },
      assembler_model: {
        model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
        temperature: 0.5,
        max_tokens: 4096,
        top_p: 0.9,
        model_server: 'http://localhost:11434',
        system_prompt: 'You are an expert document assembler. Combine and structure the reviewed content into a final, polished document.'
      },
      execution: {
        max_phases: 10,
        phase_timeout_minutes: 30,
        retry_attempts: 3,
        enable_streaming: true,
        parallel_phases: false,
        checkpoint_interval: 5,
        error_handling: 'retry'
      },
      quality_control: {
        quality_check_model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
        factuality_threshold: 0.8,
        coherence_threshold: 0.8,
        completeness_threshold: 0.8,
        auto_retry_on_low_quality: true,
        max_refinement_iterations: 3,
        enable_cross_validation: false
      },
      output: {
        default_format: 'markdown',
        include_metadata: true,
        max_output_size_mb: 10,
        output_validation: true,
        output_deduplication: false,
        include_phase_outputs: false,
        compress_large_outputs: true,
        enable_structured_output: true
      },
      caching: {
        cache_templates: true,
        cache_workflows: true,
        cache_ttl_hours: 72,
        cache_max_size_mb: 500,
        enable_distributed_cache: false,
        cache_eviction_policy: 'LRU',
        cache_compression: true
      },
      advanced: {
        log_level: 'INFO',
        enable_metrics: true,
        enable_tracing: false,
        telemetry_endpoint: '',
        performance_profiling: false,
        debug_mode: false,
        retry_backoff_strategy: 'exponential'
      }
    };

    // Deep merge existing settings with defaults to preserve existing values while ensuring all required fields exist
    const mergeDeep = (target: any, source: any): any => {
      const result = { ...target };
      for (const key in source) {
        if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
          result[key] = mergeDeep(result[key] || {}, source[key]);
        } else if (result[key] === undefined) {
          result[key] = source[key];
        }
      }
      return result;
    };

    const mergedDefaultSettings = mergeDeep(template.default_settings || {}, defaultSettingsBase);

    setNewTemplate({
      name: template.name,
      description: template.description || '',
      template_type: template.template_type,
      template_config: {
        phases,
        variables: templateConfig?.variables || [],
        output_format: templateConfig?.output_format || 'markdown'
      },
      input_schema: template.input_schema,
      output_schema: template.output_schema,
      default_settings: mergedDefaultSettings
    });
    
    setEditTemplateOpen(true);
  };


  // Create new template
  const createTemplate = async () => {
    try {
      setLoading(true);
      setError(''); // Clear any previous errors
      
      // Transform phases before sending to API: map prompt to description
      const transformedPhases = newTemplate.template_config.phases.map(phase => ({
        ...phase,
        description: phase.prompt || '',  // Map prompt to description
        // Keep prompt field if it exists for backward compatibility
      }));
      
      const createdTemplate = await metaTaskAPI.createTemplate({
        name: newTemplate.name,
        description: newTemplate.description,
        template_type: newTemplate.template_type,
        template_config: {
          ...newTemplate.template_config,
          phases: transformedPhases
        },
        input_schema: newTemplate.input_schema,
        output_schema: newTemplate.output_schema,
        default_settings: newTemplate.default_settings
      });
      
      // Immediately add the new template to the local state
      setTemplates(prevTemplates => [...prevTemplates, createdTemplate]);
      
      setSuccess('Template created successfully!');
      resetTemplateForm();
      
    } catch (err: any) {
      setError(err.message || 'Failed to create template');
    } finally {
      setLoading(false);
    }
  };

  // Update template
  const updateTemplate = async () => {
    if (!selectedTemplateForEdit) return;
    
    try {
      setLoading(true);
      setError(''); // Clear any previous errors
      
      // Transform phases before sending to API: map prompt to description
      const transformedPhases = newTemplate.template_config.phases.map(phase => ({
        ...phase,
        description: phase.prompt || '',  // Map prompt to description
      }));
      
      // Call the API and get the updated template back
      const updatedTemplate = await metaTaskAPI.updateTemplate(selectedTemplateForEdit.id, {
        name: newTemplate.name,
        description: newTemplate.description,
        template_type: newTemplate.template_type,
        template_config: {
          ...newTemplate.template_config,
          phases: transformedPhases
        },
        input_schema: newTemplate.input_schema,
        output_schema: newTemplate.output_schema,
        default_settings: newTemplate.default_settings
      });
      
      // Transform the phases back for UI display
      const transformedTemplate = {
        ...updatedTemplate,
        template_config: {
          ...updatedTemplate.template_config,
          phases: (updatedTemplate.template_config?.phases || []).map(phase => ({
            ...phase,
            prompt: phase.prompt || phase.description || ''
          }))
        }
      };
      
      // Immediately update the local templates state with the transformed data
      setTemplates(prevTemplates => 
        prevTemplates.map(template => 
          template.id === selectedTemplateForEdit.id ? transformedTemplate : template
        )
      );
      
      // Show success message
      setSuccess('Template updated successfully!');
      
      // Close dialog and reset form
      resetTemplateForm();
      
    } catch (err: any) {
      setError(err.message || 'Failed to update template');
    } finally {
      setLoading(false);
    }
  };

  // Reset template form
  const resetTemplateForm = () => {
    setEditTemplateOpen(false);
    setIsEditMode(false);
    setSelectedTemplateForEdit(null);
    
    // Reset form
    setNewTemplate({
      name: '',
      description: '',
      template_type: 'document_generation',
      template_config: {
        phases: [],
        variables: [],
        output_format: 'markdown'
      },
      default_settings: {
        enabled: true,
        analyzer_model: {
          model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
          temperature: 0.7,
          max_tokens: 4096,
          top_p: 0.9,
          model_server: 'http://localhost:11434',
          system_prompt: 'You are an expert document analyzer. Analyze the given content and extract key information, requirements, and context for document generation.'
        },
        generator_model: {
          model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
          temperature: 0.7,
          max_tokens: 4096,
          top_p: 0.9,
          model_server: 'http://localhost:11434',
          system_prompt: 'You are an expert document generator. Create high-quality, well-structured documents based on the provided requirements and context.'
        },
        reviewer_model: {
          model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
          temperature: 0.3,
          max_tokens: 4096,
          top_p: 0.9,
          model_server: 'http://localhost:11434',
          system_prompt: 'You are an expert document reviewer. Review the generated content for quality, accuracy, completeness, and adherence to requirements.'
        },
        assembler_model: {
          model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
          temperature: 0.5,
          max_tokens: 4096,
          top_p: 0.9,
          model_server: 'http://localhost:11434',
          system_prompt: 'You are an expert document assembler. Combine and structure the reviewed content into a final, polished document.'
        },
        execution: {
          max_phases: 10,
          phase_timeout_minutes: 30,
          retry_attempts: 3,
          enable_streaming: true,
          parallel_phases: false,
          checkpoint_interval: 5,
          error_handling: 'retry'
        },
        quality_control: {
          quality_check_model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
          factuality_threshold: 0.8,
          coherence_threshold: 0.8,
          completeness_threshold: 0.8,
          auto_retry_on_low_quality: true,
          max_refinement_iterations: 3,
          enable_cross_validation: false
        },
        output: {
          default_format: 'markdown',
          include_metadata: true,
          max_output_size_mb: 10,
          output_validation: true,
          output_deduplication: false,
          include_phase_outputs: false,
          compress_large_outputs: true,
          enable_structured_output: true
        },
        caching: {
          cache_templates: true,
          cache_workflows: true,
          cache_ttl_hours: 72,
          cache_max_size_mb: 500,
          enable_distributed_cache: false,
          cache_eviction_policy: 'LRU',
          cache_compression: true
        },
        advanced: {
          log_level: 'INFO',
          enable_metrics: true,
          enable_tracing: false,
          telemetry_endpoint: '',
          performance_profiling: false,
          debug_mode: false,
          retry_backoff_strategy: 'exponential'
        }
      }
    });
  };
  
  // Delete template
  const deleteTemplate = async (templateId: string) => {
    try {
      setLoading(true);
      setError(''); // Clear any previous errors
      
      await metaTaskAPI.deleteTemplate(templateId);
      
      // Immediately remove the deleted template from local state
      setTemplates(prevTemplates => prevTemplates.filter(template => template.id !== templateId));
      
      setSuccess('Template deleted successfully!');
      setDeleteConfirmOpen(false);
      setItemToDelete(null);
      
    } catch (err: any) {
      setError(err.message || 'Failed to delete template');
    } finally {
      setLoading(false);
    }
  };

  // Delete workflow
  const deleteWorkflow = async (workflowId: string) => {
    try {
      // Close any active execution event source
      const execution = executions.get(workflowId);
      if (execution?.eventSource) {
        execution.eventSource.close();
      }
      
      // Remove from executions map
      setExecutions(prev => {
        const updated = new Map(prev);
        updated.delete(workflowId);
        return updated;
      });
      
      // Remove from workflows list
      setWorkflows(prev => prev.filter(w => w.id !== workflowId));
      
      setSuccess('Workflow deleted successfully!');
      setDeleteConfirmOpen(false);
      setItemToDelete(null);
    } catch (err: any) {
      setError(err.message || 'Failed to delete workflow');
    }
  };
  
  // Handle delete confirmation
  const handleDeleteClick = (type: 'template' | 'workflow', id: string, name: string) => {
    setItemToDelete({ type, id, name });
    setDeleteConfirmOpen(true);
  };
  
  const handleDeleteConfirm = async () => {
    if (!itemToDelete) return;
    
    if (itemToDelete.type === 'template') {
      await deleteTemplate(itemToDelete.id);
    } else {
      await deleteWorkflow(itemToDelete.id);
    }
  };
  
  // Format template variables for display
  const getTemplateVariables = (template: MetaTaskTemplate): string[] => {
    const config = template.template_config as any;
    const variables = config.variables || [];
    
    // Handle both string and object formats
    return variables.map((v: any) => 
      typeof v === 'string' ? v : (v.name || v)
    );
  };
  
  // Get template category with proper display formatting
  const getTemplateCategory = (template: MetaTaskTemplate): string => {
    const typeMap: Record<string, string> = {
      'document_generation': 'Document Generation',
      'document': 'Document',
      'report': 'Report',
      'analysis': 'Analysis',
      'summary': 'Summary',
      'technical': 'Technical',
      'business': 'Business'
    };
    
    const templateType = template.template_type || 'document_generation';
    return typeMap[templateType] || templateType.charAt(0).toUpperCase() + templateType.slice(1);
  };


  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <Box sx={{ 
          position: 'sticky', 
          top: 0, 
          zIndex: 1100, 
          bgcolor: 'background.default',
          boxShadow: 1
        }}>
          <AppBar position="static">
            <Toolbar>
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                Jarvis AI Assistant - Meta-Task Templates & Workflows
              </Typography>
              <IconButton onClick={toggleDarkMode} color="inherit">
                {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
              </IconButton>
            </Toolbar>
          </AppBar>

          {/* Navigation Tabs */}
          <NavigationBar currentTab={3} />
        </Box>

        {/* Main Content */}
        <Container maxWidth={false} sx={{ flex: 1, py: 2, overflow: 'hidden', width: '100%' }}>
          <Paper sx={{ height: '100%', overflow: 'auto', p: 3 }}>
            <Box>
              {/* Header */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="h4" gutterBottom>
                  Meta-Task Templates & Workflows
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  Manage document templates and create automated workflows for complex document generation
                </Typography>
              </Box>

              {error && (
                <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
                  {error}
                </Alert>
              )}

              {success && (
                <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess('')}>
                  {success}
                </Alert>
              )}

              {/* Content Tabs */}
              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs
                  value={activeTab}
                  onChange={(_, newValue) => setActiveTab(newValue)}
                  centered
                >
                  <Tab label="Templates" icon={<TemplateIcon />} iconPosition="start" />
                  <Tab label="Workflows" icon={<WorkflowIcon />} iconPosition="start" />
                </Tabs>
              </Box>

              {/* Templates Tab */}
              <Box hidden={activeTab !== 0} sx={{ pt: 3 }}>
                <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="h5">Document Templates</Typography>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button
                      variant="outlined"
                      startIcon={<RefreshIcon />}
                      onClick={() => setRefreshKey(prev => prev + 1)}
                      disabled={loading}
                    >
                      Refresh
                    </Button>
                    <Button
                      variant="contained"
                      startIcon={<AddIcon />}
                      onClick={() => {
                        setIsEditMode(false);
                        setSelectedTemplateForEdit(null);
                        setNewTemplate({
                          name: '',
                          description: '',
                          template_type: 'document_generation',
                          template_config: {
                            phases: [],
                            variables: [],
                            output_format: 'markdown'
                          },
                          default_settings: {
                            enabled: true,
                            analyzer_model: {
                              model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
                              temperature: 0.7,
                              max_tokens: 4096,
                              top_p: 0.9,
                              model_server: 'http://localhost:11434',
                              system_prompt: 'You are an expert document analyzer. Analyze the given content and extract key information, requirements, and context for document generation.'
                            },
                            generator_model: {
                              model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
                              temperature: 0.7,
                              max_tokens: 4096,
                              top_p: 0.9,
                              model_server: 'http://localhost:11434',
                              system_prompt: 'You are an expert document generator. Create high-quality, well-structured documents based on the provided requirements and context.'
                            },
                            reviewer_model: {
                              model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
                              temperature: 0.3,
                              max_tokens: 4096,
                              top_p: 0.9,
                              model_server: 'http://localhost:11434',
                              system_prompt: 'You are an expert document reviewer. Review the generated content for quality, accuracy, completeness, and adherence to requirements.'
                            },
                            assembler_model: {
                              model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
                              temperature: 0.5,
                              max_tokens: 4096,
                              top_p: 0.9,
                              model_server: 'http://localhost:11434',
                              system_prompt: 'You are an expert document assembler. Combine and structure the reviewed content into a final, polished document.'
                            },
                            execution: {
                              max_phases: 10,
                              phase_timeout_minutes: 30,
                              retry_attempts: 3,
                              enable_streaming: true,
                              parallel_phases: false,
                              checkpoint_interval: 5,
                              error_handling: 'retry'
                            },
                            quality_control: {
                              quality_check_model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
                              factuality_threshold: 0.8,
                              coherence_threshold: 0.8,
                              completeness_threshold: 0.8,
                              auto_retry_on_low_quality: true,
                              max_refinement_iterations: 3,
                              enable_cross_validation: false
                            },
                            output: {
                              default_format: 'markdown',
                              include_metadata: true,
                              max_output_size_mb: 10,
                              output_validation: true,
                              output_deduplication: false,
                              include_phase_outputs: false,
                              compress_large_outputs: true,
                              enable_structured_output: true
                            },
                            caching: {
                              cache_templates: true,
                              cache_workflows: true,
                              cache_ttl_hours: 72,
                              cache_max_size_mb: 500,
                              enable_distributed_cache: false,
                              cache_eviction_policy: 'LRU',
                              cache_compression: true
                            },
                            advanced: {
                              log_level: 'INFO',
                              enable_metrics: true,
                              enable_tracing: false,
                              telemetry_endpoint: '',
                              performance_profiling: false,
                              debug_mode: false,
                              retry_backoff_strategy: 'exponential'
                            }
                          }
                        });
                        setEditTemplateOpen(true);
                      }}
                      disabled={loading}
                    >
                      Create Template
                    </Button>
                  </Box>
                </Box>

                {loading && templates.length === 0 ? (
                  <Grid container spacing={2}>
                    {[1, 2, 3].map((i) => (
                      <Grid item xs={12} md={6} lg={4} key={i}>
                        <Card>
                          <CardContent>
                            <Skeleton variant="text" width="60%" height={32} />
                            <Skeleton variant="text" width="30%" height={20} sx={{ mb: 1 }} />
                            <Skeleton variant="text" width="100%" height={60} />
                            <Skeleton variant="text" width="80%" height={20} />
                          </CardContent>
                          <CardActions>
                            <Skeleton variant="rectangular" width={60} height={30} />
                            <Skeleton variant="rectangular" width={80} height={30} />
                            <Skeleton variant="rectangular" width={60} height={30} />
                          </CardActions>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                ) : templates.length === 0 ? (
                  <Paper sx={{ p: 4, textAlign: 'center' }}>
                    <TemplateIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                    <Typography variant="h6" gutterBottom>
                      No Templates Found
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                      Create your first template to start generating documents
                    </Typography>
                    <Button
                      variant="contained"
                      startIcon={<AddIcon />}
                      onClick={() => {
                        setIsEditMode(false);
                        setSelectedTemplateForEdit(null);
                        setNewTemplate({
                          name: '',
                          description: '',
                          template_type: 'document_generation',
                          template_config: {
                            phases: [],
                            variables: [],
                            output_format: 'markdown'
                          },
                          default_settings: {
                            enabled: true,
                            analyzer_model: {
                              model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
                              temperature: 0.7,
                              max_tokens: 4096,
                              top_p: 0.9,
                              model_server: 'http://localhost:11434',
                              system_prompt: 'You are an expert document analyzer. Analyze the given content and extract key information, requirements, and context for document generation.'
                            },
                            generator_model: {
                              model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
                              temperature: 0.7,
                              max_tokens: 4096,
                              top_p: 0.9,
                              model_server: 'http://localhost:11434',
                              system_prompt: 'You are an expert document generator. Create high-quality, well-structured documents based on the provided requirements and context.'
                            },
                            reviewer_model: {
                              model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
                              temperature: 0.3,
                              max_tokens: 4096,
                              top_p: 0.9,
                              model_server: 'http://localhost:11434',
                              system_prompt: 'You are an expert document reviewer. Review the generated content for quality, accuracy, completeness, and adherence to requirements.'
                            },
                            assembler_model: {
                              model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
                              temperature: 0.5,
                              max_tokens: 4096,
                              top_p: 0.9,
                              model_server: 'http://localhost:11434',
                              system_prompt: 'You are an expert document assembler. Combine and structure the reviewed content into a final, polished document.'
                            },
                            execution: {
                              max_phases: 10,
                              phase_timeout_minutes: 30,
                              retry_attempts: 3,
                              enable_streaming: true,
                              parallel_phases: false,
                              checkpoint_interval: 5,
                              error_handling: 'retry'
                            },
                            quality_control: {
                              quality_check_model: 'qwen3:30b-a3b-instruct-2507-q4_K_M',
                              factuality_threshold: 0.8,
                              coherence_threshold: 0.8,
                              completeness_threshold: 0.8,
                              auto_retry_on_low_quality: true,
                              max_refinement_iterations: 3,
                              enable_cross_validation: false
                            },
                            output: {
                              default_format: 'markdown',
                              include_metadata: true,
                              max_output_size_mb: 10,
                              output_validation: true,
                              output_deduplication: false,
                              include_phase_outputs: false,
                              compress_large_outputs: true,
                              enable_structured_output: true
                            },
                            caching: {
                              cache_templates: true,
                              cache_workflows: true,
                              cache_ttl_hours: 72,
                              cache_max_size_mb: 500,
                              enable_distributed_cache: false,
                              cache_eviction_policy: 'LRU',
                              cache_compression: true
                            },
                            advanced: {
                              log_level: 'INFO',
                              enable_metrics: true,
                              enable_tracing: false,
                              telemetry_endpoint: '',
                              performance_profiling: false,
                              debug_mode: false,
                              retry_backoff_strategy: 'exponential'
                            }
                          }
                        });
                        setEditTemplateOpen(true);
                      }}
                    >
                      Create Your First Template
                    </Button>
                  </Paper>
                ) : (
                  <Grid container spacing={2}>
                    {templates.map((template) => (
                      <Grid item xs={12} md={6} lg={4} key={template.id}>
                        <Card>
                          <CardContent>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 1 }}>
                              <Typography variant="h6" gutterBottom>
                                {template.name}
                              </Typography>
                              {!template.is_active && (
                                <Chip label="Inactive" size="small" color="default" />
                              )}
                            </Box>
                            <Chip 
                              label={getTemplateCategory(template)} 
                              size="small" 
                              color="primary" 
                              sx={{ mb: 1 }} 
                            />
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                              {template.description || 'No description provided'}
                            </Typography>
                            <Typography variant="caption" display="block">
                              Variables: {getTemplateVariables(template).length > 0 
                                ? getTemplateVariables(template).join(', ')
                                : 'None defined'}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              Updated: {new Date(template.updated_at).toLocaleDateString()}
                            </Typography>
                          </CardContent>
                          <CardActions>
                            <Button 
                              size="small" 
                              startIcon={<PlayIcon />}
                              onClick={() => {
                                setNewWorkflow({
                                  templateId: template.id,
                                  name: `${template.name} - ${new Date().toLocaleDateString()}`,
                                  description: `Generated from ${template.name}`,
                                  inputData: {}
                                });
                                setCreateWorkflowOpen(true);
                              }}
                            >
                              Use
                            </Button>
                            <Button 
                              size="small" 
                              startIcon={<EditIcon />}
                              onClick={() => handleEditTemplate(template)}
                            >
                              Edit
                            </Button>
                            <Button 
                              size="small" 
                              color="error"
                              startIcon={<DeleteIcon />}
                              onClick={() => handleDeleteClick('template', template.id, template.name)}
                            >
                              Delete
                            </Button>
                          </CardActions>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                )}
              </Box>

              {/* Workflows Tab */}
              <Box hidden={activeTab !== 1} sx={{ pt: 3 }}>
                <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="h5">Document Workflows</Typography>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button
                      variant="outlined"
                      startIcon={<RefreshIcon />}
                      onClick={() => setRefreshKey(prev => prev + 1)}
                      disabled={loading}
                    >
                      Refresh
                    </Button>
                    <Button
                      variant="contained"
                      startIcon={<AddIcon />}
                      onClick={() => {
                        if (templates.length === 0) {
                          setError('Please create a template first');
                          return;
                        }
                        setCreateWorkflowOpen(true);
                      }}
                      disabled={loading || templates.length === 0}
                    >
                      Create Workflow
                    </Button>
                  </Box>
                </Box>

                {workflows.length === 0 ? (
                  <Paper sx={{ p: 4, textAlign: 'center' }}>
                    <WorkflowIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                    <Typography variant="h6" gutterBottom>
                      No Workflows Yet
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                      Create a workflow from a template to start generating documents
                    </Typography>
                    {templates.length > 0 ? (
                      <Button
                        variant="contained"
                        startIcon={<AddIcon />}
                        onClick={() => setCreateWorkflowOpen(true)}
                      >
                        Create Your First Workflow
                      </Button>
                    ) : (
                      <Button
                        variant="outlined"
                        onClick={() => setActiveTab(0)}
                      >
                        Go to Templates
                      </Button>
                    )}
                  </Paper>
                ) : (
                  <Grid container spacing={2}>
                    {workflows.map((workflow) => {
                      const execution = executions.get(workflow.id);
                      const statusIcon = 
                        workflow.status === 'completed' || execution?.status === 'completed' ? <CheckCircleIcon /> :
                        workflow.status === 'failed' || execution?.status === 'failed' ? <ErrorIcon /> :
                        workflow.status === 'running' || execution?.status === 'running' ? <CircularProgress size={16} /> :
                        null;
                      
                      return (
                        <Grid item xs={12} key={workflow.id}>
                          <Card>
                            <CardContent>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 2 }}>
                                <Box>
                                  <Typography variant="h6">
                                    {workflow.name}
                                  </Typography>
                                  <Typography variant="body2" color="text.secondary">
                                    {workflow.description || 'No description'}
                                  </Typography>
                                </Box>
                                {(workflow.status || execution?.status) && (
                                  <Chip 
                                    label={execution?.status || workflow.status}
                                    icon={statusIcon}
                                    color={
                                      (execution?.status || workflow.status) === 'completed' ? 'success' : 
                                      (execution?.status || workflow.status) === 'failed' ? 'error' : 
                                      (execution?.status || workflow.status) === 'running' ? 'primary' :
                                      'default'
                                    }
                                    size="small"
                                  />
                                )}
                              </Box>

                              {execution && execution.progress > 0 && (
                                <Box sx={{ mb: 2 }}>
                                  <LinearProgress 
                                    variant="determinate" 
                                    value={execution.progress} 
                                    sx={{ mb: 1 }}
                                  />
                                  <Typography variant="caption" color="text.secondary">
                                    {execution.currentPhase || 'Processing...'} ({execution.progress}%)
                                  </Typography>
                                </Box>
                              )}

                              <Divider sx={{ my: 2 }} />

                              <Typography variant="caption" color="text.secondary">
                                Created: {new Date(workflow.created_at).toLocaleString()}
                                {workflow.completed_at && (
                                  <> | Completed: {new Date(workflow.completed_at).toLocaleString()}</>
                                )}
                              </Typography>
                            </CardContent>
                            <CardActions>
                              <Button 
                                size="small" 
                                variant="contained"
                                startIcon={execution?.status === 'running' ? <CircularProgress size={16} /> : <PlayIcon />}
                                onClick={() => runWorkflow(workflow.id)}
                                disabled={execution?.status === 'running'}
                              >
                                {execution?.status === 'running' ? 'Running...' : 'Run'}
                              </Button>
                              {workflow.status === 'pending' && !execution?.status && (
                                <Button 
                                  size="small" 
                                  startIcon={<EditIcon />}
                                  onClick={() => handleEditWorkflow(workflow)}
                                >
                                  Edit
                                </Button>
                              )}
                              {execution && (
                                <Button 
                                  size="small" 
                                  startIcon={<InfoIcon />}
                                  onClick={() => {
                                    setCurrentExecution(execution);
                                    setExecutionDialogOpen(true);
                                  }}
                                >
                                  View Details
                                </Button>
                              )}
                              <Button 
                                size="small" 
                                color="error"
                                startIcon={<DeleteIcon />}
                                onClick={() => handleDeleteClick('workflow', workflow.id, workflow.name)}
                                disabled={execution?.status === 'running'}
                              >
                                Delete
                              </Button>
                            </CardActions>
                          </Card>
                        </Grid>
                      );
                    })}
                  </Grid>
                )}
              </Box>

              {/* Create Workflow Dialog */}
              <Dialog 
                open={createWorkflowOpen} 
                onClose={() => setCreateWorkflowOpen(false)}
                maxWidth="md"
                fullWidth
              >
                <DialogTitle>Create New Workflow</DialogTitle>
                <DialogContent>
                  <FormControl fullWidth margin="normal">
                    <InputLabel>Select Template</InputLabel>
                    <Select
                      value={newWorkflow.templateId}
                      onChange={(e) => setNewWorkflow({ ...newWorkflow, templateId: e.target.value })}
                      label="Select Template"
                    >
                      {templates.filter(t => t.is_active).map((template) => (
                        <MenuItem key={template.id} value={template.id}>
                          {template.name}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  
                  <TextField
                    fullWidth
                    label="Workflow Name"
                    value={newWorkflow.name}
                    onChange={(e) => setNewWorkflow({ ...newWorkflow, name: e.target.value })}
                    margin="normal"
                    required
                  />
                  
                  <TextField
                    fullWidth
                    label="Description"
                    value={newWorkflow.description}
                    onChange={(e) => setNewWorkflow({ ...newWorkflow, description: e.target.value })}
                    margin="normal"
                    multiline
                    rows={3}
                  />
                  
                  {newWorkflow.templateId && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Input Variables
                      </Typography>
                      {(() => {
                        const template = templates.find(t => t.id === newWorkflow.templateId);
                        const variables = template ? getTemplateVariables(template) : [];
                        
                        if (variables.length === 0) {
                          return (
                            <Typography variant="body2" color="text.secondary">
                              No input variables required for this template
                            </Typography>
                          );
                        }
                        
                        return variables.map((variable) => (
                          <TextField
                            key={variable}
                            fullWidth
                            label={variable.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                            value={newWorkflow.inputData[variable] || ''}
                            onChange={(e) => setNewWorkflow({
                              ...newWorkflow,
                              inputData: {
                                ...newWorkflow.inputData,
                                [variable]: e.target.value
                              }
                            })}
                            margin="normal"
                            size="small"
                          />
                        ));
                      })()}
                    </Box>
                  )}
                </DialogContent>
                <DialogActions>
                  <Button onClick={() => {
                    setCreateWorkflowOpen(false);
                    setNewWorkflow({
                      templateId: '',
                      name: '',
                      description: '',
                      inputData: {}
                    });
                  }}>
                    Cancel
                  </Button>
                  <Button 
                    onClick={createWorkflow} 
                    variant="contained"
                    disabled={!newWorkflow.templateId || !newWorkflow.name || loading}
                  >
                    {loading ? <CircularProgress size={20} /> : 'Create & Run'}
                  </Button>
                </DialogActions>
              </Dialog>

              {/* Edit Workflow Dialog */}
              <Dialog 
                open={editWorkflowOpen} 
                onClose={() => {
                  setEditWorkflowOpen(false);
                  setSelectedWorkflowForEdit(null);
                  setNewWorkflow({
                    templateId: '',
                    name: '',
                    description: '',
                    inputData: {}
                  });
                }}
                maxWidth="md"
                fullWidth
              >
                <DialogTitle>Edit Workflow</DialogTitle>
                <DialogContent>
                  <TextField
                    fullWidth
                    label="Workflow Name"
                    value={newWorkflow.name}
                    onChange={(e) => setNewWorkflow({ ...newWorkflow, name: e.target.value })}
                    margin="normal"
                    required
                  />
                  
                  <TextField
                    fullWidth
                    label="Description"
                    value={newWorkflow.description}
                    onChange={(e) => setNewWorkflow({ ...newWorkflow, description: e.target.value })}
                    margin="normal"
                    multiline
                    rows={3}
                  />
                  
                  {selectedWorkflowForEdit && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Input Variables
                      </Typography>
                      {(() => {
                        const template = templates.find(t => t.id === selectedWorkflowForEdit.template_id);
                        const variables = template ? getTemplateVariables(template) : [];
                        
                        if (variables.length === 0) {
                          return (
                            <Typography variant="body2" color="text.secondary">
                              No input variables required for this template
                            </Typography>
                          );
                        }
                        
                        return variables.map((variable) => (
                          <TextField
                            key={variable}
                            fullWidth
                            label={variable.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                            value={newWorkflow.inputData[variable] || ''}
                            onChange={(e) => setNewWorkflow({
                              ...newWorkflow,
                              inputData: {
                                ...newWorkflow.inputData,
                                [variable]: e.target.value
                              }
                            })}
                            margin="normal"
                            size="small"
                          />
                        ));
                      })()}
                    </Box>
                  )}
                </DialogContent>
                <DialogActions>
                  <Button onClick={() => {
                    setEditWorkflowOpen(false);
                    setSelectedWorkflowForEdit(null);
                    setNewWorkflow({
                      templateId: '',
                      name: '',
                      description: '',
                      inputData: {}
                    });
                  }}>
                    Cancel
                  </Button>
                  <Button 
                    onClick={updateWorkflow} 
                    variant="contained"
                    disabled={!newWorkflow.name || loading}
                  >
                    {loading ? <CircularProgress size={20} /> : 'Update Workflow'}
                  </Button>
                </DialogActions>
              </Dialog>

              {/* Edit Template Dialog */}
              <Dialog 
                open={editTemplateOpen} 
                onClose={() => setEditTemplateOpen(false)}
                maxWidth="md"
                fullWidth
              >
                <DialogTitle>{isEditMode ? 'Edit Template' : 'Create New Template'}</DialogTitle>
                <DialogContent>
                  <TextField
                    fullWidth
                    label="Template Name"
                    value={newTemplate.name}
                    onChange={(e) => setNewTemplate({ ...newTemplate, name: e.target.value })}
                    margin="normal"
                    required
                  />
                  <TextField
                    fullWidth
                    label="Description"
                    value={newTemplate.description}
                    onChange={(e) => setNewTemplate({ ...newTemplate, description: e.target.value })}
                    margin="normal"
                    multiline
                    rows={2}
                  />
                  <FormControl fullWidth margin="normal">
                    <InputLabel>Template Type</InputLabel>
                    <Select 
                      label="Template Type"
                      value={newTemplate.template_type}
                      onChange={(e) => setNewTemplate({ ...newTemplate, template_type: e.target.value })}
                    >
                      <MenuItem value="document_generation">Document Generation</MenuItem>
                      <MenuItem value="document">Document</MenuItem>
                      <MenuItem value="report">Report</MenuItem>
                      <MenuItem value="analysis">Analysis</MenuItem>
                      <MenuItem value="summary">Summary</MenuItem>
                      <MenuItem value="technical">Technical</MenuItem>
                      <MenuItem value="business">Business</MenuItem>
                    </Select>
                  </FormControl>
                  
                  <TextField
                    fullWidth
                    label="Variables (comma-separated)"
                    value={newTemplate.template_config.variables?.map(v => 
                      typeof v === 'string' ? v : (v.name || v)
                    ).join(', ') || ''}
                    onChange={(e) => {
                      const newVariableNames = e.target.value
                        .split(',')
                        .map(v => v.trim())
                        .filter(v => v.length > 0);
                      
                      // Preserve object structure if variables were originally objects
                      const originalVariables = newTemplate.template_config.variables || [];
                      const hasObjectVariables = originalVariables.some(v => typeof v === 'object');
                      
                      const variables = hasObjectVariables 
                        ? newVariableNames.map(name => {
                            // Try to find existing object for this name to preserve properties
                            const existingVar = originalVariables.find(v => 
                              typeof v === 'object' && v.name === name
                            );
                            return existingVar || { name, type: 'string', description: '' };
                          })
                        : newVariableNames;
                      
                      setNewTemplate({
                        ...newTemplate,
                        template_config: {
                          ...newTemplate.template_config,
                          variables
                        }
                      });
                    }}
                    margin="normal"
                    helperText="e.g., project_name, version, author"
                  />
                  
                  <FormControl fullWidth margin="normal">
                    <InputLabel>Output Format</InputLabel>
                    <Select 
                      label="Output Format"
                      value={newTemplate.template_config.output_format || 'markdown'}
                      onChange={(e) => setNewTemplate({
                        ...newTemplate,
                        template_config: {
                          ...newTemplate.template_config,
                          output_format: e.target.value
                        }
                      })}
                    >
                      <MenuItem value="markdown">Markdown</MenuItem>
                      <MenuItem value="html">HTML</MenuItem>
                      <MenuItem value="json">JSON</MenuItem>
                      <MenuItem value="plain">Plain Text</MenuItem>
                    </Select>
                  </FormControl>
                  
                  <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
                    Template Phases
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 2 }}>
                    Define the phases for document generation. Each phase represents a step in the process.
                  </Typography>
                  
                  <Box sx={{ mb: 2 }}>
                    <Button
                      variant="outlined"
                      startIcon={<AddIcon />}
                      onClick={() => {
                        const newPhase = {
                          name: `Phase ${newTemplate.template_config.phases.length + 1}`,
                          type: 'generator',
                          prompt: '',
                          model: 'default'
                        };
                        setNewTemplate({
                          ...newTemplate,
                          template_config: {
                            ...newTemplate.template_config,
                            phases: [...newTemplate.template_config.phases, newPhase]
                          }
                        });
                      }}
                      fullWidth
                    >
                      Add Phase
                    </Button>
                  </Box>
                  
                  {newTemplate.template_config.phases.map((phase, index) => (
                    <Paper key={index} sx={{ p: 2, mb: 2 }}>
                      <Grid container spacing={2}>
                        <Grid item xs={12} sm={6}>
                          <TextField
                            fullWidth
                            label="Phase Name"
                            value={phase.name}
                            onChange={(e) => {
                              const updatedPhases = [...newTemplate.template_config.phases];
                              updatedPhases[index] = { ...phase, name: e.target.value };
                              setNewTemplate({
                                ...newTemplate,
                                template_config: {
                                  ...newTemplate.template_config,
                                  phases: updatedPhases
                                }
                              });
                            }}
                            size="small"
                          />
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <FormControl fullWidth size="small">
                            <InputLabel>Phase Type</InputLabel>
                            <Select
                              label="Phase Type"
                              value={phase.type}
                              onChange={(e) => {
                                const updatedPhases = [...newTemplate.template_config.phases];
                                updatedPhases[index] = { ...phase, type: e.target.value };
                                setNewTemplate({
                                  ...newTemplate,
                                  template_config: {
                                    ...newTemplate.template_config,
                                    phases: updatedPhases
                                  }
                                });
                              }}
                            >
                              <MenuItem value="analyzer">Analyzer</MenuItem>
                              <MenuItem value="generator">Generator</MenuItem>
                              <MenuItem value="reviewer">Reviewer</MenuItem>
                              <MenuItem value="assembler">Assembler</MenuItem>
                            </Select>
                          </FormControl>
                        </Grid>
                        <Grid item xs={12}>
                          <TextField
                            fullWidth
                            label="Prompt Template"
                            value={phase.prompt || ''}
                            onChange={(e) => {
                              const updatedPhases = [...newTemplate.template_config.phases];
                              updatedPhases[index] = { ...phase, prompt: e.target.value };
                              setNewTemplate({
                                ...newTemplate,
                                template_config: {
                                  ...newTemplate.template_config,
                                  phases: updatedPhases
                                }
                              });
                            }}
                            multiline
                            rows={3}
                            size="small"
                            placeholder="Enter the prompt for this phase. Use {{variable_name}} for variables."
                          />
                        </Grid>
                        <Grid item xs={12}>
                          <Button
                            size="small"
                            color="error"
                            onClick={() => {
                              const updatedPhases = newTemplate.template_config.phases.filter((_, i) => i !== index);
                              setNewTemplate({
                                ...newTemplate,
                                template_config: {
                                  ...newTemplate.template_config,
                                  phases: updatedPhases
                                }
                              });
                            }}
                          >
                            Remove Phase
                          </Button>
                        </Grid>
                      </Grid>
                    </Paper>
                  ))}
                </DialogContent>
                <DialogActions>
                  <Button onClick={resetTemplateForm}>
                    Cancel
                  </Button>
                  <Button 
                    variant="contained"
                    onClick={isEditMode ? updateTemplate : createTemplate}
                    disabled={!newTemplate.name || newTemplate.template_config.phases.length === 0 || loading}
                  >
                    {loading ? <CircularProgress size={20} /> : (isEditMode ? 'Update Template' : 'Create Template')}
                  </Button>
                </DialogActions>
              </Dialog>
              
              {/* Delete Confirmation Dialog */}
              <Dialog
                open={deleteConfirmOpen}
                onClose={() => setDeleteConfirmOpen(false)}
              >
                <DialogTitle>
                  Delete {itemToDelete?.type === 'template' ? 'Template' : 'Workflow'}
                </DialogTitle>
                <DialogContent>
                  <Typography>
                    Are you sure you want to delete "{itemToDelete?.name}"?
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                    This action cannot be undone.
                  </Typography>
                </DialogContent>
                <DialogActions>
                  <Button onClick={() => {
                    setDeleteConfirmOpen(false);
                    setItemToDelete(null);
                  }}>
                    Cancel
                  </Button>
                  <Button 
                    onClick={handleDeleteConfirm} 
                    color="error" 
                    variant="contained"
                    disabled={loading}
                  >
                    {loading ? <CircularProgress size={20} /> : 'Delete'}
                  </Button>
                </DialogActions>
              </Dialog>
              
              {/* Execution Details Dialog */}
              <Dialog
                open={executionDialogOpen}
                onClose={() => setExecutionDialogOpen(false)}
                maxWidth="md"
                fullWidth
              >
                <DialogTitle>
                  Workflow Execution Details
                  {currentExecution && (
                    <Chip
                      label={currentExecution.status}
                      color={
                        currentExecution.status === 'completed' ? 'success' :
                        currentExecution.status === 'failed' ? 'error' :
                        currentExecution.status === 'running' ? 'primary' :
                        'default'
                      }
                      size="small"
                      sx={{ ml: 2 }}
                    />
                  )}
                </DialogTitle>
                <DialogContent>
                  {currentExecution && (
                    <Box>
                      {currentExecution.status === 'running' && (
                        <Box sx={{ mb: 3 }}>
                          <LinearProgress 
                            variant="determinate" 
                            value={currentExecution.progress} 
                            sx={{ mb: 1 }}
                          />
                          <Typography variant="body2">
                            {currentExecution.currentPhase || 'Processing...'} ({currentExecution.progress}%)
                          </Typography>
                        </Box>
                      )}
                      
                      <Typography variant="subtitle2" gutterBottom>
                        Execution Events
                      </Typography>
                      <Paper variant="outlined" sx={{ p: 2, maxHeight: 400, overflow: 'auto' }}>
                        {currentExecution.events.length === 0 ? (
                          <Typography variant="body2" color="text.secondary">
                            No events yet...
                          </Typography>
                        ) : (
                          <List dense>
                            {currentExecution.events.map((event, index) => (
                              <ListItem key={index}>
                                <ListItemText
                                  primary={
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                      {event.type === 'phase_complete' && <CheckCircleIcon color="success" fontSize="small" />}
                                      {event.type === 'phase_error' && <ErrorIcon color="error" fontSize="small" />}
                                      {event.type === 'phase_start' && <InfoIcon color="info" fontSize="small" />}
                                      <Typography variant="body2">
                                        {event.phase || event.type.replace(/_/g, ' ').toUpperCase()}
                                      </Typography>
                                    </Box>
                                  }
                                  secondary={
                                    <>
                                      {event.message && <Typography variant="caption">{event.message}</Typography>}
                                      {event.error && <Typography variant="caption" color="error">{event.error}</Typography>}
                                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                                        {new Date(event.timestamp).toLocaleTimeString()}
                                      </Typography>
                                    </>
                                  }
                                />
                              </ListItem>
                            ))}
                          </List>
                        )}
                      </Paper>
                    </Box>
                  )}
                </DialogContent>
                <DialogActions>
                  <Button onClick={() => setExecutionDialogOpen(false)}>
                    Close
                  </Button>
                  {currentExecution?.status === 'running' && (
                    <Button 
                      color="error" 
                      onClick={() => {
                        // Stop the execution
                        if (currentExecution.eventSource) {
                          currentExecution.eventSource.close();
                        }
                        setExecutions(prev => {
                          const updated = new Map(prev);
                          const exec = updated.get(currentExecution.workflowId);
                          if (exec) {
                            exec.status = 'failed';
                            exec.events.push({
                              type: 'workflow_error',
                              error: 'Execution cancelled by user',
                              timestamp: new Date().toISOString()
                            });
                          }
                          return updated;
                        });
                        setExecutionDialogOpen(false);
                      }}
                    >
                      Stop Execution
                    </Button>
                  )}
                </DialogActions>
              </Dialog>
              
              {/* Loading Backdrop */}
              <Backdrop
                sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
                open={loading && (templates.length > 0 || workflows.length > 0)}
              >
                <CircularProgress color="inherit" />
              </Backdrop>

            </Box>
          </Paper>
        </Container>

      </Box>
    </ThemeProvider>
  );
}

export default MetaTaskApp;