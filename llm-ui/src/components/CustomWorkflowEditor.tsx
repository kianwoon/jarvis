import React, { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  IconButton,
  Chip,
  Toolbar,
  Paper,
  Tooltip,
  Divider,
  Alert,
  TextField,
  Snackbar,
  CircularProgress,
  Menu,
  MenuItem,
  ButtonGroup
} from '@mui/material';
import {
  Close as CloseIcon,
  Save as SaveIcon,
  PlayArrow as PlayIcon,
  Add as AddIcon,
  Build as BuildIcon,
  Psychology as PsychologyIcon,
  Storage as StorageIcon,
  Settings as SettingsIcon,
  Fullscreen as FullscreenIcon,
  FullscreenExit as FullscreenExitIcon,
  ContentCopy,
  Http as HttpIcon,
  Loop as LoopIcon,
  Transform as TransformIcon,
  Error as ErrorIcon,
  Schedule as ScheduleIcon,
  Folder as FileIcon,
  Email as EmailIcon,
  CallSplit as SwitchIcon,
  Code as JsonIcon,
  TextFields as TextIcon,
  Functions as MathIcon,
  AccessTime as DelayIcon,
  Merge as MergeIcon,
  Notifications as NotificationIcon,
  DataObject as DatabaseIcon,
  Link as WebhookIcon,
  Memory as VariableIcon,
  Psychology as MemoryIcon,
  ExpandMore as ExpandAllIcon,
  ExpandLess as CollapseAllIcon,
  Visibility as ShowIOIcon,
  ArrowDropDown as DropDownIcon,
  AccountTree as HierarchicalIcon,
  ScatterPlot as ForceDirectedIcon,
  GridOn as GridIcon,
  AutoFixHigh as SmartIcon,
  GroupWork as GroupIcon,
  Timeline as TimelineIcon,
  Layers as LayersIcon,
  Category as CategoryIcon
} from '@mui/icons-material';

import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  BackgroundVariant,
  Panel,
  NodeTypes,
  MiniMap,
  MarkerType,
  EdgeLabelRenderer,
  getBezierPath,
  useReactFlow,
  ConnectionLineType
} from 'reactflow';
import 'reactflow/dist/style.css';
import '../styles/workflow-handles.css';
import '../styles/workflow-select-fix.css';

import dagre from 'dagre';
import { 
  applyAdvancedLayout, 
  defaultLayoutOptions, 
  LayoutOptions,
  animateLayoutTransition,
  executionSequenceLayout,
  executionForceDirectedLayout,
  executionLayeredLayout,
  executionSemanticLayout
} from '../utils/advanced-layout';

// Custom Node Components
import MCPToolNode from './workflow-nodes/MCPToolNode';
import AgentNode from './workflow-nodes/AgentNode';
import RedisNode from './workflow-nodes/RedisNode';
import ConditionNode from './workflow-nodes/ConditionNode';
import StartNode from './workflow-nodes/StartNode';
import EndNode from './workflow-nodes/EndNode';
import InputNode from './workflow-nodes/InputNode';
import OutputNode from './workflow-nodes/OutputNode';
import LLMNode from './workflow-nodes/LLMNode';
import HttpNode from './workflow-nodes/HttpNode';
import LoopNode from './workflow-nodes/LoopNode';
import VariableNode from './workflow-nodes/VariableNode';
import DataMapperNode from './workflow-nodes/DataMapperNode';
import SwitchNode from './workflow-nodes/SwitchNode';
import DelayNode from './workflow-nodes/DelayNode';
import RouterNode from './workflow-nodes/RouterNode';
import ParallelNode from './workflow-nodes/ParallelNode';
import TransformNode from './workflow-nodes/TransformNode';
import CacheNode from './workflow-nodes/CacheNode';
import TriggerNode from './workflow-nodes/TriggerNode';
import AggregatorNode from './workflow-nodes/AggregatorNode';
import APINode from './workflow-nodes/APINode';

// Custom Edge Components
import DirectionalEdge from './workflow-edges/DirectionalEdge';
import ErrorHandlerNode from './workflow-nodes/ErrorHandlerNode';
import StateNode from './workflow-nodes/StateNode';
import GroupBoxNode from './workflow-nodes/GroupBoxNode';
import NoteViewer from './NoteViewer';
import EnhancedWorkflowCompletionDialog from './EnhancedWorkflowCompletionDialog';
import { downloadWorkflowOutput } from '../utils/fileDownload';

interface CustomWorkflowEditorProps {
  open?: boolean;
  onClose?: () => void;
  workflowId?: number | string;
  workflowName?: string;
  onSave?: (workflowData: any) => void;
  initialWorkflow?: any;
  onBack?: () => void;
  isNewWorkflow?: boolean;
}

// Node types will be created dynamically with updateNodeData function

// Legacy AI-focused node templates (fallback when agent-based schemas fail to load)
const legacyNodeCategories = {
  'Workflow Control': [
    {
      type: 'start',
      label: 'Start',
      icon: <PlayIcon />,
      color: '#4caf50',
      description: 'Workflow starting point',
      inputs: [],
      outputs: [
        { id: 'output', label: 'Output', type: 'any', description: 'Initial workflow data' }
      ]
    },
    {
      type: 'end',
      label: 'End',
      icon: <CloseIcon />,
      color: '#f44336',
      description: 'Workflow endpoint',
      inputs: [
        { id: 'input', label: 'Input', type: 'any', description: 'Final workflow result' }
      ],
      outputs: []
    },
    {
      type: 'condition',
      label: 'Condition',
      icon: <SettingsIcon />,
      color: '#795548',
      description: 'Basic conditional logic branching',
      inputs: [
        { id: 'input', label: 'Input', type: 'any', description: 'Data to evaluate' },
        { id: 'condition', label: 'Condition', type: 'any', description: 'Condition value' }
      ],
      outputs: [
        { id: 'true', label: 'True', type: 'any', description: 'When condition is true' },
        { id: 'false', label: 'False', type: 'any', description: 'When condition is false' }
      ]
    },
    {
      type: 'loop',
      label: 'Loop',
      icon: <LoopIcon />,
      color: '#673ab7',
      description: 'Iterate over document batches or data collections',
      inputs: [
        { id: 'array', label: 'Array', type: 'array', description: 'Array to iterate over' },
        { id: 'input', label: 'Input', type: 'any', description: 'Data for each iteration' }
      ],
      outputs: [
        { id: 'item', label: 'Current Item', type: 'any', description: 'Current array item' },
        { id: 'index', label: 'Index', type: 'number', description: 'Current index' },
        { id: 'output', label: 'Final Result', type: 'array', description: 'Loop results' }
      ]
    },
    {
      type: 'parallelnode',
      label: 'Parallel Execution',
      icon: <HierarchicalIcon />,
      color: '#00bcd4',
      description: 'Execute multiple branches in parallel for concurrent processing',
      inputs: [
        { id: 'input', label: 'Input', type: 'any', description: 'Data to send to all branches' }
      ],
      outputs: [
        { id: 'parallel-1', label: 'Branch 1', type: 'any', description: 'First parallel branch' },
        { id: 'parallel-2', label: 'Branch 2', type: 'any', description: 'Second parallel branch' },
        { id: 'parallel-3', label: 'Branch 3', type: 'any', description: 'Third parallel branch' },
        { id: 'summary', label: 'Summary', type: 'any', description: 'Aggregated results from all branches' }
      ]
    },
    {
      type: 'triggernode',
      label: 'External Trigger',
      icon: <WebhookIcon />,
      color: '#10b981',
      description: 'Allow external systems to trigger workflow execution via HTTP endpoints',
      inputs: [],
      outputs: [
        { id: 'trigger_data', label: 'Trigger Data', type: 'any', description: 'Data received from external trigger' },
        { id: 'query_params', label: 'Query Parameters', type: 'object', description: 'HTTP query parameters' },
        { id: 'headers', label: 'Headers', type: 'object', description: 'HTTP headers' }
      ]
    }
  ],
  'AI Processing': [
    {
      type: 'llm',
      label: 'LLM',
      icon: <PsychologyIcon />,
      color: '#2196f3',
      description: 'Direct LLM access for custom prompts and experimentation',
      inputs: [
        { id: 'text', label: 'Prompt', type: 'string', description: 'Text prompt for LLM' },
        { id: 'context', label: 'Context', type: 'any', description: 'Additional context data' }
      ],
      outputs: [
        { id: 'output', label: 'Response', type: 'string', description: 'LLM generated response' },
        { id: 'metadata', label: 'Metadata', type: 'object', description: 'Response metadata' }
      ]
    },
    {
      type: 'agent',
      label: 'AI Agent',
      icon: <PsychologyIcon />,
      color: '#9c27b0',
      description: 'Structured AI workflows with predefined agents and MCP tools',
      inputs: [
        { id: 'query', label: 'Query', type: 'string', description: 'Query for the agent' },
        { id: 'context', label: 'Context', type: 'any', description: 'Context data' },
        { id: 'documents', label: 'Documents', type: 'array', description: 'Documents to process' }
      ],
      outputs: [
        { id: 'output', label: 'Result', type: 'any', description: 'Agent execution result' },
        { id: 'reasoning', label: 'Reasoning', type: 'string', description: 'Agent reasoning steps' },
        { id: 'analysis', label: 'Analysis', type: 'object', description: 'Detailed analysis results' }
      ]
    },
    {
      type: 'aiDecision',
      label: 'AI Decision',
      icon: <SettingsIcon />,
      color: '#ff9800',
      description: 'Content-aware intelligent branching based on AI analysis',
      inputs: [
        { id: 'input', label: 'Input', type: 'any', description: 'Data to analyze for decision' },
        { id: 'criteria', label: 'Criteria', type: 'string', description: 'Decision criteria prompt' }
      ],
      outputs: [
        { id: 'decision', label: 'Decision', type: 'string', description: 'AI decision result' },
        { id: 'confidence', label: 'Confidence', type: 'number', description: 'Decision confidence score' },
        { id: 'reasoning', label: 'Reasoning', type: 'string', description: 'Decision reasoning' }
      ]
    },
    {
      type: 'aggregatornode',
      label: 'Aggregator',
      icon: <MergeIcon />,
      color: '#ff9800',
      description: 'Merge and aggregate multiple inputs using AI-powered strategies',
      inputs: [
        { id: 'input1', label: 'Input 1', type: 'any', description: 'First input to aggregate' },
        { id: 'input2', label: 'Input 2', type: 'any', description: 'Second input to aggregate' },
        { id: 'input3', label: 'Input 3', type: 'any', description: 'Third input to aggregate' }
      ],
      outputs: [
        { id: 'output', label: 'Aggregated Result', type: 'any', description: 'Merged and processed output' },
        { id: 'metadata', label: 'Metadata', type: 'object', description: 'Aggregation metadata' },
        { id: 'confidence', label: 'Confidence', type: 'number', description: 'Aggregation confidence score' }
      ]
    }
  ],
  'Document Intelligence': [
    {
      type: 'documentProcessor',
      label: 'Document Processor',
      icon: <FileIcon />,
      color: '#1976d2',
      description: 'Process and analyze PDF, Word, and text documents',
      inputs: [
        { id: 'document', label: 'Document', type: 'any', description: 'Document file or content' },
        { id: 'task', label: 'Analysis Task', type: 'string', description: 'What to extract or analyze' }
      ],
      outputs: [
        { id: 'content', label: 'Extracted Content', type: 'string', description: 'Extracted text content' },
        { id: 'analysis', label: 'Analysis', type: 'object', description: 'Document analysis results' },
        { id: 'metadata', label: 'Metadata', type: 'object', description: 'Document metadata' }
      ]
    },
    {
      type: 'imageProcessor',
      label: 'Image Processor',
      icon: <TransformIcon />,
      color: '#e91e63',
      description: 'OCR, image analysis, and visual understanding',
      inputs: [
        { id: 'image', label: 'Image', type: 'any', description: 'Image file or data' },
        { id: 'task', label: 'Analysis Task', type: 'string', description: 'OCR, description, or specific analysis' }
      ],
      outputs: [
        { id: 'text', label: 'Extracted Text', type: 'string', description: 'OCR text extraction' },
        { id: 'description', label: 'Description', type: 'string', description: 'Image description' },
        { id: 'analysis', label: 'Analysis', type: 'object', description: 'Detailed image analysis' }
      ]
    },
    {
      type: 'audioProcessor',
      label: 'Audio Processor',
      icon: <NotificationIcon />,
      color: '#ff5722',
      description: 'Audio transcription and analysis',
      inputs: [
        { id: 'audio', label: 'Audio', type: 'any', description: 'Audio file or data' },
        { id: 'task', label: 'Analysis Task', type: 'string', description: 'Transcription or audio analysis' }
      ],
      outputs: [
        { id: 'transcript', label: 'Transcript', type: 'string', description: 'Audio transcription' },
        { id: 'analysis', label: 'Analysis', type: 'object', description: 'Audio content analysis' },
        { id: 'metadata', label: 'Metadata', type: 'object', description: 'Audio metadata' }
      ]
    },
    {
      type: 'multiModalFusion',
      label: 'Multi-modal Fusion',
      icon: <MergeIcon />,
      color: '#9c27b0',
      description: 'Combine and analyze text, images, and audio together',
      inputs: [
        { id: 'text', label: 'Text Data', type: 'string', description: 'Text content' },
        { id: 'images', label: 'Images', type: 'array', description: 'Image files' },
        { id: 'audio', label: 'Audio', type: 'any', description: 'Audio files' },
        { id: 'task', label: 'Fusion Task', type: 'string', description: 'How to combine and analyze' }
      ],
      outputs: [
        { id: 'analysis', label: 'Unified Analysis', type: 'object', description: 'Combined multi-modal analysis' },
        { id: 'summary', label: 'Summary', type: 'string', description: 'Unified content summary' },
        { id: 'insights', label: 'Insights', type: 'array', description: 'Cross-modal insights' }
      ]
    }
  ],
  'AI Memory & Storage': [
    {
      type: 'contextMemory',
      label: 'Context Memory',
      icon: <MemoryIcon />,
      color: '#607d8b',
      description: 'Maintain workflow context for multi-turn AI processes',
      inputs: [
        { id: 'data', label: 'Data', type: 'any', description: 'Data to store in context' },
        { id: 'key', label: 'Context Key', type: 'string', description: 'Memory key identifier' }
      ],
      outputs: [
        { id: 'context', label: 'Context', type: 'object', description: 'Current context state' },
        { id: 'history', label: 'History', type: 'array', description: 'Context history' }
      ]
    },
    {
      type: 'variable',
      label: 'Variable Store',
      icon: <VariableIcon />,
      color: '#8bc34a',
      description: 'Store workflow variables and intermediate results',
      inputs: [
        { id: 'value', label: 'Value', type: 'any', description: 'Value to store' }
      ],
      outputs: [
        { id: 'value', label: 'Value', type: 'any', description: 'Variable value' },
        { id: 'name', label: 'Name', type: 'string', description: 'Variable name' }
      ]
    },
    {
      type: 'dataMapper',
      label: 'Data Mapper',
      icon: <TransformIcon />,
      color: '#e91e63',
      description: 'Transform data structures for AI workflow compatibility',
      inputs: [
        { id: 'data', label: 'Input Data', type: 'any', description: 'Data to transform' },
        { id: 'mapping', label: 'Mapping', type: 'object', description: 'Transformation mapping' }
      ],
      outputs: [
        { id: 'data', label: 'Output Data', type: 'any', description: 'Transformed data' },
        { id: 'metadata', label: 'Metadata', type: 'object', description: 'Transformation metadata' }
      ]
    }
  ],
  'Organization': [
    {
      type: 'groupBox',
      label: 'Group Box',
      icon: <FileIcon />,
      color: '#607d8b',
      description: 'Visual grouping container for organizing and documenting workflow sections',
      inputs: [],
      outputs: []
    }
  ]
};

// Flatten all legacy node templates for easy access (fallback)
const legacyNodeTemplates = Object.values(legacyNodeCategories).flat();

// Helper function to get node template with handle metadata
const getNodeTemplate = (nodeType: string) => {
  return legacyNodeTemplates.find(template => template.type === nodeType);
};

// Helper function to get input/output handles for a node type
const getNodeHandles = (nodeType: string) => {
  const template = getNodeTemplate(nodeType);
  if (!template) return { inputs: [], outputs: [] };
  
  return {
    inputs: template.inputs || [],
    outputs: template.outputs || []
  };
};

// Create a factory function for nodeTypes to avoid recreating them
const createNodeTypes = (updateNodeData: any, showIOGetter: () => boolean, handleOutputNodeViewReport: any, setAccordionOpen: any) => ({
  llm: (props: any) => <LLMNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  mcpTool: (props: any) => <MCPToolNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  agent: (props: any) => <AgentNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  redis: (props: any) => <RedisNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  condition: (props: any) => <ConditionNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  start: (props: any) => <StartNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  end: (props: any) => <EndNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  http: (props: any) => <HttpNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  loop: (props: any) => <LoopNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  variable: (props: any) => <VariableNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  dataMapper: (props: any) => <DataMapperNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  switch: (props: any) => <SwitchNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  delay: (props: any) => <DelayNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  errorHandler: (props: any) => <ErrorHandlerNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  statenode: (props: any) => <StateNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  groupBox: (props: any) => <GroupBoxNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  
  // Agent-based workflow node types
  agentnode: (props: any) => <AgentNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} onAccordionStateChange={setAccordionOpen} />,
  inputnode: (props: any) => <InputNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  outputnode: (props: any) => <OutputNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} onViewReport={handleOutputNodeViewReport} />,
  conditionnode: (props: any) => <ConditionNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  parallelnode: (props: any) => <ParallelNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  routernode: (props: any) => <RouterNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  transformnode: (props: any) => <TransformNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  cachenode: (props: any) => <CacheNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  triggernode: (props: any) => <TriggerNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  aggregatornode: (props: any) => <AggregatorNode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  apinode: (props: any) => <APINode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
  api: (props: any) => <APINode {...props} updateNodeData={updateNodeData} showIO={props.data.showIO ?? showIOGetter()} />,
});

// Function to migrate legacy handle IDs to the correct directional handle IDs
const migrateEdgeHandleIds = (edges: Edge[], nodes: Node[]): Edge[] => {
  
  const migratedEdges = edges.map(edge => {
    const sourceNode = nodes.find(n => n.id === edge.source);
    const targetNode = nodes.find(n => n.id === edge.target);
    
    
    // Skip migration if edge doesn't have valid source/target handles
    if (!edge.sourceHandle || !edge.targetHandle) {
      return edge;
    }
    
    // Skip migration if nodes not found
    if (!sourceNode || !targetNode) {
      return edge;
    }
    
    let newSourceHandle = edge.sourceHandle;
    let newTargetHandle = edge.targetHandle;
    
    // Migrate source handle IDs
    if (sourceNode && edge.sourceHandle) {
      if (sourceNode.type === 'agent' || sourceNode.type === 'agentnode') {
        // AgentNode: 'output' -> 'output-right' (default to right)
        if (edge.sourceHandle === 'output') {
          newSourceHandle = 'output-right';
        }
      } else if (sourceNode.type === 'start' || sourceNode.type === 'startnode') {
        // StartNode: 'start' -> 'start-right' (default to right)
        if (edge.sourceHandle === 'start') {
          newSourceHandle = 'start-right';
        }
      } else if (sourceNode.type === 'input' || sourceNode.type === 'inputnode') {
        // InputNode: 'data' -> 'data-right' (default to right)
        if (edge.sourceHandle === 'data') {
          newSourceHandle = 'data-right';
        }
      } else if (sourceNode.type === 'condition' || sourceNode.type === 'conditionnode') {
        // ConditionNode: 'true'/'false' -> 'true-right'/'false-left' (preserve logic flow)
        if (edge.sourceHandle === 'true') {
          newSourceHandle = 'true-right';
        } else if (edge.sourceHandle === 'false') {
          newSourceHandle = 'false-left';
        }
      } else if (sourceNode.type === 'aggregatornode') {
        // AggregatorNode: 'output' -> 'output-right' (default to right)
        if (edge.sourceHandle === 'output') {
          newSourceHandle = 'output-right';
        } else if (edge.sourceHandle === 'metadata') {
          newSourceHandle = 'metadata-right';
        } else if (edge.sourceHandle === 'confidence') {
          newSourceHandle = 'confidence-right';
        }
      }
    }
    
    // Migrate target handle IDs
    if (targetNode && edge.targetHandle) {
      if (targetNode.type === 'agent' || targetNode.type === 'agentnode') {
        // AgentNode: 'input' -> 'input-left' (default to left)
        if (edge.targetHandle === 'input') {
          newTargetHandle = 'input-left';
        }
      } else if (targetNode.type === 'end' || targetNode.type === 'endnode') {
        // EndNode: 'end' -> 'end-left' (default to left)
        if (edge.targetHandle === 'end') {
          newTargetHandle = 'end-left';
        }
      } else if (targetNode.type === 'output' || targetNode.type === 'outputnode') {
        // OutputNode: 'result' -> 'result-left', 'end-left' -> 'result-left' (legacy patterns)
        if (edge.targetHandle === 'result') {
          newTargetHandle = 'result-left';
        } else if (edge.targetHandle === 'end-left') {
          newTargetHandle = 'result-left';
        } else if (edge.targetHandle === 'end') {
          newTargetHandle = 'result-left';
        }
      } else if (targetNode.type === 'condition' || targetNode.type === 'conditionnode') {
        // ConditionNode: 'input' -> 'input-left' (default to left)
        if (edge.targetHandle === 'input') {
          newTargetHandle = 'input-left';
        }
      } else if (targetNode.type === 'aggregatornode') {
        // AggregatorNode: 'input1', 'input2', 'input3' -> 'input1-left', 'input2-left', 'input3-left'
        if (edge.targetHandle === 'input1') {
          newTargetHandle = 'input1-left';
        } else if (edge.targetHandle === 'input2') {
          newTargetHandle = 'input2-left';
        } else if (edge.targetHandle === 'input3') {
          newTargetHandle = 'input3-left';
        }
      } else if (targetNode.type === 'parallelnode') {
        // ParallelNode: 'input' is the current correct handle ID - no migration needed
        // ParallelNode uses 'input' as its legitimate target handle
      }
    }
    
    // Log migration results
    const wasChanged = newSourceHandle !== edge.sourceHandle || newTargetHandle !== edge.targetHandle;
    
    if (wasChanged) {
    }
    
    return {
      ...edge,
      sourceHandle: newSourceHandle,
      targetHandle: newTargetHandle
    };
  });
  
  return migratedEdges;
};

// Helper function to validate connection compatibility
const isValidConnection = (sourceType: string, sourceHandle: string, targetType: string, targetHandle: string, workflowType: string) => {
  // For agent-based nodes, allow most connections since they're flexible
  if (workflowType === 'agent_based') {
    // Allow connections between agent nodes
    if (sourceType === 'agentnode' && targetType === 'agentnode') return true;
    if (sourceType === 'inputnode' && targetType === 'agentnode') return true;
    if (sourceType === 'agentnode' && targetType === 'outputnode') return true;
    if (sourceType === 'agentnode' && targetType === 'conditionnode') return true;
    if (sourceType === 'conditionnode' && targetType === 'agentnode') return true;
    if (sourceType === 'parallelnode' && targetType === 'agentnode') return true;
    if (sourceType === 'agentnode' && targetType === 'parallelnode') return true;
    if (sourceType === 'statenode' && targetType === 'agentnode') return true;
    if (sourceType === 'agentnode' && targetType === 'statenode') return true;
    if (sourceType === 'agentnode' && targetType === 'transformnode') return true;
    if (sourceType === 'transformnode' && targetType === 'agentnode') return true;
    if (sourceType === 'inputnode' && targetType === 'transformnode') return true;
    if (sourceType === 'transformnode' && targetType === 'outputnode') return true;
    
    // CacheNode connections
    if (sourceType === 'agentnode' && targetType === 'cachenode') return true;
    if (sourceType === 'cachenode' && targetType === 'agentnode') return false; // One-way only
    if (sourceType === 'inputnode' && targetType === 'cachenode') return true;
    if (sourceType === 'cachenode' && targetType === 'outputnode') return true;
    
    // AggregatorNode connections
    if (sourceType === 'agentnode' && targetType === 'aggregatornode') return true;
    if (sourceType === 'parallelnode' && targetType === 'aggregatornode') return true;
    if (sourceType === 'transformnode' && targetType === 'aggregatornode') return true;
    if (sourceType === 'aggregatornode' && targetType === 'agentnode') return true;
    if (sourceType === 'aggregatornode' && targetType === 'outputnode') return true;
    if (sourceType === 'inputnode' && targetType === 'aggregatornode') return true;
    
    // Generally allow connections for agent-based workflows
    return true;
  }
  
  // Fallback to legacy validation if templates not found
  if (!sourceTemplate || !targetTemplate) {
    return true;
  }
  
  const sourceOutput = sourceTemplate.outputs?.find((output: any) => output.id === sourceHandle);
  const targetInput = targetTemplate.inputs?.find((input: any) => input.id === targetHandle);
  
  if (!sourceOutput || !targetInput) {
    return true;
  }
  
  // Type compatibility check
  if (sourceOutput.type === 'any' || targetInput.type === 'any') return true;
  if (sourceOutput.type === targetInput.type) return true;
  
  // Allow some compatible types
  const compatibleTypes = {
    'string': ['text'],
    'text': ['string'], 
    'number': ['any'],
    'object': ['any'],
    'array': ['any']
  };
  
  const sourceCompatible = compatibleTypes[sourceOutput.type] || [];
  const targetCompatible = compatibleTypes[targetInput.type] || [];
  
  const isCompatible = sourceCompatible.includes(targetInput.type) || targetCompatible.includes(sourceOutput.type);
  
  return isCompatible;
};

// Utility function to export node schema for backend use
const exportNodeSchema = () => {
  const schema: Record<string, any> = {};
  
  Object.entries(legacyNodeCategories).forEach(([category, templates]) => {
    templates.forEach(template => {
      schema[template.type] = {
        label: template.label,
        description: template.description,
        category: category,
        color: template.color,
        inputs: template.inputs || [],
        outputs: template.outputs || [],
        // Map to backend node type names
        backendType: `Jarvis${template.type.charAt(0).toUpperCase() + template.type.slice(1)}Node`
      };
    });
  });
  
  return schema;
};

// Helper function to get icon for layout algorithm
const getLayoutIcon = (algorithm: string) => {
  switch (algorithm) {
    case 'hierarchical':
      return <HierarchicalIcon />;
    case 'force-directed':
      return <ForceDirectedIcon />;
    case 'grid':
      return <GridIcon />;
    case 'smart':
      return <SmartIcon />;
    case 'execution-sequence':
      return <TimelineIcon />;
    case 'execution-force-directed':
      return <ForceDirectedIcon />;
    case 'execution-layered':
      return <LayersIcon />;
    case 'execution-semantic':
      return <CategoryIcon />;
    default:
      return <SettingsIcon />;
  }
};

// Enhanced auto-layout function with ReactFlow collision detection
const getLayoutedElements = (
  nodes: Node[], 
  edges: Edge[], 
  options: Partial<LayoutOptions> = {},
  getIntersectingNodes?: (node: Node) => Node[]
) => {
  const layoutOptions: LayoutOptions = {
    ...defaultLayoutOptions,
    ...options
  };
  
  return applyAdvancedLayout(nodes, edges, layoutOptions, getIntersectingNodes);
};

// Safe fallback function - always uses grid to prevent clustering
const getLayoutedElementsLegacy = (nodes: Node[], edges: Edge[], direction = 'TB') => {
  return getLayoutedElements(nodes, edges, { 
    direction: direction as 'TB' | 'BT' | 'LR' | 'RL',
    algorithm: 'grid'
  });
};

// Helper function to convert agent schemas to LangGraph-based node categories
const convertAgentSchemasToNodeCategories = (agentSchemas: any, categories: any[]) => {
  if (!agentSchemas || !categories) return {};
  
  // LangGraph-inspired workflow pattern categories
  const langGraphCategories: any = {
    'Core Workflow': [],
    'Agent Execution': [],
    'Flow Control': []
  };
  
  // Convert each schema to node template format with LangGraph patterns
  Object.entries(agentSchemas).forEach(([key, schema]: [string, any]) => {
    const template = {
      type: schema.backendType.toLowerCase(),
      label: schema.label,
      icon: getIconForNodeType(schema.backendType),
      color: schema.color,
      description: schema.description,
      inputs: schema.inputs || [],
      outputs: schema.outputs || [],
      properties: schema.properties || {},
      workflowPattern: getWorkflowPattern(schema.backendType)
    };
    
    // Categorize based on LangGraph workflow patterns
    const category = categorizeByLangGraphPattern(schema.backendType);
    if (langGraphCategories[category]) {
      langGraphCategories[category].push(template);
    }
  });
  
  // Add traditional Start/End nodes to agent-based workflows for backward compatibility
  langGraphCategories['Core Workflow'].unshift(
    {
      type: 'start',
      label: 'Start',
      icon: <PlayIcon />,
      color: '#4caf50',
      description: 'Traditional workflow starting point',
      inputs: [],
      outputs: [
        { id: 'output', label: 'Output', type: 'any', description: 'Initial workflow data' }
      ],
      workflowPattern: 'workflow_start'
    },
    {
      type: 'end',
      label: 'End',
      icon: <CloseIcon />,
      color: '#f44336',
      description: 'Traditional workflow endpoint',
      inputs: [
        { id: 'input', label: 'Input', type: 'any', description: 'Final workflow result' }
      ],
      outputs: [],
      workflowPattern: 'workflow_end'
    }
  );
  
  return langGraphCategories;
};

// Helper function to determine workflow pattern for each node type
const getWorkflowPattern = (nodeType: string): string => {
  const patternMap: any = {
    'AgentNode': 'agent_execution',
    'InputNode': 'workflow_start', 
    'OutputNode': 'workflow_end',
    'ConditionNode': 'routing',
    'ParallelNode': 'parallelization',
    'StateNode': 'state_management'
  };
  return patternMap[nodeType] || 'general';
};

// Helper function to categorize nodes by LangGraph workflow patterns
const categorizeByLangGraphPattern = (nodeType: string): string => {
  const categoryMap: any = {
    'AgentNode': 'Agent Execution',
    'InputNode': 'Core Workflow',
    'OutputNode': 'Core Workflow', 
    'ConditionNode': 'Flow Control',
    'ParallelNode': 'Flow Control',
    'StateNode': 'Core Workflow'
  };
  return categoryMap[nodeType] || 'Core Workflow';
};

// Helper function to get appropriate icon for node types
const getIconForNodeType = (nodeType: string) => {
  const iconMap: any = {
    'AgentNode': <PsychologyIcon />,
    'InputNode': <AddIcon />,
    'OutputNode': <CloseIcon />,
    'ConditionNode': <SettingsIcon />,
    'ParallelNode': <LoopIcon />,
    'StateNode': <StorageIcon />
  };
  return iconMap[nodeType] || <BuildIcon />;
};

const CustomWorkflowEditor: React.FC<CustomWorkflowEditorProps> = ({
  open = true,
  onClose,
  workflowId,
  workflowName,
  onSave,
  initialWorkflow,
  onBack,
  isNewWorkflow = false,
}) => {
  // Remove debug logs
  
  const [nodes, setNodes, baseOnNodesChange] = useNodesState([]);
  
  // DEBUG: Remove debug logs
  const [edges, setEdges, baseOnEdgesChange] = useEdgesState([]);
  
  // Wrapped change handler for edges to track unsaved changes
  const onEdgesChange = useCallback((changes: any) => {
    baseOnEdgesChange(changes);
    setHasUnsavedChanges(true);
  }, [baseOnEdgesChange]);
  
  // Wrapped setters to track changes
  const setNodesWithChangeTracking = useCallback((newNodes: any) => {
    setNodes(newNodes);
    // Don't set unsaved changes for initial load
    if (nodes.length > 0 || typeof newNodes === 'function') {
      setHasUnsavedChanges(true);
    }
  }, [setNodes, nodes.length]);
  
  const setEdgesWithChangeTracking = useCallback((newEdges: any) => {
    setEdges(newEdges);
    // Don't set unsaved changes for initial load
    if (edges.length > 0 || typeof newEdges === 'function') {
      setHasUnsavedChanges(true);
    }
  }, [setEdges, edges.length]);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [selectedNodeType, setSelectedNodeType] = useState<string | null>(null);
  const [nodeCounter, setNodeCounter] = useState(1);
  const [currentWorkflowName, setCurrentWorkflowName] = useState(workflowName || 'New Workflow');
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [executionError, setExecutionError] = useState<string | null>(null);
  
  // Agent-based workflow states
  const [agentNodeSchemas, setAgentNodeSchemas] = useState<any>(null);
  const [nodeCategories, setNodeCategories] = useState<any[]>([]);
  const [isLoadingSchemas, setIsLoadingSchemas] = useState(true);
  const [workflowType, setWorkflowType] = useState<'legacy' | 'agent_based'>('agent_based');
  
  // Workflow completion states
  const [noteViewerOpen, setNoteViewerOpen] = useState(false);
  const [completionDialogOpen, setCompletionDialogOpen] = useState(false);
  const [workflowResult, setWorkflowResult] = useState<{
    content: string;
    format: string;
    metadata?: any;
  } | null>(null);
  
  // Execution states
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionMessage, setExecutionMessage] = useState('');
  const [showExecutionPanel, setShowExecutionPanel] = useState(false);
  const [executionLogs, setExecutionLogs] = useState<any[]>([]);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [isAutoSaving, setIsAutoSaving] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [saveTimeoutRef, setSaveTimeoutRef] = useState<NodeJS.Timeout | null>(null);
  
  // I/O Display states
  const [showAllIO, setShowAllIO] = useState(false);
  const showAllIORef = useRef(showAllIO);

  // Enhanced Layout states
  const [layoutOptions, setLayoutOptions] = useState<LayoutOptions>(defaultLayoutOptions);
  const [isLayoutAnimating, setIsLayoutAnimating] = useState(false);
  const [layoutMenuAnchor, setLayoutMenuAnchor] = useState<null | HTMLElement>(null);
  
  // Flow Direction Animation state
  const [dropdownOpenCount, setDropdownOpenCount] = useState(0);
  const [accordionOpenCount, setAccordionOpenCount] = useState(0);
  const [showFlowAnimation, setShowFlowAnimation] = useState(false);
  
  // Accordion state for dynamic zoom control
  const [accordionOpen, setAccordionOpen] = useState(false);
  
  // Grouping states
  const [selectedNodeIds, setSelectedNodeIds] = useState<string[]>([]);
  
  // Copy/Paste states
  const [copiedNodes, setCopiedNodes] = useState<Node[]>([]);
  const [nodeContextMenu, setNodeContextMenu] = useState<{
    nodeId: string;
    x: number;
    y: number;
  } | null>(null);

  // Update existing edges when animation state changes
  useEffect(() => {
    setEdges((eds) =>
      eds.map((edge) => ({
        ...edge,
        animated: showFlowAnimation,
      }))
    );
  }, [showFlowAnimation, setEdges]);


  // Function to calculate and assign sequence information to agent nodes
  const calculateAndAssignSequenceInfo = useCallback((currentNodes: Node[], currentEdges: Edge[]) => {
    // Calculate execution sequence using the same logic as the backend
    const executionOrder = calculateExecutionSequence(currentNodes, currentEdges);
    
    // Find agent nodes and assign sequence information
    const agentNodes = currentNodes.filter(node => 
      node.type === 'agent' || node.type === 'agentnode'
    );
    
    if (agentNodes.length === 0) return currentNodes;
    
    // Create a map of node IDs to their execution order index
    const sequenceMap = new Map<string, number>();
    let agentSequenceIndex = 0;
    
    // Only count agent nodes in the sequence
    executionOrder.forEach(nodeId => {
      const node = currentNodes.find(n => n.id === nodeId);
      if (node && (node.type === 'agent' || node.type === 'agentnode')) {
        sequenceMap.set(nodeId, agentSequenceIndex);
        agentSequenceIndex++;
      }
    });
    
    // Update nodes with sequence information
    return currentNodes.map(node => {
      if (node.type === 'agent' || node.type === 'agentnode') {
        const agentIndex = sequenceMap.get(node.id);
        const sequenceInfo = agentIndex !== undefined ? {
          agentIndex: agentIndex,
          totalAgents: agentNodes.length,
          sequenceDisplay: `${agentIndex + 1}/${agentNodes.length}`
        } : {
          agentIndex: 0,
          totalAgents: agentNodes.length,
          sequenceDisplay: `1/${agentNodes.length}`
        };
        
        return {
          ...node,
          data: {
            ...node.data,
            ...sequenceInfo
          }
        };
      }
      return node;
    });
  }, []);

  // Function to update nodes with sequence information
  const updateNodesWithSequenceInfo = useCallback(() => {
    setNodes((currentNodes) => {
      return calculateAndAssignSequenceInfo(currentNodes, edges);
    });
  }, [edges, calculateAndAssignSequenceInfo, setNodes]);

  // Update sequence information whenever edges change
  useEffect(() => {
    if (nodes.length > 0 && edges.length >= 0) {
      // Small delay to ensure edge state is fully updated
      const timeoutId = setTimeout(() => {
        updateNodesWithSequenceInfo();
      }, 100);
      
      return () => clearTimeout(timeoutId);
    }
  }, [edges, nodes.length, updateNodesWithSequenceInfo]);

  // Reset workflow state when workflowId changes (switching between create new and edit existing)
  useEffect(() => {
    if (open || !onClose) { // Either dialog is open or standalone mode
      // Priority: workflowId > initialWorkflow > new workflow
      if (workflowId) {
        // If we have a workflowId, it will be loaded by the separate data loading useEffect
        // Don't do anything here, let the database loading handle it
      } else if (initialWorkflow) {
        // If we have an initial workflow but no workflowId, load it
        const { nodes: initialNodes = [], edges: initialEdges = [] } = initialWorkflow;
        
        // Initialize RouterNodes with connected nodes
        const nodesWithRouterConnections = initialNodes.map((node: any) => {
          if (node.type === 'routernode' || node.type === 'router') {
            const connectedNodeIds = initialEdges
              .filter((edge: any) => edge.source === node.id)
              .map((edge: any) => edge.target);
            
            return {
              ...node,
              data: {
                ...node.data,
                connectedNodes: connectedNodeIds
              }
            };
          }
          return node;
        });
        
        setNodes(nodesWithRouterConnections);
        setEdges(initialEdges);
        setCurrentWorkflowName(workflowName || initialWorkflow.name || 'Untitled Workflow');
        
        // Find the highest node counter
        let maxCounter = 0;
        initialNodes.forEach((node: any) => {
          const match = node.id.match(/-(\d+)$/);
          if (match) {
            const num = parseInt(match[1]);
            if (num > maxCounter) maxCounter = num;
          }
        });
        setNodeCounter(maxCounter + 1);
      } else {
        // If no workflowId and no initial workflow, we're creating a new workflow
        setNodes([]);
        setEdges([]);
        setNodeCounter(1);
        setCurrentWorkflowName(workflowName || 'New Workflow');
        setExecutionError(null);
        setExecutionLogs([]);
        setShowExecutionPanel(false);
      }
    }
  }, [workflowId, open, initialWorkflow]);
  
  // Handle expand/collapse all I/O
  const handleToggleAllIO = useCallback(() => {
    const newShowAllIO = !showAllIO;
    setShowAllIO(newShowAllIO);
    showAllIORef.current = newShowAllIO;
    
    // Update all nodes to include the showIO property
    setNodes((nds) => 
      nds.map(node => ({
        ...node,
        data: {
          ...node.data,
          showIO: newShowAllIO
        }
      }))
    );
  }, [showAllIO, setNodes]);

  // Handle node context menu (right-click)
  const handleNodeContextMenu = useCallback((event: React.MouseEvent, node: Node) => {
    event.preventDefault();
    event.stopPropagation();
    
    // Select the node if not already selected
    if (!selectedNodeIds.includes(node.id)) {
      setSelectedNodeIds([node.id]);
    }
    
    setNodeContextMenu({
      nodeId: node.id,
      x: event.clientX,
      y: event.clientY
    });
  }, [selectedNodeIds]);

  // Close context menu
  const closeNodeContextMenu = useCallback(() => {
    setNodeContextMenu(null);
  }, []);

  // Copy selected nodes to clipboard
  const copyNodes = useCallback(() => {
    const nodesToCopy = nodes.filter(node => selectedNodeIds.includes(node.id));
    if (nodesToCopy.length > 0) {
      setCopiedNodes(nodesToCopy);
    }
  }, [nodes, selectedNodeIds]);

  // Duplicate selected nodes
  const duplicateNodes = useCallback(() => {
    const nodesToDuplicate = nodes.filter(node => selectedNodeIds.includes(node.id));
    if (nodesToDuplicate.length === 0) return;

    const newNodes: Node[] = [];
    const idMapping: Record<string, string> = {};
    
    // Create new nodes with offset positions
    nodesToDuplicate.forEach(node => {
      const newId = `${node.type}-${nodeCounter + newNodes.length}`;
      idMapping[node.id] = newId;
      
      const newNode: Node = {
        ...node,
        id: newId,
        position: {
          x: node.position.x + 50,
          y: node.position.y + 50
        },
        data: {
          ...node.data,
          // Clear execution data for duplicated nodes
          executionData: { status: 'idle' },
          // Clear any node-specific IDs or references
          label: node.data.label ? `${node.data.label} (Copy)` : undefined
        }
      };
      
      newNodes.push(newNode);
    });
    
    // Add the new nodes
    setNodesWithChangeTracking((currentNodes) => {
      const updatedNodes = [...currentNodes, ...newNodes];
      return calculateAndAssignSequenceInfo(updatedNodes, edges);
    });
    
    // Update node counter
    setNodeCounter(prev => prev + newNodes.length);
    
    // Select the new nodes
    setSelectedNodeIds(newNodes.map(n => n.id));
    
  }, [nodes, selectedNodeIds, nodeCounter, setNodesWithChangeTracking, edges, calculateAndAssignSequenceInfo]);

  // Paste copied nodes
  const pasteNodes = useCallback((position?: { x: number; y: number }) => {
    if (copiedNodes.length === 0) return;

    const newNodes: Node[] = [];
    const idMapping: Record<string, string> = {};
    
    // Calculate bounding box of copied nodes
    const minX = Math.min(...copiedNodes.map(n => n.position.x));
    const minY = Math.min(...copiedNodes.map(n => n.position.y));
    
    // Default paste position or use provided position
    const pasteX = position?.x || minX + 100;
    const pasteY = position?.y || minY + 100;
    
    // Create new nodes
    copiedNodes.forEach(node => {
      const newId = `${node.type}-${nodeCounter + newNodes.length}`;
      idMapping[node.id] = newId;
      
      const newNode: Node = {
        ...node,
        id: newId,
        position: {
          x: node.position.x - minX + pasteX,
          y: node.position.y - minY + pasteY
        },
        data: {
          ...node.data,
          // Clear execution data for pasted nodes
          executionData: { status: 'idle' },
          // Update label if it exists
          label: node.data.label
        }
      };
      
      newNodes.push(newNode);
    });
    
    // Add the new nodes
    setNodesWithChangeTracking((currentNodes) => {
      const updatedNodes = [...currentNodes, ...newNodes];
      return calculateAndAssignSequenceInfo(updatedNodes, edges);
    });
    
    // Update node counter
    setNodeCounter(prev => prev + newNodes.length);
    
    // Select the new nodes
    setSelectedNodeIds(newNodes.map(n => n.id));
    
  }, [copiedNodes, nodeCounter, setNodesWithChangeTracking, edges, calculateAndAssignSequenceInfo]);

  // Keyboard shortcuts for copy/paste
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Check if the workflow editor is open and focused
      if (!open) return;
      
      // Check if user is typing in an input field
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return;
      
      // Copy (Ctrl/Cmd + C)
      if ((e.ctrlKey || e.metaKey) && e.key === 'c' && selectedNodeIds.length > 0) {
        e.preventDefault();
        copyNodes();
      }
      
      // Paste (Ctrl/Cmd + V)
      if ((e.ctrlKey || e.metaKey) && e.key === 'v' && copiedNodes.length > 0) {
        e.preventDefault();
        pasteNodes();
      }
      
      // Duplicate (Ctrl/Cmd + D)
      if ((e.ctrlKey || e.metaKey) && e.key === 'd' && selectedNodeIds.length > 0) {
        e.preventDefault();
        duplicateNodes();
      }
      
      // Delete (Delete or Backspace)
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedNodeIds.length > 0) {
        e.preventDefault();
        // Delete selected nodes
        setNodesWithChangeTracking((nds) => nds.filter(node => !selectedNodeIds.includes(node.id)));
        setEdgesWithChangeTracking((eds) => eds.filter(edge => 
          !selectedNodeIds.includes(edge.source) && !selectedNodeIds.includes(edge.target)
        ));
        setSelectedNodeIds([]);
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [open, selectedNodeIds, copiedNodes, copyNodes, pasteNodes, duplicateNodes, setNodesWithChangeTracking, setEdgesWithChangeTracking]);

  // Process execution logs and update node execution data
  const processExecutionLogForNodes = useCallback((log: any) => {
    
    // Handle individual agent/node execution updates
    if (log.type === 'agent_start' || log.type === 'node_start' || log.type === 'agent_execution_start') {
      if (log.node_id) {
        setNodes(nds => nds.map(node => {
          if (node.id === log.node_id) {
            return {
              ...node,
              data: {
                ...node.data,
                executionData: {
                  ...node.data.executionData,
                  status: 'running',
                  timestamp: log.timestamp
                }
              }
            };
          }
          return node;
        }));
      }
    }

    // Force ParallelNode animation on parallel_execution_start
    if (log.type === 'parallel_execution_start') {
      if (log.parallel_id) {
        setNodes(nds => nds.map(node => {
          if (node.id === log.parallel_id) {
            return {
              ...node,
              data: {
                ...node.data,
                executionData: {
                  status: 'running',
                  timestamp: log.timestamp
                }
              }
            };
          }
          return node;
        }));
      }
    }
    
    if (log.type === 'agent_complete' || log.type === 'node_complete' || log.type === 'agent_result' || log.type === 'agent_execution_complete') {
      if (log.node_id) {
        setNodes(nds => nds.map(node => {
          if (node.id === log.node_id) {
            return {
              ...node,
              data: {
                ...node.data,
                executionData: {
                  input: log.input || node.data.executionData?.input,
                  output: log.output || log.response || log.result,
                  tools_used: log.tools_used || node.data.executionData?.tools_used,
                  status: 'success',
                  timestamp: log.timestamp,
                  error: null
                }
              }
            };
          }
          return node;
        }));
      }
    }
    
    // Handle parallel execution completion
    if (log.type === 'parallel_execution_complete') {
      if (log.parallel_id) {
        setNodes(nds => nds.map(node => {
          if (node.id === log.parallel_id) {
            return {
              ...node,
              data: {
                ...node.data,
                executionData: {
                  ...node.data.executionData,
                  status: 'success',
                  timestamp: log.timestamp,
                  results: log.results,
                  combined_output: log.combined_output,
                  summary: log.summary,
                  completed_count: log.completed_count,
                  total_count: log.total_count,
                  strategy_used: log.strategy_used,
                  error: null
                }
              }
            };
          }
          return node;
        }));
      }
    }
    
    if (log.type === 'agent_error' || log.type === 'node_error' || log.type === 'agent_execution_error') {
      if (log.node_id) {
        setNodes(nds => nds.map(node => {
          if (node.id === log.node_id) {
            return {
              ...node,
              data: {
                ...node.data,
                executionData: {
                  ...node.data.executionData,
                  status: 'error',
                  timestamp: log.timestamp,
                  error: log.error || log.message || 'Unknown error'
                }
              }
            };
          }
          return node;
        }));
      }
    }
    
    // Handle router decision events
    if (log.type === 'router_decision') {
      if (log.router_id) {
        setNodes(nds => nds.map(node => {
          if (node.id === log.router_id) {
            return {
              ...node,
              data: {
                ...node.data,
                executionData: {
                  ...node.data.executionData,
                  matched_routes: log.matched_routes,
                  target_nodes: log.target_nodes,
                  timestamp: log.timestamp
                }
              }
            };
          }
          return node;
        }));
      }
    }
    
    // Handle condition evaluation events
    if (log.type === 'condition_evaluation_complete') {
      if (log.condition_id) {
        setNodes(nds => nds.map(node => {
          if (node.id === log.condition_id) {
            return {
              ...node,
              data: {
                ...node.data,
                executionData: {
                  ...node.data.executionData,
                  result: log.result,
                  branch: log.branch,
                  evaluation_details: log.evaluation_details,
                  timestamp: log.timestamp
                }
              }
            };
          }
          return node;
        }));
      }
    }
    
    // Force stop all ParallelNode animations when workflow completes
    if (log.type === 'workflow_result') {
      setNodes(nds => nds.map(node => {
        if (node.type === 'parallelnode') {
          return {
            ...node,
            data: {
              ...node.data,
              executionData: {
                ...node.data.executionData,
                status: 'success'
              }
            }
          };
        }
        return node;
      }));
    }
    
    if (log.type === 'workflow_result') {
      
      // Find output node to get format configuration
      const outputNode = nodes.find(node => 
        node.data.type === 'OutputNode' || 
        node.type === 'outputnode' ||
        (node.data.node && node.data.node.type === 'OutputNode')
      );
      
      
      // Prioritize output format from backend (saved database values) over frontend UI
      const outputFormat = log.output_config?.output_format ||
                          outputNode?.data?.output_format || 
                          outputNode?.data?.node?.output_format || 
                          'text';
      
      // Set workflow result and check auto-display setting
      const workflowData = {
        content: log.response || log.result || 'Workflow completed successfully.',
        format: outputFormat,
        metadata: {
          workflowName: currentWorkflowName,
          executionTime: Date.now() - (new Date(log.timestamp).getTime() - 5000), // Rough calculation
          generatedAt: new Date().toISOString(),
          executionId: log.execution_id,
          workflowId: log.workflow_id
        }
      };
      
      setWorkflowResult(workflowData);
      
      // Update OutputNode to show report is available and populate execution data
      setNodes(nds => nds.map(node => {
        if (node.data.type === 'OutputNode' || 
            node.type === 'outputnode' ||
            (node.data.node && node.data.node.type === 'OutputNode')) {
          return {
            ...node,
            data: {
              ...node.data,
              hasReport: true,
              lastReportTimestamp: workflowData.metadata.generatedAt,
              executionData: {
                status: log.error ? 'error' : 'success',
                output: workflowData.content,
                metadata: workflowData.metadata,
                error: log.error
              }
            }
          };
        }
        return node;
      }));
      
      // Check OutputNode configuration for auto-behavior from backend result
      
      // Prioritize configuration from backend (saved database values) over frontend UI
      const autoDisplay = outputNode ? (
        log.output_config?.auto_display ??
        outputNode?.data?.auto_display ??
        outputNode?.data?.node?.auto_display ??
        false  // Default to false - only show modal if explicitly enabled
      ) : false;
      const autoSave = log.output_config?.auto_save ??
                      outputNode?.data?.auto_save ??
                      outputNode?.data?.node?.auto_save ??
                      false; // Default to false if no configuration found
      
      
      if (autoDisplay && autoSave) {
        // Both auto actions
        setNoteViewerOpen(true);
        downloadWorkflowOutput(
          workflowData.content,
          workflowData.format as any,
          workflowData.metadata,
          true // autoSave = true
        );
      } else if (autoDisplay) {
        // Auto display only
        setNoteViewerOpen(true);
      } else if (autoSave) {
        // Auto save only
        downloadWorkflowOutput(
          workflowData.content,
          workflowData.format as any,
          workflowData.metadata,
          true // autoSave = true
        );
      } else if (outputNode) {
        // If both auto options are disabled, show completion dialog for user choice
        setCompletionDialogOpen(true);
      } else {
        // No output node, no completion dialog
      }
      
      setIsExecuting(false);
      return;
    }
    
    // Handle cache-specific events for CacheNode components
    if (log.type === 'cache_hit' || log.type === 'cache_connection') {
      setNodes(nodes =>
        nodes.map(node => {
          // Update CacheNode with cache hit information
          if (node.data.type === 'CacheNode' && (
            node.id === log.cache_id || 
            node.id === log.source_cache_node ||
            node.id === log.node_id
          )) {
            return {
              ...node,
              data: {
                ...node.data,
                executionData: {
                  ...node.data.executionData,
                  status: 'success',
                  cacheHit: log.type === 'cache_hit',
                  cacheKey: log.cache_key,
                  cacheSize: log.cache_size,
                  cacheMetadata: log.cache_metadata,
                  timestamp: log.timestamp
                }
              }
            };
          }
          return node;
        })
      );
    }
    
    // Find the corresponding node based on node_id (preferred) or agent name
    if (log.node_id || log.agent_name) {
      setNodes((nds) =>
        nds.map((node) => {
          // CRITICAL FIX: Match by node_id first (for proper visual updates), fallback to agent_name
          const nodeMatches = log.node_id ? 
            node.id === log.node_id : 
            (node.data.agentName || node.data.agent_name) === log.agent_name;
          
          if (nodeMatches) {
            const newExecutionData = { ...node.data.executionData };
            
            // Update execution data based on log type
            switch (log.type) {
              case 'agent_execution_start':
                newExecutionData.status = 'running';
                newExecutionData.timestamp = log.timestamp;
                break;
                
              case 'agent_execution_complete':
                newExecutionData.status = 'success';
                newExecutionData.input = log.input;
                newExecutionData.output = log.output;
                newExecutionData.tools_used = log.tools_used || [];
                newExecutionData.timestamp = log.timestamp;
                break;
                
              case 'agent_execution_error':
                newExecutionData.status = 'error';
                newExecutionData.error = log.error;
                newExecutionData.timestamp = log.timestamp;
                break;
                
              case 'cache_hit':
                newExecutionData.status = 'success';
                newExecutionData.cached = true;
                newExecutionData.cacheHit = true;
                newExecutionData.cacheKey = log.cache_key;
                newExecutionData.cacheSize = log.cache_size;
                newExecutionData.cacheMetadata = log.cache_metadata;
                newExecutionData.output = log.cached_data;
                newExecutionData.timestamp = log.timestamp;
                break;
                
              case 'cache_connection':
                // Handle direct cache connections to agents
                newExecutionData.cached_input = {
                  source_cache_node: log.source_cache_node,
                  cached_data: log.cached_data
                };
                break;
            }
            
            return {
              ...node,
              data: {
                ...node.data,
                executionData: newExecutionData,
                // Update sequence information from streaming message (if provided)
                // Otherwise preserve the existing sequence information from frontend calculation
                agentIndex: log.agent_index !== undefined ? log.agent_index : node.data.agentIndex,
                totalAgents: log.total_agents !== undefined ? log.total_agents : node.data.totalAgents,
                sequenceDisplay: log.sequence_display || node.data.sequenceDisplay
              }
            };
          }
          return node;
        })
      );
    }
  }, [setNodes, nodes, currentWorkflowName]);

  // Workflow completion handlers
  const handleCloseNoteViewer = useCallback(() => {
    setNoteViewerOpen(false);
  }, []);

  const handleViewNow = useCallback(() => {
    setCompletionDialogOpen(false);
    setNoteViewerOpen(true);
  }, []);

  const handleSaveOnly = useCallback(() => {
    if (workflowResult) {
      downloadWorkflowOutput(
        workflowResult.content,
        workflowResult.format as any,
        workflowResult.metadata,
        false // manual save, show dialog
      );
    }
    setCompletionDialogOpen(false);
  }, [workflowResult]);

  const handleViewAndSave = useCallback(() => {
    if (workflowResult) {
      downloadWorkflowOutput(
        workflowResult.content,
        workflowResult.format as any,
        workflowResult.metadata,
        false // manual save, show dialog
      );
    }
    setCompletionDialogOpen(false);
    setNoteViewerOpen(true);
  }, [workflowResult]);

  const handleViewLater = useCallback(() => {
    setCompletionDialogOpen(false);
    // Report remains available via OutputNode click
  }, []);

  const handleDismiss = useCallback(() => {
    setCompletionDialogOpen(false);
  }, []);

  const handleOutputNodeViewReport = useCallback(() => {
    if (workflowResult) {
      setNoteViewerOpen(true);
    }
  }, [workflowResult]);
  
  // Add event listener for OutputNode view report
  useEffect(() => {
    const handleViewReport = () => {
      if (workflowResult) {
        setNoteViewerOpen(true);
      }
    };
    
    const workflowEditor = document.querySelector('[data-testid="workflow-editor"]');
    if (workflowEditor) {
      workflowEditor.addEventListener('viewWorkflowReport', handleViewReport);
      return () => {
        workflowEditor.removeEventListener('viewWorkflowReport', handleViewReport);
      };
    }
  }, [workflowResult]);

  // Reset all node execution data
  const resetAllNodeExecutionData = useCallback(() => {
    setNodes((nds) =>
      nds.map((node) => ({
        ...node,
        data: {
          ...node.data,
          executionData: { status: 'idle' }
        }
      }))
    );
  }, [setNodes]);

  // Function to update node data
  const updateNodeData = useCallback((nodeId: string, newData: any) => {
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id === nodeId) {
          const updatedNode = { ...node, data: { ...node.data, ...newData } };
          return updatedNode;
        }
        return node;
      })
    );
  }, [setNodes]);

  // Update RouterNode connected nodes when edges change
  useEffect(() => {
    // Find all router nodes
    const routerNodes = nodes.filter(node => 
      node.type === 'routernode' || node.type === 'router'
    );

    // For each router node, find connected target nodes
    routerNodes.forEach(routerNode => {
      const connectedNodeIds = edges
        .filter(edge => edge.source === routerNode.id)
        .map(edge => edge.target);

      // Update the router node data with connected nodes if changed
      const currentConnectedNodes = routerNode.data.connectedNodes || [];
      if (JSON.stringify(currentConnectedNodes.sort()) !== JSON.stringify(connectedNodeIds.sort())) {
        updateNodeData(routerNode.id, {
          ...routerNode.data,
          connectedNodes: connectedNodeIds
        });
      }
    });
  }, [edges, nodes, updateNodeData]);

  // Fetch agent-based node schemas from backend
  const fetchAgentNodeSchemas = useCallback(async () => {
    try {
      setIsLoadingSchemas(true);
      const response = await fetch('http://127.0.0.1:8000/api/v1/automation/schema/agent-nodes');
      
      if (response.ok) {
        const data = await response.json();
        
        setAgentNodeSchemas(data.schema);
        setNodeCategories(data.categories || []);
        setIsLoadingSchemas(false);
      } else {
        // Fallback to legacy mode if API fails
        setWorkflowType('legacy');
        setIsLoadingSchemas(false);
      }
    } catch (error) {
      // Fallback to legacy mode if API fails
      setWorkflowType('legacy');
      setIsLoadingSchemas(false);
    }
  }, []);

  // Dynamic node categories based on workflow type
  const currentNodeCategories = useMemo(() => {
    if (workflowType === 'agent_based' && agentNodeSchemas && nodeCategories.length > 0) {
      const convertedCategories = convertAgentSchemasToNodeCategories(agentNodeSchemas, nodeCategories);
      return convertedCategories;
    }
    // Fallback to legacy node categories (keeping original structure for backward compatibility)
    return legacyNodeCategories;
  }, [workflowType, agentNodeSchemas, nodeCategories]);

  // Dynamic helper functions for current node categories
  const getCurrentNodeTemplates = useCallback(() => {
    const templates = Object.values(currentNodeCategories).flat();
    return templates;
  }, [currentNodeCategories]);

  const getCurrentNodeTemplate = useCallback((nodeType: string) => {
    const templates = getCurrentNodeTemplates();
    return templates.find((template: any) => template.type === nodeType);
  }, [getCurrentNodeTemplates]);

  const getCurrentNodeHandles = useCallback((nodeType: string) => {
    const template = getCurrentNodeTemplate(nodeType);
    if (!template) return { inputs: [], outputs: [] };
    
    return {
      inputs: template.inputs || [],
      outputs: template.outputs || []
    };
  }, [getCurrentNodeTemplate]);

  // Create nodeTypes with updateNodeData callback
  // Note: We use a factory function and getter to avoid recreating nodeTypes
  const nodeTypes: NodeTypes = useMemo(() => 
    createNodeTypes(
      updateNodeData, 
      () => showAllIORef.current,
      handleOutputNodeViewReport,
      setAccordionOpen
    ), 
    [updateNodeData, handleOutputNodeViewReport, setAccordionOpen]
  );

  // Create edgeTypes with custom edge components for better direction visibility
  const edgeTypes = useMemo(() => ({
    directional: DirectionalEdge,
    default: DirectionalEdge, // Use directional edge as default for better UX
  }), []);

  // Fetch agent schemas when dialog opens
  useEffect(() => {
    if (open) {
      fetchAgentNodeSchemas();
    }
  }, [open, fetchAgentNodeSchemas]);

  // Initialize with appropriate nodes based on workflow type
  useEffect(() => {
    if (open && nodes.length === 0 && !workflowId) {
      // Force create default nodes regardless of loading state
      const initialNodes: Node[] = [
        {
          id: 'start-1',
          type: 'start',
          position: { x: 100, y: 50 },
          data: { 
            label: 'Start',
            inputHandles: [],
            outputHandles: [],
            nodeTemplate: {}
          },
          draggable: true,
        },
        {
          id: 'end-1',
          type: 'end',
          position: { x: 100, y: 300 },
          data: { 
            label: 'End',
            inputHandles: [],
            outputHandles: [],
            nodeTemplate: {}
          },
          draggable: true,
        },
      ];
      
      setNodes(initialNodes);
      setNodeCounter(2);
    }
  }, [open, nodes.length, setNodes, workflowId]);

  // Load existing workflow data
  // Clear previous workflow state immediately when opening with new workflowId
  // Add previousWorkflowId ref to prevent unnecessary resets
  const previousWorkflowIdRef = useRef<number | null>(null);
  
  useEffect(() => {
    if (open && workflowId) {
      // Only clear state if workflowId has actually changed (not just object reference)
      if (previousWorkflowIdRef.current !== workflowId) {
        // Don't clear nodes and edges - let loadWorkflowData handle it
        setCurrentWorkflowName('Loading...');
        
        // Clear any execution state
        setExecutionError(null);
        setWorkflowResult(null);
        setExecutionLogs([]);
        setCompletionDialogOpen(false);
        setNoteViewerOpen(false);
        
        // Update the ref to track this workflowId
        previousWorkflowIdRef.current = workflowId;
      }
    }
  }, [workflowId, open]);

  // Load workflow data immediately when workflowId is provided
  useEffect(() => {
    if (workflowId) {
      loadWorkflowData(workflowId);
    }
  }, [workflowId]);

  const loadWorkflowData = async (id: number) => {
    try {
      // Loading workflow from database
      const response = await fetch(`http://127.0.0.1:8000/api/v1/automation/workflows/${id}`);
      if (response.ok) {
        const workflow = await response.json();
        // Workflow loaded successfully
        
        const langflowConfig = workflow.langflow_config || {};
        
        setCurrentWorkflowName(workflow.name);
        
        if (langflowConfig.nodes && langflowConfig.edges) {
          
          // DEBUG: Check for CacheNode in raw backend data
          const rawCacheNodes = langflowConfig.nodes.filter(node => 
            node.data?.type === 'CacheNode' || node.type === 'cachenode'
          );
          const rawCacheNodeEdges = langflowConfig.edges.filter(edge => 
            rawCacheNodes.some(n => n.id === edge.source || n.id === edge.target)
          );
          
          // Transform backend format to frontend format for editing
          const frontendNodes = transformNodesFromBackendFormat(langflowConfig.nodes);
          
          // DEBUG: Check for CacheNode after transformation
          const frontendCacheNodes = frontendNodes.filter(node => node.type === 'cachenode');
          
          // Migrate legacy handle IDs to correct directional handle IDs
          const migratedEdges = migrateEdgeHandleIds(langflowConfig.edges, frontendNodes);
          
          // Apply sequence information to agent nodes
          const nodesWithSequenceInfo = calculateAndAssignSequenceInfo(frontendNodes, migratedEdges);
          
          
          // EMERGENCY: Filter edges to prevent ReactFlow errors
          const validEdges = migratedEdges.filter(edge => {
            // DEBUG: Log every edge being processed, especially CacheNode edges
            const isCacheNodeEdge = rawCacheNodes.some(n => n.id === edge.source || n.id === edge.target);
            if (isCacheNodeEdge) {
            }
            
            const sourceNode = nodesWithSequenceInfo.find(n => n.id === edge.source);
            const targetNode = nodesWithSequenceInfo.find(n => n.id === edge.target);
            
            // Skip edges with missing nodes
            if (!sourceNode || !targetNode) {
              if (isCacheNodeEdge) {
              }
              return false;
            }
            
            // Skip edges with legacy handle IDs that would cause ReactFlow errors
            // IMPORTANT: ParallelNode legitimately uses 'input' as its target handle - don't filter it!
            const isLegacyEdge = (
              (edge.targetHandle === 'input' && targetNode?.type !== 'parallelnode') ||
              edge.sourceHandle === 'output'
            );
            
            if (isLegacyEdge) {
              if (isCacheNodeEdge) {
              }
              return false;
            }
            
            // Log if this is a ParallelNode edge that's being preserved
            if (edge.targetHandle === 'input' && targetNode?.type === 'parallelnode') {
            }
            
            if (isCacheNodeEdge) {
            }
            
            return true;
          });
          
          
          // DEBUG: Final check for CacheNode edges
          const finalCacheNodeEdges = validEdges.filter(edge => 
            nodesWithSequenceInfo.some(n => n.type === 'cachenode' && (n.id === edge.source || n.id === edge.target))
          );
          
          // Initialize RouterNodes with connected nodes
          const nodesWithRouterConnections = nodesWithSequenceInfo.map(node => {
            if (node.type === 'routernode' || node.type === 'router') {
              const connectedNodeIds = validEdges
                .filter(edge => edge.source === node.id)
                .map(edge => edge.target);
              
              return {
                ...node,
                data: {
                  ...node.data,
                  connectedNodes: connectedNodeIds
                }
              };
            }
            return node;
          });
          
          setNodes(nodesWithRouterConnections);
          setEdges(validEdges);
          
          // Set workflow type from loaded workflow metadata
          if (langflowConfig.workflow_type) {
            setWorkflowType(langflowConfig.workflow_type);
            console.log(`[WORKFLOW DEBUG] Setting workflow type from metadata: ${langflowConfig.workflow_type}`);
          } else if (langflowConfig.metadata && langflowConfig.metadata.workflow_type) {
            setWorkflowType(langflowConfig.metadata.workflow_type);
            console.log(`[WORKFLOW DEBUG] Setting workflow type from metadata.workflow_type: ${langflowConfig.metadata.workflow_type}`);
          } else {
            // Detect workflow type based on node types for legacy workflows
            const hasAgentNodes = langflowConfig.nodes.some(node => 
              node.type === 'agentnode' || node.data?.type === 'AgentNode'
            );
            const detectedType = hasAgentNodes ? 'agent_based' : 'legacy';
            setWorkflowType(detectedType);
            console.log(`[WORKFLOW DEBUG] Auto-detected workflow type based on nodes: ${detectedType}`);
          }
          
          // Set higher node counter to avoid ID conflicts
          const maxId = Math.max(
            ...frontendNodes.map(node => {
              const match = node.id.match(/(\d+)$/);
              return match ? parseInt(match[1]) : 0;
            }),
            0
          );
          setNodeCounter(maxId + 1);
        } else {
        }
      } else {
      }
    } catch (error) {
    }
  };

  const onConnect = useCallback(
    (params: Connection) => {
      
      // Always allow connections for now - we'll validate at execution time
      if (params.source && params.target) {
        const sourceNode = nodes.find(n => n.id === params.source);
        const targetNode = nodes.find(n => n.id === params.target);
        
        
        if (sourceNode && targetNode) {
          // For agent-based workflows, be very permissive
          if (workflowType === 'agent_based') {
            setEdgesWithChangeTracking((eds) => addEdge(params, eds));
            return;
          }
          
          // For legacy workflows, also be permissive but with some validation
          setEdgesWithChangeTracking((eds) => addEdge(params, eds));
          return;
        } else {
        }
      } else {
      }
    },
    [setEdges, nodes, workflowType]
  );

  // Add new node to the canvas
  const addNode = useCallback((nodeType: string) => {
    const template = getCurrentNodeTemplate(nodeType);
    if (!template) {
      return;
    }

    // Get handle metadata for the node type
    const handles = getCurrentNodeHandles(nodeType);

    // Initialize node data based on workflow type and node properties
    const nodeData: any = {
      label: template?.label || nodeType,
      inputHandles: handles.inputs,
      outputHandles: handles.outputs,
      nodeTemplate: template
    };

    // Add agent-based node specific data
    if (workflowType === 'agent_based') {
      switch (nodeType) {
        case 'agentnode':
          nodeData.agentName = ''; // User must select agent manually
          nodeData.customPrompt = ''; // User must set custom prompt manually  
          nodeData.tools = []; // User must select tools manually
          break;
        case 'inputnode':
          nodeData.inputSchema = template?.properties?.input_schema?.default || {};
          nodeData.defaultValues = template?.properties?.default_values?.default || {};
          break;
        case 'outputnode':
          nodeData.output_format = template?.properties?.output_format?.default || 'text';
          nodeData.include_metadata = template?.properties?.include_metadata?.default || false;
          nodeData.include_tool_calls = template?.properties?.include_tool_calls?.default || false;
          nodeData.auto_display = template?.properties?.auto_display?.default || false;
          nodeData.auto_save = template?.properties?.auto_save?.default || false;
          break;
        case 'conditionnode':
          nodeData.conditionType = template?.properties?.condition_type?.default || 'simple';
          nodeData.operator = template?.properties?.operator?.default || 'equals';
          break;
        case 'parallelnode':
          nodeData.maxParallel = template?.properties?.max_parallel?.default || 3;
          nodeData.waitForAll = template?.properties?.wait_for_all?.default || true;
          nodeData.combineStrategy = template?.properties?.combine_strategy?.default || 'merge';
          break;
        case 'statenode':
          nodeData.stateOperation = template?.properties?.state_operation?.default || 'merge';
          nodeData.persistence = template?.properties?.persistence?.default || true;
          break;
        case 'transformnode':
          nodeData.transformType = template?.properties?.transform_type?.default || 'jsonpath';
          nodeData.expression = template?.properties?.expression?.default || '$';
          nodeData.errorHandling = template?.properties?.error_handling?.default || 'continue';
          nodeData.cacheResults = template?.properties?.cache_results?.default || false;
          break;
        case 'routernode':
          nodeData.routing_mode = template?.properties?.routing_mode?.default || 'multi-select';
          nodeData.match_type = template?.properties?.match_type?.default || 'exact';
          nodeData.routes = template?.properties?.routes?.default || [];
          nodeData.case_sensitive = template?.properties?.case_sensitive?.default || false;
          nodeData.output_field = template?.properties?.output_field?.default || '';
          nodeData.fallback_route = template?.properties?.fallback_route?.default || '';
          break;
        case 'cachenode':
          nodeData.cacheKeyPattern = template?.properties?.cache_key_pattern?.default || 'auto';
          nodeData.ttl = template?.properties?.ttl?.default || 3600;
          nodeData.cachePolicy = template?.properties?.cache_policy?.default || 'always';
          nodeData.cacheNamespace = template?.properties?.cache_namespace?.default || 'default';
          nodeData.showStatistics = template?.properties?.show_statistics?.default ?? true;
          nodeData.enableWarming = template?.properties?.enable_warming?.default || false;
          nodeData.maxCacheSize = template?.properties?.max_cache_size?.default || 10;
          nodeData.invalidateOn = template?.properties?.invalidate_on?.default || ['input_change'];
          break;
      }
    } else {
      // Legacy node specific data
      nodeData.model = nodeType === 'llm' ? 'qwen3:30b-a3b' : undefined;
      nodeData.selectedTools = nodeType === 'llm' ? [] : undefined;
      nodeData.enableTools = nodeType === 'llm' ? true : undefined;
      nodeData.toolName = nodeType === 'mcpTool' ? 'get_datetime' : undefined;
      nodeData.agentName = nodeType === 'agent' ? 'Researcher Agent' : undefined;
      nodeData.operation = nodeType === 'redis' ? 'get' : undefined;
      nodeData.condition = nodeType === 'condition' ? 'equals' : undefined;
    }

    const newNode: Node = {
      id: `${nodeType}-${nodeCounter}`,
      type: nodeType,
      position: { x: 200 + Math.random() * 200, y: 100 + Math.random() * 200 },
      data: nodeData,
      draggable: true,
    };

    setNodesWithChangeTracking((nds) => {
      const updatedNodes = nds.concat(newNode);
      // Apply sequence information when adding new nodes
      return calculateAndAssignSequenceInfo(updatedNodes, edges);
    });
    setNodeCounter(prev => prev + 1);
    setSelectedNodeType(null);
  }, [nodeCounter, setNodesWithChangeTracking, getCurrentNodeTemplate, getCurrentNodeHandles, workflowType, edges, calculateAndAssignSequenceInfo]);

  // Simple auto-spacing using ReactFlow collision detection
  const onLayout = useCallback(async (algorithm?: 'hierarchical' | 'force-directed' | 'grid' | 'smart' | 'execution-sequence' | 'execution-force-directed' | 'execution-layered' | 'execution-semantic') => {
    if (isLayoutAnimating) {
      return;
    }
    
    setIsLayoutAnimating(true);
    
    try {
      // Use simple layout that only fixes overlaps
      const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(
        nodes, 
        edges, 
        { ...layoutOptions, algorithm: algorithm || 'smart' }
        // Note: getIntersectingNodes not available here, will use fallback
      );
      
      
      // Apply sequence information after layout
      const nodesWithSequenceInfo = calculateAndAssignSequenceInfo(layoutedNodes, layoutedEdges);
      
      // Update nodes
      setNodes([...nodesWithSequenceInfo]);
      setEdges([...layoutedEdges]);
      
    } catch (error) {
      // Fallback: keep original positions
      
    } finally {
      setIsLayoutAnimating(false);
    }
  }, [nodes, edges, setNodes, setEdges, layoutOptions, isLayoutAnimating, calculateAndAssignSequenceInfo]);

  // Layout direction change handler
  const onLayoutDirectionChange = useCallback((direction: 'TB' | 'BT' | 'LR' | 'RL') => {
    const newLayoutOptions = { ...layoutOptions, direction };
    setLayoutOptions(newLayoutOptions);
    onLayout();
  }, [layoutOptions, onLayout]);

  // Layout spacing change handler
  const onLayoutSpacingChange = useCallback((spacing: Partial<LayoutOptions['spacing']>) => {
    const newLayoutOptions = {
      ...layoutOptions,
      spacing: { ...layoutOptions.spacing, ...spacing }
    };
    setLayoutOptions(newLayoutOptions);
  }, [layoutOptions]);

  // Helper methods for filtering configuration corruption
  const filterContextCorruption = (context: any) => {
    if (!context || typeof context !== 'object') return {};
    
    const filtered = { ...context };
    
    // Remove hardcoded model specifications and unwanted template defaults
    delete filtered.model; // Removes "claude-3-sonnet-20240229"
    delete filtered.system_prompt; // Remove template system prompts
    
    // Remove hardcoded temperature/timeout if they match common defaults
    if (filtered.temperature === 0.7) delete filtered.temperature;
    if (filtered.timeout === 300 || filtered.timeout === 360) delete filtered.timeout;
    
    return filtered;
  };

  const isUserCustomPrompt = (prompt: string) => {
    if (!prompt) return false;
    
    // Detect template prompts by length and common template phrases
    const templateIndicators = [
      'You are a skilled',
      'You are analyzing',
      'You are a vigilant',
      'You are a results-driven',
      'You are a precise',
      '/NO_THINK'
    ];
    
    // If prompt is very long (>100 chars) or contains template indicators, it's likely corrupted
    if (prompt.length > 100 || templateIndicators.some(indicator => prompt.includes(indicator))) {
      return false;
    }
    
    return true;
  };

  const cleanAgentContext = (context: any) => {
    if (!context || typeof context !== 'object') return {};
    
    const cleaned = { ...context };
    
    // Remove polluted template data from context
    delete cleaned.system_prompt; // Duplicate of custom_prompt
    delete cleaned.model; // Removes "claude-3-sonnet-20240229"
    delete cleaned.temperature; // Use main temperature field
    delete cleaned.timeout; // Use main timeout field
    
    return cleaned;
  };

  // Transform backend nodes to frontend format
  // Helper function to ensure nodes have valid positions
  const ensureNodePosition = (node: any, index: number) => {
    if (!node.position || typeof node.position.x !== 'number' || typeof node.position.y !== 'number') {
      return {
        ...node,
        position: { x: 100 + index * 150, y: 100 + index * 100 }
      };
    }
    return node;
  };

  const transformNodesFromBackendFormat = (backendNodes: any[]) => {
    
    return backendNodes.map((node, index) => {
      const nodeData = node.data;
      
      // DEBUG: Log CacheNode specifically
      if (nodeData?.type === 'CacheNode' || node.type === 'cachenode') {
        // Skip debug logging
      }
      
      
      // Transform based on backend node type
      switch (nodeData.type) {
        case 'JarvisLLMNode':
          return {
            ...node,
            type: 'llm',
            data: {
              label: 'Ollama LLM',
              model: nodeData.node.model,
              prompt: nodeData.node.prompt,
              parameters: nodeData.node.parameters,
              selectedTools: nodeData.node.selected_tools,
              enableTools: nodeData.node.enable_tools
            }
          };
          
        case 'JarvisAgentNode':
          return {
            ...node,
            type: 'agent',
            data: {
              label: nodeData.label || 'AI Agent', // Use saved label instead of hardcoded value
              agentName: nodeData.node.agent_name,
              query: nodeData.node.query,
              context: nodeData.node.context,
              selectedTools: nodeData.node.selected_tools || [],
              enableTools: nodeData.node.enable_tools ?? true,
              // Enhanced state management fields
              stateEnabled: nodeData.node.state_enabled || false,
              stateOperation: nodeData.node.state_operation || 'passthrough',
              outputFormat: nodeData.node.output_format || 'text',
              chainKey: nodeData.node.chain_key || ''
            }
          };
          
        case 'JarvisAIDecisionNode':
          return {
            ...node,
            type: 'aiDecision',
            data: {
              label: 'AI Decision',
              criteria: nodeData.node.criteria,
              decisionPrompt: nodeData.node.decision_prompt,
              confidenceThreshold: nodeData.node.confidence_threshold
            }
          };
          
        case 'JarvisDocumentProcessorNode':
          return {
            ...node,
            type: 'documentProcessor',
            data: {
              label: 'Document Processor',
              documentType: nodeData.node.document_type,
              analysisTask: nodeData.node.analysis_task,
              extractionParams: nodeData.node.extraction_params
            }
          };
          
        case 'JarvisImageProcessorNode':
          return {
            ...node,
            type: 'imageProcessor',
            data: {
              label: 'Image Processor',
              analysisType: nodeData.node.analysis_type,
              analysisTask: nodeData.node.analysis_task,
              imageParams: nodeData.node.image_params
            }
          };
          
        case 'JarvisAudioProcessorNode':
          return {
            ...node,
            type: 'audioProcessor',
            data: {
              label: 'Audio Processor',
              analysisType: nodeData.node.analysis_type,
              analysisTask: nodeData.node.analysis_task,
              audioParams: nodeData.node.audio_params
            }
          };
          
        case 'JarvisMultiModalFusionNode':
          return {
            ...node,
            type: 'multiModalFusion',
            data: {
              label: 'Multi-modal Fusion',
              fusionTask: nodeData.node.fusion_task,
              fusionStrategy: nodeData.node.fusion_strategy,
              outputFormat: nodeData.node.output_format
            }
          };
          
        case 'JarvisContextMemoryNode':
          return {
            ...node,
            type: 'contextMemory',
            data: {
              label: 'Context Memory',
              memoryKey: nodeData.node.memory_key,
              operation: nodeData.node.operation,
              retentionPolicy: nodeData.node.retention_policy
            }
          };
          
        case 'JarvisConditionNode':
          return {
            ...node,
            type: 'condition',
            data: {
              label: 'Condition',
              operator: nodeData.node.operator,
              leftOperand: nodeData.node.left_operand,
              rightOperand: nodeData.node.right_operand
            }
          };
          
        case 'JarvisStartNode':
          return {
            ...node,
            type: 'start',
            data: {
              label: nodeData.node.label || 'Start'
            }
          };
          
        case 'JarvisEndNode':
          return {
            ...node,
            type: 'end',
            data: {
              label: nodeData.node.label || 'End'
            }
          };

        // Agent-based node types
        case 'AgentNode':
          console.log('AgentNode data:', {
            state_enabled: nodeData.node.state_enabled,
            state_operation: nodeData.node.state_operation,
            output_format: nodeData.node.output_format,
            chain_key: nodeData.node.chain_key
          });
          // Filter out hardcoded model specs from context while preserving legitimate config
          const filteredContext = filterContextCorruption(nodeData.node.context || {});
          
          const agentResult = {
            ...node,
            type: 'agentnode',
            data: {
              label: nodeData.label || 'AI Agent',
              agentName: nodeData.node.agent_name || '',
              // Load whatever custom_prompt is saved in the database
              customPrompt: nodeData.node.custom_prompt || '',
              tools: nodeData.node.tools || [],
              timeout: nodeData.node.timeout || 45,
              temperature: nodeData.node.temperature || 0.7,
              model: nodeData.node.model || 'qwen3:30b-a3b',
              max_tokens: nodeData.node.max_tokens || 4000,
              query: nodeData.node.query || '',
              // Filter corruption from context but preserve legitimate fields
              context: cleanAgentContext(filteredContext),
              // Enhanced state management fields
              stateEnabled: nodeData.node.state_enabled || false,
              stateOperation: nodeData.node.state_operation || 'passthrough',
              outputFormat: nodeData.node.output_format || 'text',
              chainKey: nodeData.node.chain_key || ''
            }
          };
          return agentResult;

        case 'InputNode':
          return {
            ...node,
            type: 'inputnode',
            data: {
              label: 'Workflow Input',
              inputSchema: nodeData.node.input_schema || {},
              defaultValues: nodeData.node.default_values || {}
            }
          };

        case 'OutputNode':
          const loadedOutputNode = {
            ...node,
            type: 'outputnode',
            data: {
              label: 'Workflow Output',
              // Use camelCase for UI compatibility
              outputFormat: nodeData.node.output_format || 'text',
              includeMetadata: nodeData.node.include_metadata || false,
              includeToolCalls: nodeData.node.include_tool_calls || false,
              autoDisplay: nodeData.node.auto_display || false,
              autoSave: nodeData.node.auto_save || false,
              // Also preserve snake_case for component compatibility
              output_format: nodeData.node.output_format || 'text',
              include_metadata: nodeData.node.include_metadata || false,
              include_tool_calls: nodeData.node.include_tool_calls || false,
              auto_display: nodeData.node.auto_display || false,
              auto_save: nodeData.node.auto_save || false
            }
          };
          return loadedOutputNode;

        case 'ConditionNode':
          return {
            ...node,
            type: 'conditionnode',
            data: {
              label: 'Condition',
              conditionType: nodeData.node.condition_type || 'simple',
              operator: nodeData.node.operator || 'equals',
              compareValue: nodeData.node.compare_value || '',
              aiCriteria: nodeData.node.ai_criteria || ''
            }
          };

        case 'ParallelNode':
          return {
            ...node,
            type: 'parallelnode',
            data: {
              label: nodeData.node.label || 'Parallel Execution',
              // Store backend properties directly
              max_parallel: nodeData.node.max_parallel || 3,
              wait_for_all: nodeData.node.wait_for_all ?? true,
              combine_strategy: nodeData.node.combine_strategy || 'merge'
            }
          };

        case 'TransformNode':
          return {
            ...node,
            type: 'transformnode',
            data: {
              label: 'Transform',
              transformType: nodeData.node.transform_type || 'jsonpath',
              expression: nodeData.node.expression || '$',
              errorHandling: nodeData.node.error_handling || 'continue',
              defaultValue: nodeData.node.default_value || null,
              inputValidation: nodeData.node.input_validation || {},
              outputValidation: nodeData.node.output_validation || {},
              cacheResults: nodeData.node.cache_results || false,
              testData: nodeData.node.test_data || {example: 'data'}
            }
          };

        case 'StateNode':
          return {
            ...node,
            type: 'statenode',
            data: {
              label: 'State Manager',
              stateOperation: nodeData.node.state_operation || 'merge',
              stateKeys: nodeData.node.state_keys || [],
              stateValues: nodeData.node.state_values || {},
              persistence: nodeData.node.persistence || true,
              stateSchema: nodeData.node.state_schema || {},
              checkpointName: nodeData.node.checkpoint_name || ''
            }
          };

        case 'RouterNode':
          return {
            ...node,
            type: 'routernode',
            data: {
              label: nodeData.node.label || 'Router Node',
              routing_mode: nodeData.node.routing_mode || 'multi-select',
              match_type: nodeData.node.match_type || 'exact',
              routes: nodeData.node.routes || [],
              fallback_route: nodeData.node.fallback_route || '',
              case_sensitive: nodeData.node.case_sensitive || false,
              output_field: nodeData.node.output_field || ''
            }
          };

        case 'CacheNode':
          console.log('CacheNode data:', {
            originalNode: node,
            nodeData: nodeData.node
          });
          const transformedCacheNode = {
            ...node,
            type: 'cachenode',
            data: {
              label: nodeData.node.label || 'Cache',
              cacheKey: nodeData.node.cache_key || '',
              cacheKeyPattern: nodeData.node.cache_key_pattern || 'auto',
              ttl: nodeData.node.ttl || 3600,
              cachePolicy: nodeData.node.cache_policy || 'always',
              invalidateOn: nodeData.node.invalidate_on || ['input_change'],
              cacheCondition: nodeData.node.cache_condition || '',
              enableWarming: nodeData.node.enable_warming || false,
              maxCacheSize: nodeData.node.max_cache_size || 10,
              cacheNamespace: nodeData.node.cache_namespace || 'default',
              showStatistics: nodeData.node.show_statistics ?? true
            }
          };
          return transformedCacheNode;
          
        case 'TriggerNode':
          console.log('TriggerNode data:', {
            originalNode: node,
            nodeData: nodeData.node
          });
          const transformedTriggerNode = {
            ...node,
            type: 'triggernode',
            data: {
              label: nodeData.node.label || 'External Trigger',
              trigger_name: nodeData.node.trigger_name || '',
              http_methods: nodeData.node.http_methods || ['POST'],
              authentication_type: nodeData.node.authentication_type || 'api_key',
              auth_header_name: nodeData.node.auth_header_name || 'X-API-Key',
              auth_token: nodeData.node.auth_token || '',
              basic_auth_username: nodeData.node.basic_auth_username || '',
              basic_auth_password: nodeData.node.basic_auth_password || '',
              rate_limit: nodeData.node.rate_limit || 60,
              timeout: nodeData.node.timeout || 300,
              response_format: nodeData.node.response_format || 'workflow_output',
              custom_response_template: nodeData.node.custom_response_template || '',
              cors_enabled: nodeData.node.cors_enabled !== false,
              cors_origins: nodeData.node.cors_origins || '*',
              log_requests: nodeData.node.log_requests !== false
            }
          };
          return transformedTriggerNode;
          
        case 'APINode':
          // Extract data from nested node structure
          const apiNodeData = nodeData.node || nodeData;
          return {
            ...node,
            type: 'apinode',
            data: {
              label: apiNodeData.label || nodeData.label || 'APINode',
              base_url: apiNodeData.base_url || '',
              endpoint_path: apiNodeData.endpoint_path || '',
              http_method: apiNodeData.http_method || 'GET',
              authentication_type: apiNodeData.authentication_type || 'none',
              auth_header_name: apiNodeData.auth_header_name || 'X-API-Key',
              auth_token: apiNodeData.auth_token || '',
              basic_auth_username: apiNodeData.basic_auth_username || '',
              basic_auth_password: apiNodeData.basic_auth_password || '',
              request_schema: apiNodeData.request_schema || {},
              response_schema: apiNodeData.response_schema || {},
              timeout: apiNodeData.timeout || 30,
              retry_count: apiNodeData.retry_count || 3,
              rate_limit: apiNodeData.rate_limit || 60,
              custom_headers: apiNodeData.custom_headers || {},
              response_transformation: apiNodeData.response_transformation || '',
              error_handling: apiNodeData.error_handling || 'throw',
              enable_mcp_tool: apiNodeData.enable_mcp_tool !== false,
              tool_description: apiNodeData.tool_description || '',
              // Preserve execution data if present
              executionData: nodeData.executionData
            }
          };
          
        default:
          // Keep original format for unknown types
          return node;
      }
    });
  };

  // Calculate execution sequence based on node dependencies and edges
  const calculateExecutionSequence = (nodes: Node[], edges: Edge[]): string[] => {
    const dependencyMap = new Map<string, string[]>();
    const indegree = new Map<string, number>();
    
    // Initialize dependency tracking
    nodes.forEach(node => {
      dependencyMap.set(node.id, []);
      indegree.set(node.id, 0);
    });
    
    // Build dependency graph from edges
    edges.forEach(edge => {
      const dependencies = dependencyMap.get(edge.target) || [];
      dependencies.push(edge.source);
      dependencyMap.set(edge.target, dependencies);
      indegree.set(edge.target, (indegree.get(edge.target) || 0) + 1);
    });
    
    // Topological sort for execution sequence
    const executionOrder: string[] = [];
    const queue: string[] = [];
    
    // Find nodes with no dependencies (start nodes)
    indegree.forEach((degree, nodeId) => {
      if (degree === 0) queue.push(nodeId);
    });
    
    while (queue.length > 0) {
      const currentNode = queue.shift()!;
      executionOrder.push(currentNode);
      
      // Find nodes that depend on current node
      edges.forEach(edge => {
        if (edge.source === currentNode) {
          const targetIndegree = indegree.get(edge.target)! - 1;
          indegree.set(edge.target, targetIndegree);
          if (targetIndegree === 0) {
            queue.push(edge.target);
          }
        }
      });
    }
    
    return executionOrder;
  };
  
  // Map node relationships with handle information
  const mapNodeRelationships = (edges: Edge[]) => {
    const relationships: Record<string, any> = {};
    
    edges.forEach(edge => {
      const sourceNode = edge.source;
      const targetNode = edge.target;
      
      if (!relationships[sourceNode]) {
        relationships[sourceNode] = { outputs: [], inputs: [] };
      }
      if (!relationships[targetNode]) {
        relationships[targetNode] = { outputs: [], inputs: [] };
      }
      
      relationships[sourceNode].outputs.push({
        target_node: targetNode,
        source_handle: edge.sourceHandle,
        target_handle: edge.targetHandle,
        edge_id: edge.id
      });
      
      relationships[targetNode].inputs.push({
        source_node: sourceNode,
        source_handle: edge.sourceHandle,
        target_handle: edge.targetHandle,
        edge_id: edge.id
      });
    });
    
    return relationships;
  };
  
  // Enhance edges with 4-way handle information
  const enhanceEdgesWithHandleInfo = (edges: Edge[]) => {
    return edges.map(edge => ({
      ...edge,
      connectivity_info: {
        source_handle_id: edge.sourceHandle,
        target_handle_id: edge.targetHandle,
        source_position: edge.sourceHandle?.includes('top') ? 'top' :
                        edge.sourceHandle?.includes('right') ? 'right' :
                        edge.sourceHandle?.includes('bottom') ? 'bottom' : 'left',
        target_position: edge.targetHandle?.includes('top') ? 'top' :
                        edge.targetHandle?.includes('right') ? 'right' :
                        edge.targetHandle?.includes('bottom') ? 'bottom' : 'left',
        direction_vector: {
          from: edge.sourceHandle,
          to: edge.targetHandle
        }
      }
    }));
  };

  // Transform frontend nodes to backend format for AI-focused workflow
  const transformNodesToBackendFormat = (frontendNodes: Node[]) => {
    return frontendNodes.map((node, index) => {
      const nodeData = node.data;
      
      // Transform based on node type
      try {
        switch (node.type) {
        case 'llm':
          return {
            ...node,
            data: {
              type: 'JarvisLLMNode',
              node: {
                model: nodeData.model || 'qwen3:30b-a3b',
                prompt: nodeData.prompt || '',
                parameters: nodeData.parameters || {},
                selected_tools: nodeData.selectedTools || [],
                enable_tools: nodeData.enableTools ?? true
              }
            }
          };
          
        case 'agent':
          console.log('agent data:', {
            stateEnabled: nodeData.stateEnabled,
            stateOperation: nodeData.stateOperation,
            outputFormat: nodeData.outputFormat,
            chainKey: nodeData.chainKey,
            label: nodeData.label
          });
          return {
            ...node,
            data: {
              type: 'JarvisAgentNode',
              label: nodeData.label || '', // Preserve the label field
              node: {
                agent_name: nodeData.agentName || '',
                query: nodeData.query || '',
                context: nodeData.context || {},
                selected_tools: nodeData.selectedTools || [],
                enable_tools: nodeData.enableTools ?? true,
                // Enhanced state management fields
                state_enabled: nodeData.stateEnabled || false,
                state_operation: nodeData.stateOperation || 'passthrough',
                output_format: nodeData.outputFormat || 'text',
                chain_key: nodeData.chainKey || ''
              }
            }
          };
          
        case 'aiDecision':
          return {
            ...node,
            data: {
              type: 'JarvisAIDecisionNode',
              node: {
                criteria: nodeData.criteria || '',
                decision_prompt: nodeData.decisionPrompt || '',
                confidence_threshold: nodeData.confidenceThreshold || 0.8
              }
            }
          };
          
        case 'documentProcessor':
          return {
            ...node,
            data: {
              type: 'JarvisDocumentProcessorNode',
              node: {
                document_type: nodeData.documentType || 'pdf',
                analysis_task: nodeData.analysisTask || 'extract_text',
                extraction_params: nodeData.extractionParams || {}
              }
            }
          };
          
        case 'imageProcessor':
          return {
            ...node,
            data: {
              type: 'JarvisImageProcessorNode',
              node: {
                analysis_type: nodeData.analysisType || 'ocr',
                analysis_task: nodeData.analysisTask || 'extract_text',
                image_params: nodeData.imageParams || {}
              }
            }
          };
          
        case 'audioProcessor':
          return {
            ...node,
            data: {
              type: 'JarvisAudioProcessorNode',
              node: {
                analysis_type: nodeData.analysisType || 'transcription',
                analysis_task: nodeData.analysisTask || 'transcribe',
                audio_params: nodeData.audioParams || {}
              }
            }
          };
          
        case 'multiModalFusion':
          return {
            ...node,
            data: {
              type: 'JarvisMultiModalFusionNode',
              node: {
                fusion_task: nodeData.fusionTask || 'analyze_all',
                fusion_strategy: nodeData.fusionStrategy || 'combined_analysis',
                output_format: nodeData.outputFormat || 'unified_summary'
              }
            }
          };
          
        case 'contextMemory':
          return {
            ...node,
            data: {
              type: 'JarvisContextMemoryNode',
              node: {
                memory_key: nodeData.memoryKey || 'default',
                operation: nodeData.operation || 'store',
                retention_policy: nodeData.retentionPolicy || 'workflow_scoped'
              }
            }
          };
          
        case 'condition':
          return {
            ...node,
            data: {
              type: 'JarvisConditionNode',
              node: {
                operator: nodeData.operator || 'equals',
                left_operand: nodeData.leftOperand || '',
                right_operand: nodeData.rightOperand || ''
              }
            }
          };
          
        case 'loop':
          return {
            ...node,
            data: {
              type: 'JarvisLoopNode',
              node: {
                array_input: nodeData.arrayInput || '',
                item_variable: nodeData.itemVariable || 'item',
                index_variable: nodeData.indexVariable || 'index'
              }
            }
          };

        case 'dataMapper':
          return {
            ...node,
            data: {
              type: 'JarvisDataMapperNode',
              node: {
                mapping_config: nodeData.mappingConfig || {},
                input_schema: nodeData.inputSchema || {},
                output_schema: nodeData.outputSchema || {}
              }
            }
          };

        case 'aggregatornode':
          return {
            ...node,
            data: {
              type: 'JarvisAggregatorNode',
              node: {
                aggregation_strategy: nodeData.aggregationStrategy || 'semantic_merge',
                confidence_threshold: nodeData.confidenceThreshold || 0.7,
                max_inputs: nodeData.maxInputs || 5,
                deduplication_enabled: nodeData.deduplicationEnabled || true,
                similarity_threshold: nodeData.similarityThreshold || 0.8,
                output_format: nodeData.outputFormat || 'comprehensive',
                include_source_attribution: nodeData.includeSourceAttribution || false,
                conflict_resolution: nodeData.conflictResolution || 'weighted_vote',
                semantic_analysis: nodeData.semanticAnalysis || true,
                preserve_structure: nodeData.preserveStructure || false,
                fallback_strategy: nodeData.fallbackStrategy || 'simple_concatenate'
              }
            }
          };

        case 'variable':
          return {
            ...node,
            data: {
              type: 'JarvisVariableNode',
              node: {
                operation: nodeData.operation || 'get',
                variable_name: nodeData.variableName || '',
                variable_value: nodeData.variableValue || '',
                scope: nodeData.scope || 'workflow'
              }
            }
          };
          
        case 'start':
          return {
            ...node,
            data: {
              type: 'JarvisStartNode',
              node: {
                label: nodeData.label || 'Start'
              }
            }
          };
          
        case 'end':
          return {
            ...node,
            data: {
              type: 'JarvisEndNode',
              node: {
                label: nodeData.label || 'End'
              }
            }
          };
          
        // Agent-based workflow node types
        case 'agentnode':
          console.log('agentnode data:', {
            stateEnabled: nodeData.stateEnabled,
            stateOperation: nodeData.stateOperation,
            outputFormat: nodeData.outputFormat,
            chainKey: nodeData.chainKey,
            label: nodeData.label
          });
          const agentSaveData = {
            ...node,
            data: {
              type: 'AgentNode',
              label: nodeData.label || '', // ADD THE FUCKING LABEL!
              node: {
                agent_name: nodeData.agentName || '',
                custom_prompt: nodeData.customPrompt || '',
                tools: nodeData.tools || [],
                timeout: nodeData.timeout || 45,
                temperature: nodeData.temperature || 0.7,
                model: nodeData.model || 'qwen3:30b-a3b',
                max_tokens: nodeData.max_tokens || 4000,
                query: nodeData.query || '',
                context: cleanAgentContext(nodeData.context || {}),
                // Enhanced state management fields
                state_enabled: nodeData.stateEnabled || false,
                state_operation: nodeData.stateOperation || 'passthrough',
                output_format: nodeData.outputFormat || 'text',
                chain_key: nodeData.chainKey || ''
              }
            }
          };
          return agentSaveData;
          
        case 'inputnode':
          return {
            ...node,
            data: {
              type: 'InputNode',
              node: {
                input_schema: nodeData.inputSchema || {},
                default_values: nodeData.defaultValues || {}
              }
            }
          };
          
        case 'outputnode':
          const outputNodeResult = {
            ...node,
            data: {
              type: 'OutputNode',
              node: {
                output_format: nodeData.output_format || nodeData.outputFormat || 'text',
                include_metadata: nodeData.include_metadata || nodeData.includeMetadata || false,
                include_tool_calls: nodeData.include_tool_calls || nodeData.includeToolCalls || false,
                auto_display: nodeData.auto_display || nodeData.autoDisplay || false,
                auto_save: nodeData.auto_save || nodeData.autoSave || false
              }
            }
          };
          return outputNodeResult;
          
        case 'conditionnode':
          return {
            ...node,
            data: {
              type: 'ConditionNode',
              node: {
                label: nodeData.label || 'Condition',
                condition_type: nodeData.conditionType || 'simple',
                operator: nodeData.operator || 'equals',
                left_operand: nodeData.leftOperand || '',
                right_operand: nodeData.rightOperand || '',
                compare_value: nodeData.compareValue || nodeData.rightOperand || '',
                ai_criteria: nodeData.aiCriteria || '',
                case_sensitive: nodeData.caseSensitive !== false,
                data_type: nodeData.dataType || 'string'
              }
            }
          };
          
        case 'parallelnode':
          return {
            ...node,
            data: {
              type: 'ParallelNode',
              node: {
                max_parallel: nodeData.max_parallel || nodeData.maxParallel || 3,
                wait_for_all: nodeData.wait_for_all ?? nodeData.waitForAll ?? true,
                combine_strategy: nodeData.combine_strategy || nodeData.combineStrategy || 'merge'
              }
            }
          };
          
        case 'transformnode':
          return {
            ...node,
            data: {
              type: 'TransformNode',
              node: {
                transform_type: nodeData.transformType || 'jsonpath',
                expression: nodeData.expression || '$',
                error_handling: nodeData.errorHandling || 'continue',
                default_value: nodeData.defaultValue || null,
                input_validation: nodeData.inputValidation || {},
                output_validation: nodeData.outputValidation || {},
                cache_results: nodeData.cacheResults || false,
                test_data: nodeData.testData || {example: 'data'}
              }
            }
          };
          
        case 'statenode':
          return {
            ...node,
            data: {
              type: 'StateNode',
              node: {
                state_operation: nodeData.stateOperation || 'merge',
                persistence: nodeData.persistence || true,
                state_schema: nodeData.stateSchema || {},
                checkpoint_name: nodeData.checkpointName || ''
              }
            }
          };

        case 'routernode':
          return {
            ...node,
            data: {
              type: 'RouterNode',
              node: {
                label: nodeData.label || 'Router Node',
                routing_mode: nodeData.routing_mode || 'multi-select',
                match_type: nodeData.match_type || 'exact',
                routes: nodeData.routes || [],
                fallback_route: nodeData.fallback_route || '',
                case_sensitive: nodeData.case_sensitive || false,
                output_field: nodeData.output_field || ''
              }
            }
          };

        case 'cachenode':
          console.log('cachenode data:', {
            originalNode: node,
            nodeData: nodeData
          });
          const backendCacheNode = {
            ...node,
            data: {
              type: 'CacheNode',
              node: {
                label: nodeData.label || 'Cache',
                cache_key: nodeData.cacheKey || '',
                cache_key_pattern: nodeData.cacheKeyPattern || 'auto',
                ttl: nodeData.ttl || 3600,
                cache_policy: nodeData.cachePolicy || 'always',
                invalidate_on: nodeData.invalidateOn || ['input_change'],
                cache_condition: nodeData.cacheCondition || '',
                enable_warming: nodeData.enableWarming || false,
                max_cache_size: nodeData.maxCacheSize || 10,
                cache_namespace: nodeData.cacheNamespace || 'default',
                show_statistics: nodeData.showStatistics ?? true
              }
            }
          };
          return backendCacheNode;

        case 'triggernode':
          const backendTriggerNode = {
            ...node,
            data: {
              type: 'TriggerNode',
              node: {
                label: nodeData.label || 'External Trigger',
                trigger_name: nodeData.trigger_name || '',
                http_methods: nodeData.http_methods || ['POST'],
                authentication_type: nodeData.authentication_type || 'api_key',
                auth_header_name: nodeData.auth_header_name || 'X-API-Key',
                auth_token: nodeData.auth_token || '',
                basic_auth_username: nodeData.basic_auth_username || '',
                basic_auth_password: nodeData.basic_auth_password || '',
                rate_limit: nodeData.rate_limit || 60,
                timeout: nodeData.timeout || 300,
                response_format: nodeData.response_format || 'workflow_output',
                custom_response_template: nodeData.custom_response_template || '',
                cors_enabled: nodeData.cors_enabled !== false,
                cors_origins: nodeData.cors_origins || '*',
                log_requests: nodeData.log_requests !== false
              }
            }
          };
          return backendTriggerNode;

        case 'apinode':
        case 'api':
          // Transform APINode to backend format (match the nested structure like other nodes)
          const backendApiNode = {
            id: node.id,
            type: 'APINode',
            position: node.position,
            data: {
              type: 'APINode',
              label: node.data.label || 'APINode',
              node: {
                label: node.data.label || 'APINode',
                base_url: node.data.base_url || '',
                endpoint_path: node.data.endpoint_path || '',
                http_method: node.data.http_method || 'GET',
                authentication_type: node.data.authentication_type || 'none',
                auth_header_name: node.data.auth_header_name || 'X-API-Key',
                auth_token: node.data.auth_token || '',
                basic_auth_username: node.data.basic_auth_username || '',
                basic_auth_password: node.data.basic_auth_password || '',
                request_schema: node.data.request_schema || {},
                response_schema: node.data.response_schema || {},
                timeout: node.data.timeout || 30,
                retry_count: node.data.retry_count || 3,
                rate_limit: node.data.rate_limit || 60,
                custom_headers: node.data.custom_headers || {},
                response_transformation: node.data.response_transformation || '',
                error_handling: node.data.error_handling || 'throw',
                enable_mcp_tool: node.data.enable_mcp_tool !== false,
                tool_description: node.data.tool_description || ''
              }
            }
          };
          return backendApiNode;

        default:
          // Keep original format for unknown types
          return ensureNodePosition(node, index);
        }
      } catch (error) {
        throw new Error(`Failed to transform node ${node.id} (type: ${node.type}): ${error.message}`);
      }
    });
  };

  // Internal save function (actual save logic)
  const performSave = useCallback(async () => {
    // Prevent concurrent saves
    if (isSaving || isAutoSaving) {
      return;
    }

    // Validate workflow name before saving
    if (!currentWorkflowName || currentWorkflowName.trim() === '') {
      setExecutionError('Please provide a workflow name before saving');
      return;
    }
    
    setIsSaving(true);
    
    // DEBUG: Check for CacheNode specifically
    const cacheNodes = nodes.filter(node => node.type === 'cachenode');
    const cacheNodeEdges = edges.filter(edge => 
      cacheNodes.some(n => n.id === edge.source || n.id === edge.target)
    );
    
    // Transform nodes to backend format
    let transformedNodes = [];
    try {
      // Add safety check for empty nodes
      if (!nodes || nodes.length === 0) {
        
        // Check if this is a new workflow that should have default nodes
        if (!workflowId && workflowType) {
          const startType = workflowType === 'agent_based' ? 'inputnode' : 'start';
          const endType = workflowType === 'agent_based' ? 'outputnode' : 'end';
          
          const startTemplate = getCurrentNodeTemplate(startType);
          const endTemplate = getCurrentNodeTemplate(endType);
          const startHandles = getCurrentNodeHandles(startType);
          const endHandles = getCurrentNodeHandles(endType);
          
          if (startTemplate && endTemplate) {
            const initialNodes: Node[] = [
              {
                id: 'start-1',
                type: startType,
                position: { x: 100, y: 50 },
                data: { 
                  label: startTemplate.label || 'Start',
                  inputHandles: startHandles.inputs,
                  outputHandles: startHandles.outputs,
                  nodeTemplate: startTemplate
                },
                draggable: true,
              },
              {
                id: 'end-1',
                type: endType,
                position: { x: 100, y: 300 },
                data: { 
                  label: endTemplate.label || 'End',
                  inputHandles: endHandles.inputs,
                  outputHandles: endHandles.outputs,
                  nodeTemplate: endTemplate
                },
                draggable: true,
              },
            ];
            
            setNodes(initialNodes);
            setNodeCounter(2);
            setIsSaving(false);
            // Try to save again after a short delay
            setTimeout(() => {
              performSave();
            }, 100);
            return;
          }
        }
        
        alert('Cannot save workflow: No nodes found. Please add nodes to your workflow.');
        setIsSaving(false);
        return;
      }
      
      transformedNodes = transformNodesToBackendFormat(nodes);
      
      // Validate transformed nodes
      if (!transformedNodes || transformedNodes.length === 0) {
        setExecutionError('Failed to transform workflow nodes. Please check the console for errors.');
        setIsSaving(false);
        return;
      }
      
      
      // DEBUG: Check transformed CacheNodes
      const transformedCacheNodes = transformedNodes.filter(node => 
        node.data?.type === 'CacheNode' || node.type === 'cachenode'
      );
    
      // Enhanced workflow data with execution sequence and relationship mapping
      const executionSequence = calculateExecutionSequence(nodes, edges);
      const nodeRelationships = mapNodeRelationships(edges);
      
      // DEBUG: Check edges before enhancement
      
      const enhancedEdges = enhanceEdgesWithHandleInfo(edges);
      
      // DEBUG: Check edges after enhancement
      const enhancedCacheNodeEdges = enhancedEdges.filter(edge => 
        cacheNodes.some(n => n.id === edge.source || n.id === edge.target)
      );
      
      const workflowData = {
        name: currentWorkflowName || 'Untitled Workflow',
        description: `Enhanced automation workflow with ${nodes.length} nodes and ${enhancedEdges.length} connections`,
        langflow_config: {
          nodes: transformedNodes,
          edges: enhancedEdges,
          execution_sequence: executionSequence,
          node_relationships: nodeRelationships,
          workflow_type: workflowType || 'agent_based',
          version: "2.0",
          metadata: {
            created_at: new Date().toISOString(),
            total_nodes: transformedNodes.length,
            total_edges: enhancedEdges.length,
            node_types: [...new Set(transformedNodes.map(n => n.data?.type || n.type))],
            has_cache_nodes: transformedCacheNodes.length > 0,
            cache_node_count: transformedCacheNodes.length
          }
        },
        is_active: true
      };
      
      
      // CRITICAL: Final validation before save
      if (!workflowData.langflow_config.nodes || workflowData.langflow_config.nodes.length === 0) {
        alert('Cannot save workflow: No nodes in final data. This is a critical error.');
        setIsSaving(false);
        return;
      }
      
      // DEBUG: Final check of CacheNode edges in workflow data
      const finalCacheNodeEdges = workflowData.langflow_config.edges.filter(edge => 
        cacheNodes.some(n => n.id === edge.source || n.id === edge.target)
      );

      // Always handle save directly to avoid state loss
      {
        // Make API calls directly
        let response;
        if (workflowId) {
          // Update existing workflow
          response = await fetch(`http://127.0.0.1:8000/api/v1/automation/workflows/${workflowId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(workflowData),
          });
        } else {
          // Create new workflow
          response = await fetch('http://127.0.0.1:8000/api/v1/automation/workflows', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(workflowData),
          });
        }


        if (response.ok) {
          const result = await response.json();
          
          // CRITICAL: Validate the saved workflow data
          if (result.langflow_config) {
            const savedNodes = result.langflow_config.nodes || [];
            if (savedNodes.length === 0) {
              setExecutionError('WARNING: Workflow saved but nodes may be missing. Please refresh and check your workflow.');
              // Don't set success to prevent false confidence
              return;
            }
          }
          
          // If creating a new workflow, we get the new ID back
          if (!workflowId && result.id) {
            // Update the URL or state to reflect the new workflow ID for future saves
            // Don't update workflowId state here to avoid re-renders that clear the editor
          }
          
          // Backend cache issue detected - the save is successful but cache isn't updating
          // We need to restart the backend or fix the cache invalidation on the backend side
          
          // Don't close the editor - let user continue editing
          setSaveSuccess(true);
          // Reset unsaved changes flag after successful save
          setHasUnsavedChanges(false);
        } else {
          const errorText = await response.text();
          setExecutionError(`Failed to save workflow: ${response.status} - ${errorText}`);
        }
      }
    } catch (error) {
      if (error.message && error.message.includes('transform')) {
        setExecutionError(`Failed to transform nodes: ${error.message}`);
      } else {
        setExecutionError(`Failed to save workflow: ${error.message}`);
      }
    } finally {
      setIsSaving(false);
    }
  }, [nodes, edges, workflowId, currentWorkflowName, workflowType, isSaving, isAutoSaving]);

  // Debounced save function (prevents rapid clicking)
  const handleSave = useCallback(() => {
    // Clear any existing timeout
    if (saveTimeoutRef) {
      clearTimeout(saveTimeoutRef);
    }
    
    // Set new timeout for debouncing (500ms delay)
    const timeoutId = setTimeout(() => {
      performSave();
      setSaveTimeoutRef(null);
    }, 500);
    
    setSaveTimeoutRef(timeoutId);
  }, [performSave, saveTimeoutRef]);
  
  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (saveTimeoutRef) {
        clearTimeout(saveTimeoutRef);
      }
    };
  }, [saveTimeoutRef]);

  // Execute workflow with streaming support
  const handleExecute = async () => {
    if (!workflowId) {
      setExecutionError('Please save the workflow first before executing');
      return;
    }

    setIsExecuting(true);
    setShowExecutionPanel(true);
    setExecutionLogs([]);
    setExecutionError(null);
    
    // Reset all node execution data
    resetAllNodeExecutionData();

    try {
      // All workflows now use the streaming endpoint
      const endpoint = `http://127.0.0.1:8000/api/v1/automation/workflows/${workflowId}/execute/stream`;

      // DEBUG LOGGING for workflow execution
      console.log(`[WORKFLOW DEBUG] === EXECUTING WORKFLOW ID ${workflowId} ===`);
      console.log(`[WORKFLOW DEBUG] Workflow type: ${workflowType}`);
      console.log(`[WORKFLOW DEBUG] Endpoint: ${endpoint}`);
      console.log(`[WORKFLOW DEBUG] Execution message: ${executionMessage}`);

      const requestBody = {
        input_data: {},
        execution_mode: 'stream', // All workflows now use streaming
        ...(executionMessage && { message: executionMessage })
      };

      console.log(`[WORKFLOW DEBUG] Request body:`, requestBody);

      // All workflows now use streaming execution
      console.log(`[WORKFLOW DEBUG] Making fetch request to: ${endpoint}`);
      console.log(`[WORKFLOW DEBUG] Method: POST`);
      console.log(`[WORKFLOW DEBUG] Headers: Content-Type: application/json`);
      
      const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
        });

        console.log(`[WORKFLOW DEBUG] Response status: ${response.status}`);

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error('Failed to get response reader');
        }

        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.trim() && line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                const logEntry = {
                  timestamp: new Date().toISOString(),
                  ...data
                };
                setExecutionLogs(prev => [...prev, logEntry]);
                
                // Update corresponding nodes with execution data
                processExecutionLogForNodes(logEntry);
              } catch (e) {
              }
            }
          }
        }
      }
    } catch (error) {
      setExecutionError(error instanceof Error ? error.message : 'Unknown error occurred');
    } finally {
      setIsExecuting(false);
    }
  };

  // Auto-save before execution wrapper
  const autoSaveBeforeExecution = async (): Promise<boolean> => {
    // If no unsaved changes and workflow is already saved, skip auto-save
    if (!hasUnsavedChanges && workflowId) {
      return true;
    }
    
    // If it's a new workflow (no workflowId), allow execution without saving
    // but use a temporary name for the workflow
    if (!workflowId) {
      // For execution, we can proceed without saving to database
      // The workflow will run with current nodes/edges in memory
      return true;
    }
    
    setIsAutoSaving(true);
    
    try {
      // For existing workflows, auto-save before execution
      // Only validate name for manual saves, not auto-saves before execution
      if (!currentWorkflowName || currentWorkflowName.trim() === '') {
        // For execution auto-save, use a temporary name if none provided
        const tempName = `Workflow_${Date.now()}`;
        
        // Temporarily set the workflow name for auto-save
        const originalName = currentWorkflowName;
        setCurrentWorkflowName(tempName);
        
        try {
          await performSave();
          setHasUnsavedChanges(false);
          return true;
        } finally {
          // Restore original name after save
          setCurrentWorkflowName(originalName);
        }
      } else {
        // Normal save with existing name
        await performSave();
        setHasUnsavedChanges(false);
        return true;
      }
    } catch (error) {
      setExecutionError('Failed to auto-save workflow. Please save manually before executing.');
      return false;
    } finally {
      setIsAutoSaving(false);
    }
  };

  // Quick execute function for Play button without message
  const handleQuickExecute = async () => {
    // Auto-save before execution
    const saveSuccessful = await autoSaveBeforeExecution();
    if (!saveSuccessful) {
      return;
    }
    
    setExecutionMessage('');
    handleExecute();
  };

  // Helper function to get the first agent node in execution sequence
  const getFirstAgentNode = useCallback(() => {
    // Filter agent nodes (both 'agent' and 'agentnode' types)
    const agentNodes = nodes.filter(node => 
      node.type === 'agent' || node.type === 'agentnode'
    );
    
    if (agentNodes.length === 0) return null;
    
    // Simple approach: find the agent node with no incoming edges from other agents
    const agentNodeIds = new Set(agentNodes.map(n => n.id));
    const agentNodesWithIncomingEdges = new Set();
    
    edges.forEach(edge => {
      if (agentNodeIds.has(edge.target) && agentNodeIds.has(edge.source)) {
        agentNodesWithIncomingEdges.add(edge.target);
      }
    });
    
    // Find the first agent (no incoming edges from other agents)
    const firstAgent = agentNodes.find(node => !agentNodesWithIncomingEdges.has(node.id));
    return firstAgent || agentNodes[0]; // Fallback to first agent if logic fails
  }, [nodes, edges]);

  // Execute with message function
  const handleExecuteWithMessage = async () => {
    // Auto-save before execution
    const saveSuccessful = await autoSaveBeforeExecution();
    if (!saveSuccessful) {
      return;
    }
    
    // No need to append message to first agent's query - the backend now properly
    // passes the original user message to ALL agents through workflow state
    // This prevents duplicate message appending on re-runs
    
    // Keep the execution message for re-runs and execute
    handleExecute();
  };

  // Simplified group creation function
  const handleCreateGroupFromSelected = useCallback(() => {
    const selectedNodes = nodes.filter(node => selectedNodeIds.includes(node.id));
    
    if (selectedNodes.length === 0) {
      setExecutionError('Please select at least one node to group');
      return;
    }

    // Calculate group box position and size based on selected nodes
    // We need to get actual node dimensions, defaulting to reasonable sizes if not available
    const nodeRects = selectedNodes.map(node => {
      // Get actual node dimensions or use defaults based on node type
      let width = node.width || 400;
      let height = node.height || 300;
      
      // Adjust defaults based on node type
      if (node.type === 'start' || node.type === 'end') {
        width = node.width || 120;
        height = node.height || 71;
      } else if (node.type === 'agentnode' || node.type === 'agent') {
        width = node.width || 400;
        height = node.height || 391;
      } else if (node.type === 'routernode') {
        width = node.width || 600;
        height = node.height || 633;
      } else if (node.type === 'outputnode') {
        width = node.width || 450;
        height = node.height || 170;
      } else if (node.type === 'cachenode') {
        width = node.width || 450;
        height = node.height || 272;
      }
      
      return {
        x: node.position.x,
        y: node.position.y,
        width: width,
        height: height
      };
    });

    // Calculate bounding box with padding
    const padding = 40;
    const minX = Math.min(...nodeRects.map(rect => rect.x)) - padding;
    const minY = Math.min(...nodeRects.map(rect => rect.y)) - padding - 20; // Extra top padding for title
    const maxX = Math.max(...nodeRects.map(rect => rect.x + rect.width)) + padding;
    const maxY = Math.max(...nodeRects.map(rect => rect.y + rect.height)) + padding;

    // Create group box node
    const groupBoxId = `groupbox-${Date.now()}`;
    const groupBox: Node = {
      id: groupBoxId,
      type: 'groupBox',
      position: { x: minX, y: minY },
      data: {
        title: `Group ${selectedNodes.length} nodes`,
        label: selectedNodes.map(n => n.data.label || n.type).join(', '),
        borderColor: '#9c27b0',
        backgroundColor: 'rgba(156, 39, 176, 0.05)',
        opacity: 1
      },
      style: {
        width: maxX - minX,
        height: maxY - minY,
        zIndex: -1000 // Ensure it's behind all other nodes
      },
      width: maxX - minX,
      height: maxY - minY,
      draggable: true,
      selectable: true
    };

    // Add group box to nodes (at the beginning so it renders behind)
    setNodes(prevNodes => [groupBox, ...prevNodes]);
    setNodeCounter(prev => prev + 1);
    
    // Clear selection
    setSelectedNodeIds([]);
    
    // Show success message using existing notification system
    setSaveSuccess(true);
    
  }, [nodes, selectedNodeIds, nodeCounter, setNodes]);

  // Handle node selection changes
  const handleSelectionChange = useCallback((selectedNodes: Node[]) => {
    setSelectedNodeIds(selectedNodes.map(node => node.id));
  }, []);

  // Boundary detection - check if a node is inside a group box
  const isNodeInsideGroup = useCallback((node: Node, groupBox: Node): boolean => {
    if (!groupBox.style?.width || !groupBox.style?.height) return false;
    
    const nodeRect = {
      x: node.position.x,
      y: node.position.y,
      width: node.width || 320,
      height: node.height || 280
    };
    
    const groupRect = {
      x: groupBox.position.x,
      y: groupBox.position.y,
      width: Number(groupBox.style.width),
      height: Number(groupBox.style.height)
    };
    
    // Check if node is completely inside group (with some margin for edge cases)
    const margin = 10;
    return nodeRect.x >= groupRect.x - margin && 
           nodeRect.y >= groupRect.y - margin &&
           nodeRect.x + nodeRect.width <= groupRect.x + groupRect.width + margin &&
           nodeRect.y + nodeRect.height <= groupRect.y + groupRect.height + margin;
  }, []);

  // Get all nodes that are inside a specific group box
  const getNodesInGroup = useCallback((groupBox: Node): Node[] => {
    return nodes.filter(node => 
      node.type !== 'groupBox' && 
      node.id !== groupBox.id && 
      isNodeInsideGroup(node, groupBox)
    );
  }, [nodes, isNodeInsideGroup]);

  // Handle group box movement - move contained nodes with the group
  const handleGroupMovement = useCallback((changes: any[]) => {
    changes.forEach(change => {
      if (change.type === 'position' && change.position) {
        const movedNode = nodes.find(n => n.id === change.id);
        if (movedNode && movedNode.type === 'groupBox') {
          // Calculate movement delta
          const deltaX = change.position.x - movedNode.position.x;
          const deltaY = change.position.y - movedNode.position.y;
          
          // Find all nodes inside this group
          const childNodes = getNodesInGroup(movedNode);
          
          if (childNodes.length > 0 && (Math.abs(deltaX) > 1 || Math.abs(deltaY) > 1)) {
            
            // Move all child nodes by the same delta
            const childChanges = childNodes.map(childNode => ({
              id: childNode.id,
              type: 'position',
              position: {
                x: childNode.position.x + deltaX,
                y: childNode.position.y + deltaY
              }
            }));
            
            // Apply child node movements
            setNodes(prevNodes => 
              prevNodes.map(node => {
                const childChange = childChanges.find(c => c.id === node.id);
                if (childChange) {
                  return {
                    ...node,
                    position: childChange.position
                  };
                }
                return node;
              })
            );
          }
        }
      }
    });
  }, [nodes, getNodesInGroup]);

  // Custom nodes change handler that includes group movement logic and change tracking
  const onNodesChange = useCallback((changes: any[]) => {
    // Check for group movement first
    handleGroupMovement(changes);
    
    // Apply the base ReactFlow node changes
    baseOnNodesChange(changes);
    
    // Track changes for auto-save
    setHasUnsavedChanges(true);
  }, [baseOnNodesChange, handleGroupMovement]);

  const miniMapStyle = {
    height: 120,
  };

  const editorContent = (
    <>
      <Box component={onClose ? DialogTitle : Paper} sx={onClose ? {} : { p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box display="flex" alignItems="center" gap={2}>
            <TextField
              value={currentWorkflowName}
              onChange={(e) => setCurrentWorkflowName(e.target.value)}
              variant="outlined"
              size="small"
              sx={{ minWidth: 250 }}
              placeholder="Enter workflow name..."
            />
            <Chip label="Visual Editor" color="primary" size="small" />
            {isLoadingSchemas ? (
              <Chip label="Loading..." color="default" size="small" icon={<CircularProgress size={16} />} />
            ) : (
              <>
                <Chip 
                  label={workflowId ? 'Editing' : 'Creating New'} 
                  color={workflowId ? 'info' : 'success'} 
                  size="small"
                  variant="outlined"
                />
                <Chip 
                  label={workflowType === 'agent_based' ? 'Agent-Based' : 'Legacy'} 
                  color={workflowType === 'agent_based' ? 'success' : 'warning'} 
                  size="small" 
                />
              </>
            )}
          </Box>
          
          <Box display="flex" alignItems="center" gap={1}>
            {/* Enhanced Layout Controls */}
            <ButtonGroup size="small" variant="outlined">
              <Tooltip title={`Auto Layout (${layoutOptions.algorithm})`}>
                <IconButton 
                  onClick={() => onLayout()} 
                  size="small"
                  disabled={isLayoutAnimating}
                  color={isLayoutAnimating ? 'secondary' : 'default'}
                >
                  {isLayoutAnimating ? (
                    <CircularProgress size={16} />
                  ) : (
                    getLayoutIcon(layoutOptions.algorithm)
                  )}
                </IconButton>
              </Tooltip>
              
              <Tooltip title="Layout Options">
                <IconButton 
                  size="small"
                  onClick={(event) => setLayoutMenuAnchor(event.currentTarget)}
                  disabled={isLayoutAnimating}
                >
                  <DropDownIcon />
                </IconButton>
              </Tooltip>
            </ButtonGroup>
            
            {/* Flow Direction Animation Toggle */}
            <Tooltip title={showFlowAnimation ? "Hide Flow Direction" : "Show Flow Direction"}>
              <IconButton
                size="small"
                onClick={() => setShowFlowAnimation(!showFlowAnimation)}
                color={showFlowAnimation ? 'primary' : 'default'}
                sx={{ 
                  border: showFlowAnimation ? '1px solid' : 'none',
                  borderColor: 'primary.main'
                }}
              >
                <LoopIcon />
              </IconButton>
            </Tooltip>
            
            {/* Group Creation Button */}
            <Tooltip title={selectedNodeIds.length > 0 ? `Create Group (${selectedNodeIds.length} selected)` : "Select nodes to group"}>
              <span>
                <IconButton
                  size="small"
                  onClick={handleCreateGroupFromSelected}
                  disabled={selectedNodeIds.length === 0}
                  color={selectedNodeIds.length > 0 ? 'primary' : 'default'}
                  sx={{ 
                    border: selectedNodeIds.length > 0 ? '1px solid' : 'none',
                    borderColor: 'primary.main'
                  }}
                >
                  <GroupIcon />
                </IconButton>
              </span>
            </Tooltip>
            
            {/* Layout Options Menu */}
            <Menu
              anchorEl={layoutMenuAnchor}
              open={Boolean(layoutMenuAnchor)}
              onClose={() => setLayoutMenuAnchor(null)}
              anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
              transformOrigin={{ vertical: 'top', horizontal: 'left' }}
            >
              <MenuItem 
                onClick={() => {
                  onLayout('hierarchical');
                  setLayoutMenuAnchor(null);
                }}
                selected={layoutOptions.algorithm === 'hierarchical'}
              >
                <HierarchicalIcon sx={{ mr: 1 }} />
                Hierarchical Layout
              </MenuItem>
              
              <MenuItem 
                onClick={() => {
                  onLayout('force-directed');
                  setLayoutMenuAnchor(null);
                }}
                selected={layoutOptions.algorithm === 'force-directed'}
              >
                <ForceDirectedIcon sx={{ mr: 1 }} />
                Force-Directed Layout
              </MenuItem>
              
              <MenuItem 
                onClick={() => {
                  onLayout('grid');
                  setLayoutMenuAnchor(null);
                }}
                selected={layoutOptions.algorithm === 'grid'}
              >
                <GridIcon sx={{ mr: 1 }} />
                Grid Layout
              </MenuItem>
              
              <MenuItem 
                onClick={() => {
                  onLayout('smart');
                  setLayoutMenuAnchor(null);
                }}
                selected={layoutOptions.algorithm === 'smart'}
              >
                <SmartIcon sx={{ mr: 1 }} />
                Smart Layout
              </MenuItem>
              
              <Divider />
              
              <MenuItem 
                onClick={() => {
                  onLayout('execution-sequence');
                  setLayoutMenuAnchor(null);
                }}
                selected={layoutOptions.algorithm === 'execution-sequence'}
              >
                <TimelineIcon sx={{ mr: 1 }} />
                Execution Sequence Layout
              </MenuItem>
              
              <MenuItem 
                onClick={() => {
                  onLayout('execution-force-directed');
                  setLayoutMenuAnchor(null);
                }}
                selected={layoutOptions.algorithm === 'execution-force-directed'}
              >
                <ForceDirectedIcon sx={{ mr: 1 }} />
                Execution Force-Directed
              </MenuItem>
              
              <MenuItem 
                onClick={() => {
                  onLayout('execution-layered');
                  setLayoutMenuAnchor(null);
                }}
                selected={layoutOptions.algorithm === 'execution-layered'}
              >
                <LayersIcon sx={{ mr: 1 }} />
                Execution Layered Layout
              </MenuItem>
              
              <MenuItem 
                onClick={() => {
                  onLayout('execution-semantic');
                  setLayoutMenuAnchor(null);
                }}
                selected={layoutOptions.algorithm === 'execution-semantic'}
              >
                <CategoryIcon sx={{ mr: 1 }} />
                Semantic-Aware Layout
              </MenuItem>
              
              <Divider />
              
              <MenuItem 
                onClick={() => {
                  onLayoutDirectionChange(layoutOptions.direction === 'TB' ? 'LR' : 'TB');
                  setLayoutMenuAnchor(null);
                }}
              >
                <TransformIcon sx={{ mr: 1 }} />
                {layoutOptions.direction === 'TB' ? 'Switch to Left-Right' : 'Switch to Top-Bottom'}
              </MenuItem>
            </Menu>
            
            <Tooltip title={`Switch to ${workflowType === 'agent_based' ? 'Legacy' : 'Agent-Based'} Mode`}>
              <IconButton 
                onClick={() => setWorkflowType(workflowType === 'agent_based' ? 'legacy' : 'agent_based')} 
                size="small"
                color={workflowType === 'agent_based' ? 'success' : 'warning'}
              >
                {workflowType === 'agent_based' ? <PsychologyIcon /> : <BuildIcon />}
              </IconButton>
            </Tooltip>
            
            <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />
            
            <Tooltip title="Copy Selected Nodes (Ctrl+C)">
              <span>
                <IconButton 
                  onClick={copyNodes} 
                  size="small"
                  disabled={selectedNodeIds.length === 0}
                >
                  <ContentCopy />
                </IconButton>
              </span>
            </Tooltip>
            
            <Tooltip title="Duplicate Selected Nodes (Ctrl+D)">
              <span>
                <IconButton 
                  onClick={duplicateNodes} 
                  size="small"
                  disabled={selectedNodeIds.length === 0}
                >
                  <ContentCopy />
                </IconButton>
              </span>
            </Tooltip>
            
            <Tooltip title={isAutoSaving ? "Auto-saving workflow..." : "Quick Execute Workflow"}>
              <span>
                <IconButton 
                  onClick={handleQuickExecute} 
                  size="small" 
                  disabled={!workflowId || isExecuting || isAutoSaving}
                  color={isExecuting ? 'warning' : isAutoSaving ? 'info' : 'default'}
                >
                  {isExecuting || isAutoSaving ? <CircularProgress size={20} /> : <PlayIcon />}
                </IconButton>
              </span>
            </Tooltip>
            
            <Tooltip title={isSaving ? "Saving workflow..." : "Save Workflow"}>
              <span>
                <IconButton 
                  onClick={handleSave} 
                  size="small"
                  disabled={isSaving || isAutoSaving}
                  color={isSaving ? 'info' : 'default'}
                >
                  {isSaving ? <CircularProgress size={20} /> : <SaveIcon />}
                </IconButton>
              </span>
            </Tooltip>
            
            <Tooltip title={showAllIO ? "Collapse All I/O" : "Expand All I/O"}>
              <IconButton 
                onClick={handleToggleAllIO} 
                size="small"
                color={showAllIO ? 'primary' : 'default'}
              >
                {showAllIO ? <CollapseAllIcon /> : <ExpandAllIcon />}
              </IconButton>
            </Tooltip>
            
            <Tooltip title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}>
              <IconButton onClick={() => setIsFullscreen(!isFullscreen)} size="small">
                {isFullscreen ? <FullscreenExitIcon /> : <FullscreenIcon />}
              </IconButton>
            </Tooltip>
            
            <IconButton onClick={onBack || onClose} size="small">
              <CloseIcon />
            </IconButton>
          </Box>
        </Box>
      </Box>

      <Box component={onClose ? DialogContent : Box} sx={{ p: 0, display: 'flex', flexDirection: 'column', overflow: 'visible', flex: 1 }}>
        {/* Node Palette */}
        <Paper elevation={1} sx={{ p: 1, borderBottom: 1, borderColor: 'divider' }}>
          <Box display="flex" alignItems="center" gap={1} flexWrap="wrap">
            <Typography variant="body2" sx={{ mr: 1, fontWeight: 600 }}>
              Add Nodes:
            </Typography>
            {getCurrentNodeTemplates().map((template: any) => (
              <Tooltip key={template?.type || 'unknown'} title={template?.description || 'Node template'}>
                <Button
                  variant={selectedNodeType === template?.type ? "contained" : "outlined"}
                  size="small"
                  startIcon={template?.icon}
                  onClick={() => addNode(template?.type || 'unknown')}
                  draggable
                  onDragStart={(event) => {
                    event.dataTransfer.setData('application/reactflow', template?.type || 'unknown');
                    event.dataTransfer.effectAllowed = 'move';
                  }}
                  sx={{
                    borderColor: template?.color || '#666',
                    color: selectedNodeType === template?.type ? 'white' : (template?.color || '#666'),
                    backgroundColor: selectedNodeType === template?.type ? (template?.color || '#666') : 'transparent',
                    '&:hover': {
                      backgroundColor: template?.color || '#666',
                      color: 'white'
                    }
                  }}
                >
                  {template?.label || 'Unknown'}
                </Button>
              </Tooltip>
            ))}
          </Box>
        </Paper>

        {/* Execution Interface - Message Field and Controls */}
        {workflowType === 'agent_based' && (
          <Paper elevation={1} sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
            <Box display="flex" alignItems="center" gap={2}>
              <Typography variant="body2" sx={{ fontWeight: 600, minWidth: 'fit-content' }}>
                Execute with Message:
              </Typography>
              <TextField
                value={executionMessage}
                onChange={(e) => setExecutionMessage(e.target.value)}
                placeholder="Enter a message to trigger the workflow..."
                variant="outlined"
                size="small"
                fullWidth
                disabled={isExecuting || isAutoSaving}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleExecuteWithMessage();
                  }
                }}
              />
              <Button
                variant="contained"
                startIcon={isExecuting || isAutoSaving ? <CircularProgress size={16} /> : <PlayIcon />}
                onClick={handleExecuteWithMessage}
                disabled={!workflowId || isExecuting || isAutoSaving}
                size="small"
              >
                {isAutoSaving ? 'Auto-saving...' : 'Execute'}
              </Button>
            </Box>
          </Paper>
        )}


        {/* React Flow Canvas */}
        <Box sx={{ flex: 1, height: '100%', overflow: 'visible' }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onSelectionChange={({ nodes }) => handleSelectionChange(nodes)}
            onNodeContextMenu={handleNodeContextMenu}
            onDrop={(event) => {
              event.preventDefault();
              const type = event.dataTransfer.getData('application/reactflow');
              if (typeof type === 'undefined' || !type) {
                return;
              }
              
              const reactFlowBounds = event.currentTarget.getBoundingClientRect();
              const position = {
                x: event.clientX - reactFlowBounds.left,
                y: event.clientY - reactFlowBounds.top,
              };
              
              // Create node at dropped position using the same logic as addNode
              const template = getCurrentNodeTemplate(type);
              if (!template) {
                return;
              }
              
              const handles = getCurrentNodeHandles(type);
              const nodeData: any = {
                label: template?.label || type,
                inputHandles: handles.inputs,
                outputHandles: handles.outputs,
                nodeTemplate: template
              };
              
              // Add node-specific initialization (same as addNode function)
              if (workflowType === 'agent_based') {
                switch (type) {
                  case 'agentnode':
                    nodeData.agentName = '';
                    nodeData.customPrompt = '';
                    nodeData.tools = [];
                    break;
                  case 'inputnode':
                    nodeData.inputType = 'text';
                    nodeData.defaultValue = '';
                    break;
                  case 'outputnode':
                    nodeData.outputType = 'text';
                    nodeData.outputField = '';
                    break;
                  case 'cachenode':
                    nodeData.cacheKey = '';
                    nodeData.cacheKeyPattern = 'auto';
                    nodeData.ttl = 3600;
                    break;
                }
              }
              
              const newNode = {
                id: `${type}-${nodeCounter}`,
                type: type,
                position,
                data: nodeData,
                draggable: true,
              };
              
              setNodesWithChangeTracking((nds) => {
                const updatedNodes = nds.concat(newNode);
                return calculateAndAssignSequenceInfo(updatedNodes, edges);
              });
              setNodeCounter(prev => prev + 1);
            }}
            onDragOver={(event) => {
              event.preventDefault();
              event.dataTransfer.dropEffect = 'move';
            }}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            fitView
            multiSelectionKeyCode="Meta"
            selectionOnDrag={true}
            panOnDrag={[1, 2]}
            nodesDraggable={true}
            selectNodesOnDrag={false}
            zoomOnScroll={!accordionOpen}
            zoomOnPinch={true}
            zoomOnDoubleClick={true}
            style={{ overflow: 'visible' }}
            defaultEdgeOptions={{
              type: 'smoothstep',
              animated: showFlowAnimation,
              markerEnd: {
                type: MarkerType.ArrowClosed,
                width: 20,
                height: 20,
                color: '#555',
              },
              style: {
                strokeWidth: 3,
                stroke: '#666',
              },
            }}
            connectionLineStyle={{
              strokeWidth: 2,
              stroke: '#b1b1b7',
              strokeDasharray: '5,5',
            }}
            connectionLineType={ConnectionLineType.SmoothStep}
          >
            <Background variant={BackgroundVariant.Dots} />
            <Controls />
            <MiniMap style={miniMapStyle} zoomable pannable />
            
            <Panel position="top-right">
              <Alert severity="info" sx={{ maxWidth: 300 }}>
                <Typography variant="caption">
                  <strong>Tips:</strong><br />
                  • Drag nodes from the toolbar above<br />
                  • Connect nodes by dragging handles<br />
                  • Right-click nodes for copy/duplicate<br />
                  • Use Ctrl+C/V/D for shortcuts<br />
                  • Each node has 4 connection points
                  {showFlowAnimation && (
                    <>
                      <br />
                      <strong>Flow animation enabled</strong>
                    </>
                  )}
                </Typography>
              </Alert>
            </Panel>
          </ReactFlow>
        </Box>
      </Box>

      {!isFullscreen && (
        <Box component={onClose ? DialogActions : Box} sx={{ px: 3, py: 2, borderTop: 1, borderColor: 'divider' }}>
          <Box display="flex" justifyContent="space-between" alignItems="center" width="100%">
            <Typography variant="body2" color="text.secondary">
              Nodes: {nodes.length} | Connections: {edges.length}
            </Typography>
            
            <Box display="flex" gap={1}>
              <Button onClick={() => onLayout()} variant="outlined" size="small">
                Auto Layout
              </Button>
              <Button 
                onClick={handleSave} 
                variant="contained" 
                size="small" 
                startIcon={isSaving ? <CircularProgress size={16} /> : <SaveIcon />}
                disabled={isSaving || isAutoSaving}
              >
                {isSaving ? 'Saving...' : 'Save Workflow'}
              </Button>
              <Button onClick={onBack || onClose} variant="outlined" size="small">
                {onBack ? 'Back' : 'Close'}
              </Button>
            </Box>
          </Box>
        </Box>
      )}

      {/* Success notification */}
      <Snackbar
        open={saveSuccess}
        autoHideDuration={3000}
        onClose={() => setSaveSuccess(false)}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert onClose={() => setSaveSuccess(false)} severity="success" sx={{ width: '100%' }}>
          Workflow saved successfully! You can continue editing.
        </Alert>
      </Snackbar>

      {/* Execution Error Snackbar */}
      <Snackbar
        open={!!executionError}
        autoHideDuration={6000}
        onClose={() => setExecutionError(null)}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert onClose={() => setExecutionError(null)} severity="error" sx={{ width: '100%' }}>
          <Typography variant="body2">
            <strong>Execution Failed:</strong> {executionError}
          </Typography>
        </Alert>
      </Snackbar>

      {/* Enhanced Workflow Completion Dialog */}
      <EnhancedWorkflowCompletionDialog
        open={completionDialogOpen}
        onViewNow={handleViewNow}
        onSaveOnly={handleSaveOnly}
        onViewAndSave={handleViewAndSave}
        onViewLater={handleViewLater}
        onDismiss={handleDismiss}
        workflowName={currentWorkflowName}
        outputFormat={workflowResult?.format}
        executionTime={workflowResult?.metadata?.executionTime}
        metadata={workflowResult?.metadata}
      />

      {/* Note Viewer Dialog */}
      <NoteViewer
        open={noteViewerOpen}
        onClose={handleCloseNoteViewer}
        content={workflowResult?.content || ''}
        title={`${currentWorkflowName} - Report`}
        format={workflowResult?.format as any}
        metadata={workflowResult?.metadata}
      />

      {/* Node Context Menu */}
      <Menu
        open={Boolean(nodeContextMenu)}
        onClose={closeNodeContextMenu}
        anchorReference="anchorPosition"
        anchorPosition={
          nodeContextMenu
            ? { top: nodeContextMenu.y, left: nodeContextMenu.x }
            : undefined
        }
      >
        <MenuItem
          onClick={() => {
            copyNodes();
            closeNodeContextMenu();
          }}
          disabled={selectedNodeIds.length === 0}
        >
          <ContentCopy sx={{ mr: 1, fontSize: 18 }} />
          Copy {selectedNodeIds.length > 1 ? `(${selectedNodeIds.length} nodes)` : ''}
        </MenuItem>
        
        <MenuItem
          onClick={() => {
            duplicateNodes();
            closeNodeContextMenu();
          }}
          disabled={selectedNodeIds.length === 0}
        >
          <ContentCopy sx={{ mr: 1, fontSize: 18 }} />
          Duplicate {selectedNodeIds.length > 1 ? `(${selectedNodeIds.length} nodes)` : ''}
        </MenuItem>
        
        <MenuItem
          onClick={() => {
            pasteNodes();
            closeNodeContextMenu();
          }}
          disabled={copiedNodes.length === 0}
        >
          <ContentCopy sx={{ mr: 1, fontSize: 18 }} />
          Paste {copiedNodes.length > 1 ? `(${copiedNodes.length} nodes)` : ''}
        </MenuItem>
        
        <Divider />
        
        <MenuItem
          onClick={() => {
            // Delete selected nodes
            setNodesWithChangeTracking((nds) => nds.filter(node => !selectedNodeIds.includes(node.id)));
            setEdgesWithChangeTracking((eds) => eds.filter(edge => 
              !selectedNodeIds.includes(edge.source) && !selectedNodeIds.includes(edge.target)
            ));
            setSelectedNodeIds([]);
            closeNodeContextMenu();
          }}
          disabled={selectedNodeIds.length === 0}
        >
          <CloseIcon sx={{ mr: 1, fontSize: 18, color: 'error.main' }} />
          Delete {selectedNodeIds.length > 1 ? `(${selectedNodeIds.length} nodes)` : ''}
        </MenuItem>
      </Menu>

    </>
  );

  // If onClose is provided, wrap in Dialog (for use in App.tsx)
  // Otherwise, return the content directly (for use in WorkflowApp)
  if (onClose) {
    return (
      <Dialog
        open={open}
        onClose={onClose}
        maxWidth={false}
        fullScreen={isFullscreen}
        sx={{
          '& .MuiDialog-paper': {
            width: '100vw',
            height: '100vh',
            maxWidth: 'none',
            maxHeight: 'none',
            margin: 0,
            borderRadius: 0
          }
        }}
        data-testid="workflow-editor"
      >
        {editorContent}
      </Dialog>
    );
  }

  // Standalone mode (for WorkflowApp)
  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }} data-testid="workflow-editor">
      {editorContent}
    </Box>
  );
};

export default CustomWorkflowEditor;