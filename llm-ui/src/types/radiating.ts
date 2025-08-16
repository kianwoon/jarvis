// TypeScript interfaces for Universal Radiating Coverage System

export interface RadiatingConfig {
  enabled: boolean;
  maxDepth: number;
  strategy: 'breadth-first' | 'depth-first' | 'best-first' | 'adaptive';
  relevanceThreshold: number;
  maxEntitiesPerLevel: number;
  includeRelationships: boolean;
  autoExpand: boolean;
  cacheResults: boolean;
  timeoutMs: number;
}

export interface RadiatingEntity {
  id: string;
  name: string;
  type: string;
  depth: number;
  relevanceScore: number;
  metadata?: Record<string, any>;
  parentId?: string;
  children?: RadiatingEntity[];
  relationships?: RadiatingRelationship[];
  isExpanded?: boolean;
  isLoading?: boolean;
}

export interface RadiatingRelationship {
  id: string;
  sourceId: string;
  targetId: string;
  type: string;
  weight: number;
  metadata?: Record<string, any>;
}

export interface RadiatingProgress {
  isActive: boolean;
  currentDepth: number;
  totalDepth: number;
  entitiesDiscovered: number;
  relationshipsFound: number;
  processedEntities: number;
  queueSize: number;
  elapsedTime: number;
  estimatedTimeRemaining?: number;
  currentEntity?: string;
  status: 'idle' | 'initializing' | 'traversing' | 'processing' | 'completing' | 'completed' | 'error';
  error?: string;
}

export interface RadiatingResults {
  rootEntity: RadiatingEntity;
  entities: RadiatingEntity[];
  relationships: RadiatingRelationship[];
  totalEntities: number;
  totalRelationships: number;
  maxDepthReached: number;
  processingTime: number;
  strategy: string;
  relevanceThreshold: number;
  timestamp: Date;
}

export interface RadiatingVisualizationNode {
  id: string;
  name: string;
  type: string;
  group: number; // depth level
  radius: number; // based on relevance
  color: string;
  x?: number;
  y?: number;
  fx?: number; // fixed position
  fy?: number;
}

export interface RadiatingVisualizationLink {
  source: string;
  target: string;
  value: number; // relationship weight
  type: string;
}

export interface RadiatingVisualizationData {
  nodes: RadiatingVisualizationNode[];
  links: RadiatingVisualizationLink[];
}

// API Request/Response types
export interface RadiatingToggleRequest {
  enabled: boolean;
  conversation_id?: string;
}

export interface RadiatingToggleResponse {
  enabled: boolean;
  message: string;
}

export interface RadiatingConfigRequest {
  config: RadiatingConfig;
  conversation_id?: string;
}

export interface RadiatingConfigResponse {
  success: boolean;
  config: RadiatingConfig;
  message?: string;
}

export interface RadiatingStartRequest {
  query: string;
  config?: Partial<RadiatingConfig>;
  conversation_id?: string;
}

export interface RadiatingStartResponse {
  job_id: string;
  status: 'started' | 'queued' | 'error';
  message?: string;
}

export interface RadiatingStatusRequest {
  job_id: string;
}

export interface RadiatingStatusResponse {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  progress: RadiatingProgress;
  results?: RadiatingResults;
  error?: string;
}

export interface RadiatingCancelRequest {
  job_id: string;
}

export interface RadiatingCancelResponse {
  success: boolean;
  message: string;
}

export interface RadiatingExportRequest {
  results: RadiatingResults;
  format: 'json' | 'csv' | 'graphml';
}

export interface RadiatingExportResponse {
  success: boolean;
  data?: string;
  download_url?: string;
  error?: string;
}

// Settings persistence
export interface RadiatingSettings {
  defaultConfig: RadiatingConfig;
  presets: RadiatingPreset[];
  visualizationPreferences: {
    nodeSize: 'fixed' | 'relevance' | 'connections';
    linkThickness: 'fixed' | 'weight';
    colorScheme: 'depth' | 'type' | 'relevance';
    layout: 'force' | 'radial' | 'hierarchical';
    showLabels: boolean;
    animationSpeed: number;
  };
}

export interface RadiatingPreset {
  id: string;
  name: string;
  description: string;
  config: RadiatingConfig;
  isDefault?: boolean;
}