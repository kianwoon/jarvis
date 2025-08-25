// TypeScript interfaces and API client for Notebook functionality

// Interfaces matching backend models
export interface Notebook {
  id: string;
  name: string;
  description?: string;
  user_id?: string;
  source_filter?: Record<string, any>;
  metadata?: Record<string, any>;
  created_at: string;
  updated_at: string;
  document_count?: number;
  conversation_count?: number;
}

export interface Document {
  document_id: string;
  filename: string;
  file_type: string;
  file_size_bytes: number;
  processing_status: string;
  milvus_collection?: string;
  created_at: string;
}

export interface NotebookDocument {
  id: string;
  notebook_id: string;
  document_id: string;
  document_name?: string;
  document_type?: string;
  milvus_collection?: string;
  added_at: string;
  metadata?: Record<string, any>;
}

export interface NotebookWithDocuments extends Notebook {
  documents: NotebookDocument[];
  conversations: Array<{
    id: string;
    started_at: string;
    last_activity: string;
  }>;
}

// System-wide Document Management Interfaces
export interface SystemDocument {
  document_id: string;
  filename: string;
  file_type: string;
  file_size_bytes: number;
  processing_status: string;
  milvus_collection?: string;
  created_at: string;
  updated_at: string;
  chunks_processed: number;
  total_chunks: number;
  file_hash: string;
  notebook_count: number;
  notebooks_using: Array<{
    id: string;
    name: string;
    added_at: string;
  }>;
  is_orphaned: boolean;
  processing_completed_at?: string;
  can_be_deleted: boolean;
}

export interface PaginationInfo {
  page: number;
  page_size: number;
  total_count: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface SystemDocumentStats {
  total_documents: number;
  unique_file_types: number;
  unique_collections: number;
  total_size_bytes: number;
  completed_documents: number;
  failed_documents: number;
  orphaned_documents: number;
}

export interface CreateNotebookRequest {
  name: string;
  description?: string;
  user_id?: string;
  source_filter?: Record<string, any>;
  metadata?: Record<string, any>;
}

export interface UpdateNotebookRequest {
  name?: string;
  description?: string;
}

export interface NotebookChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  notebook_id: string;
  context?: Array<{
    content: string;
    source: string;
    score?: number;
  }>;
  metadata?: any;
}

export interface NotebookChatRequest {
  message: string;
  conversation_id?: string;
  include_context?: boolean;
  max_sources?: number;
}

export interface ApiResponse<T> {
  data: T;
  message?: string;
}

export interface NotebookListResponse {
  notebooks: Notebook[];
  total_count: number;
  page: number;
  page_size: number;
}

export interface ApiError {
  detail: string;
  status_code?: number;
}

// Document deletion interfaces
export interface DocumentUsageInfo {
  document_id: string;
  filename: string;
  file_type?: string;
  file_size_bytes?: number;
  milvus_collection?: string;
  notebook_count: number;
  notebooks_using: Array<{
    id: string;
    name: string;
    added_at?: string;
  }>;
  cross_references: number;
  deletion_impact: {
    will_remove_from_notebooks: number;
    will_delete_vectors: boolean;
    will_delete_cross_references: boolean;
  };
}

export interface DocumentDeleteRequest {
  document_ids: string[];
  remove_from_notebooks: boolean;
  confirm_permanent_deletion: boolean;
}

export interface DocumentDeletionSummary {
  document_id: string;
  started_at: string;
  completed_at?: string;
  success: boolean;
  milvus_deleted: boolean;
  database_deleted: boolean;
  notebooks_removed: number;
  neo4j_deleted: boolean;
  cache_cleared: boolean;
  errors: string[];
}

export interface DocumentDeleteResponse {
  success: boolean;
  message: string;
  total_requested?: number;
  successful_deletions?: number;
  failed_deletions?: number;
  deletion_details: DocumentDeletionSummary[];
  overall_errors: string[];
  timestamp?: string;
}

class NotebookAPI {
  private baseUrl = '/api/v1/notebooks';

  // Helper method for handling API responses
  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error occurred' }));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }
    return response.json();
  }

  // Notebook CRUD operations
  async getNotebooks(): Promise<Notebook[]> {
    const response = await fetch(this.baseUrl);
    const data = await this.handleResponse<NotebookListResponse>(response);
    return data.notebooks;
  }

  async getNotebook(notebookId: string): Promise<NotebookWithDocuments> {
    const response = await fetch(`${this.baseUrl}/${notebookId}`);
    const data = await this.handleResponse<NotebookWithDocuments>(response);
    return data;
  }

  async createNotebook(request: CreateNotebookRequest): Promise<Notebook> {
    const response = await fetch(this.baseUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request)
    });
    const data = await this.handleResponse<Notebook>(response);
    return data;
  }

  async updateNotebook(notebookId: string, request: UpdateNotebookRequest): Promise<Notebook> {
    const response = await fetch(`${this.baseUrl}/${notebookId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request)
    });
    const data = await this.handleResponse<Notebook>(response);
    return data;
  }

  async deleteNotebook(notebookId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/${notebookId}`, {
      method: 'DELETE'
    });
    await this.handleResponse<void>(response);
  }

  // Document management in notebooks
  async addDocumentToNotebook(notebookId: string, documentId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/${notebookId}/documents/${documentId}`, {
      method: 'POST'
    });
    await this.handleResponse<void>(response);
  }

  async removeDocumentFromNotebook(notebookId: string, documentId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/${notebookId}/documents/${documentId}`, {
      method: 'DELETE'
    });
    await this.handleResponse<void>(response);
  }

  async getNotebookDocuments(notebookId: string): Promise<NotebookDocument[]> {
    const response = await fetch(`${this.baseUrl}/${notebookId}/documents`);
    const data = await this.handleResponse<ApiResponse<NotebookDocument[]>>(response);
    return data.data;
  }

  // Get available documents that can be added to notebook
  async getAvailableDocuments(): Promise<Document[]> {
    const response = await fetch('/api/v1/documents/');
    const data = await this.handleResponse<{documents: Document[], total_count: number, page: number, page_size: number}>(response);
    return data.documents;
  }

  // Document permanent deletion methods
  async getDocumentUsageInfo(documentId: string): Promise<DocumentUsageInfo> {
    const response = await fetch(`${this.baseUrl}/documents/${documentId}/usage`);
    const data = await this.handleResponse<DocumentUsageInfo>(response);
    return data;
  }

  async deleteDocumentsPermanently(request: DocumentDeleteRequest): Promise<DocumentDeleteResponse> {
    const response = await fetch(`${this.baseUrl}/documents/permanent`, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request)
    });
    const data = await this.handleResponse<DocumentDeleteResponse>(response);
    return data;
  }

  async deleteSingleDocumentPermanently(
    documentId: string, 
    options?: { remove_from_notebooks?: boolean; confirm_permanent_deletion?: boolean }
  ): Promise<DocumentDeleteResponse> {
    const request: DocumentDeleteRequest = {
      document_ids: [documentId],
      remove_from_notebooks: options?.remove_from_notebooks ?? true,
      confirm_permanent_deletion: options?.confirm_permanent_deletion ?? false
    };
    
    const response = await fetch(`${this.baseUrl}/documents/${documentId}/permanent`, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request)
    });
    const data = await this.handleResponse<DocumentDeleteResponse>(response);
    return data;
  }

  // Notebook-scoped chat functionality
  async startNotebookChat(notebookId: string, request: NotebookChatRequest): Promise<Response> {
    try {
      const response = await fetch(`${this.baseUrl}/${notebookId}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request)
      });
      
      return response;
    } catch (error) {
      console.error('Error in startNotebookChat:', error);
      throw error;
    }
  }

  // Search within notebook documents
  async searchNotebook(notebookId: string, query: string, limit: number = 10): Promise<Array<{
    content: string;
    source: string;
    score: number;
  }>> {
    const response = await fetch(`${this.baseUrl}/${notebookId}/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query, limit })
    });
    const data = await this.handleResponse<ApiResponse<Array<{
      content: string;
      source: string;
      score: number;
    }>>>(response);
    return data.data;
  }

  // Upload document directly to notebook
  async uploadDocumentToNotebook(notebookId: string, file: File): Promise<Document> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/${notebookId}/upload`, {
      method: 'POST',
      body: formData
    });
    const data = await this.handleResponse<ApiResponse<Document>>(response);
    return data.data;
  }

  // Get notebook statistics
  async getNotebookStats(notebookId: string): Promise<{
    document_count: number;
    total_size: number;
    last_updated: string;
  }> {
    const response = await fetch(`${this.baseUrl}/${notebookId}/stats`);
    const data = await this.handleResponse<ApiResponse<{
      document_count: number;
      total_size: number;
      last_updated: string;
    }>>(response);
    return data.data;
  }

  // System-wide Document Management
  async getAllSystemDocuments(options: {
    page?: number;
    page_size?: number;
    search?: string;
    file_type?: string;
    status?: string;
    collection?: string;
    sort_by?: string;
    sort_order?: 'asc' | 'desc';
  } = {}): Promise<{
    documents: SystemDocument[];
    pagination: PaginationInfo;
    summary_stats: SystemDocumentStats;
    filters_applied: any;
  }> {
    const params = new URLSearchParams();
    
    if (options.page) params.append('page', options.page.toString());
    if (options.page_size) params.append('page_size', options.page_size.toString());
    if (options.search) params.append('search', options.search);
    if (options.file_type) params.append('file_type', options.file_type);
    if (options.status) params.append('status', options.status);
    if (options.collection) params.append('collection', options.collection);
    if (options.sort_by) params.append('sort_by', options.sort_by);
    if (options.sort_order) params.append('sort_order', options.sort_order);

    const response = await fetch(`${this.baseUrl}/system/documents?${params.toString()}`);
    return await this.handleResponse(response);
  }
}

// Create singleton instance
export const notebookAPI = new NotebookAPI();

// Error handling utilities
export const isApiError = (error: any): error is ApiError => {
  return error && typeof error.detail === 'string';
};

export const getErrorMessage = (error: unknown): string => {
  if (error instanceof Error) {
    return error.message;
  }
  if (isApiError(error)) {
    return error.detail;
  }
  return 'An unexpected error occurred';
};

// Utility functions for formatting
export const formatFileSize = (bytes: number): string => {
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  if (bytes === 0) return '0 Bytes';
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
};

export const formatDate = (dateString: string): string => {
  const date = new Date(dateString);
  return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

export const formatRelativeTime = (dateString: string): string => {
  const date = new Date(dateString);
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);
  
  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;
  return formatDate(dateString);
};

export default notebookAPI;