/**
 * Meta-Task API Service
 * Handles all API interactions for meta-task functionality
 */

interface MetaTaskTemplate {
  id: string;
  name: string;
  description?: string;
  template_type: string;
  template_config: Record<string, any>;
  input_schema?: Record<string, any>;
  output_schema?: Record<string, any>;
  default_settings?: Record<string, any>;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

interface MetaTaskWorkflow {
  id: string;
  template_id: string;
  name: string;
  description?: string;
  workflow_config: Record<string, any>;
  status: 'pending' | 'running' | 'completed' | 'failed';
  input_data?: Record<string, any>;
  output_data?: Record<string, any>;
  progress?: Record<string, any>;
  error_message?: string;
  started_at?: string;
  completed_at?: string;
  created_at: string;
  updated_at: string;
}

interface MetaTaskNode {
  id: string;
  workflow_id: string;
  name: string;
  node_type: string;
  node_config: Record<string, any>;
  position_x: number;
  position_y: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  input_data?: Record<string, any>;
  output_data?: Record<string, any>;
  execution_order?: number;
  error_message?: string;
  retry_count: number;
  max_retries: number;
  started_at?: string;
  completed_at?: string;
}

interface ExecutionEvent {
  type: 'phase_start' | 'phase_complete' | 'phase_error' | 'workflow_complete' | 'workflow_error' | 'progress';
  phase?: string;
  status?: string;
  progress?: number;
  message?: string;
  data?: any;
  error?: string;
  timestamp: string;
}

class MetaTaskAPIService {
  private baseUrl = '/api/v1/meta-task';

  // Template Operations
  async getTemplates(activeOnly: boolean = true): Promise<MetaTaskTemplate[]> {
    try {
      const response = await fetch(`${this.baseUrl}/templates?active_only=${activeOnly}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch templates: ${response.statusText}`);
      }
      const data = await response.json();
      return data.templates || [];
    } catch (error) {
      console.error('Error fetching templates:', error);
      throw error;
    }
  }

  async getTemplate(templateId: string): Promise<MetaTaskTemplate | null> {
    try {
      const response = await fetch(`${this.baseUrl}/templates/${templateId}`);
      if (response.status === 404) {
        return null;
      }
      if (!response.ok) {
        throw new Error(`Failed to fetch template: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error fetching template:', error);
      throw error;
    }
  }

  async createTemplate(templateData: Partial<MetaTaskTemplate>): Promise<MetaTaskTemplate> {
    try {
      const response = await fetch(`${this.baseUrl}/templates`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(templateData),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to create template');
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error creating template:', error);
      throw error;
    }
  }

  async updateTemplate(templateId: string, updates: Partial<MetaTaskTemplate>): Promise<MetaTaskTemplate> {
    try {
      const response = await fetch(`${this.baseUrl}/templates/${templateId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updates),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to update template');
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error updating template:', error);
      throw error;
    }
  }

  async deleteTemplate(templateId: string): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/templates/${templateId}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to delete template');
      }
    } catch (error) {
      console.error('Error deleting template:', error);
      throw error;
    }
  }

  async getWorkflows(): Promise<MetaTaskWorkflow[]> {
    try {
      const response = await fetch(`${this.baseUrl}/workflows`);
      if (!response.ok) {
        throw new Error(`Failed to fetch workflows: ${response.statusText}`);
      }
      const data = await response.json();
      return data.workflows || [];
    } catch (error) {
      console.error('Error fetching workflows:', error);
      throw error;
    }
  }

  // Workflow Operations
  async createWorkflow(templateId: string, name: string, description?: string, inputData?: Record<string, any>): Promise<MetaTaskWorkflow> {
    try {
      const response = await fetch(`${this.baseUrl}/workflows`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          template_id: templateId,
          name,
          description,
          input_data: inputData || {},
        }),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to create workflow');
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error creating workflow:', error);
      throw error;
    }
  }

  async updateWorkflow(workflowId: string, updates: {
    name?: string;
    description?: string;
    input_data?: Record<string, any>;
  }): Promise<MetaTaskWorkflow> {
    try {
      const response = await fetch(`${this.baseUrl}/workflows/${workflowId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updates),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to update workflow');
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error updating workflow:', error);
      throw error;
    }
  }

  // Execution Operations with Server-Sent Events
  executeWorkflow(workflowId: string, onEvent: (event: ExecutionEvent) => void): EventSource {
    const eventSource = new EventSource(`${this.baseUrl}/workflows/${workflowId}/execute`);
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onEvent(data);
      } catch (error) {
        console.error('Error parsing SSE event:', error);
      }
    };
    
    eventSource.onerror = (error) => {
      console.error('SSE connection error:', error);
      eventSource.close();
      onEvent({
        type: 'workflow_error',
        error: 'Connection lost',
        timestamp: new Date().toISOString(),
      });
    };
    
    return eventSource;
  }

  async executePhase(phaseConfig: Record<string, any>, inputData: Record<string, any>, context?: Record<string, any>): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/execute/phase`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          phase_config: phaseConfig,
          input_data: inputData,
          context,
        }),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to execute phase');
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error executing phase:', error);
      throw error;
    }
  }

  executeMultiPhaseWorkflow(phases: any[], inputData: Record<string, any>, onEvent: (event: ExecutionEvent) => void): EventSource {
    const params = new URLSearchParams();
    const requestBody = JSON.stringify({
      phases,
      input_data: inputData,
    });

    // Create a custom EventSource with POST support
    const eventSource = new EventSource(`${this.baseUrl}/execute/workflow`);
    
    // Since EventSource doesn't support POST directly, we'll use a workaround
    // by first initiating the workflow via POST, then connecting to SSE endpoint
    fetch(`${this.baseUrl}/execute/workflow`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: requestBody,
    }).then(response => {
      if (!response.ok) {
        throw new Error('Failed to start workflow execution');
      }
      
      // The response should be a stream
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      
      const readStream = async () => {
        if (!reader) return;
        
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const data = JSON.parse(line.slice(6));
                  onEvent(data);
                } catch (e) {
                  console.error('Error parsing event data:', e);
                }
              }
            }
          }
        } catch (error) {
          console.error('Stream reading error:', error);
          onEvent({
            type: 'workflow_error',
            error: 'Stream reading failed',
            timestamp: new Date().toISOString(),
          });
        }
      };
      
      readStream();
    }).catch(error => {
      console.error('Failed to start workflow:', error);
      onEvent({
        type: 'workflow_error',
        error: error.message,
        timestamp: new Date().toISOString(),
      });
    });
    
    return eventSource;
  }

  // Utility Operations
  async getHealth(): Promise<{ status: string; service: string; timestamp: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      if (!response.ok) {
        throw new Error('Health check failed');
      }
      return await response.json();
    } catch (error) {
      console.error('Error checking health:', error);
      throw error;
    }
  }

  async getStats(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/stats`);
      if (!response.ok) {
        throw new Error('Failed to fetch stats');
      }
      return await response.json();
    } catch (error) {
      console.error('Error fetching stats:', error);
      throw error;
    }
  }
}

// Export singleton instance
export const metaTaskAPI = new MetaTaskAPIService();

// Export types
export type {
  MetaTaskTemplate,
  MetaTaskWorkflow,
  MetaTaskNode,
  ExecutionEvent,
};