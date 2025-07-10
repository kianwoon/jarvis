import { useState, useEffect, useCallback } from 'react';

interface DocumentState {
  tempDocId: string;
  filename: string;
  status: 'uploading' | 'processing' | 'ready' | 'error';
  isIncluded: boolean;
  metadata: {
    upload_timestamp?: string;
    file_size?: number;
    chunk_count?: number;
    in_memory_rag_enabled?: boolean;
    [key: string]: any;
  };
  uploadProgress?: number;
  error?: string;
}

interface DocumentStateHook {
  documents: DocumentState[];
  loading: boolean;
  error: string | null;
  uploadDocument: (file: File, options?: UploadOptions) => Promise<DocumentState | null>;
  deleteDocument: (tempDocId: string) => Promise<boolean>;
  toggleDocumentInclusion: (tempDocId: string, included: boolean) => Promise<boolean>;
  refreshDocuments: () => Promise<void>;
  clearDocuments: () => Promise<void>;
  getActiveDocuments: () => DocumentState[];
  getTotalDocuments: () => number;
}

interface UploadOptions {
  ttlHours?: number;
  autoInclude?: boolean;
  enableInMemoryRag?: boolean;
}

export const useDocumentState = (conversationId: string): DocumentStateHook => {
  const [documents, setDocuments] = useState<DocumentState[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Refresh documents from server
  const refreshDocuments = useCallback(async () => {
    if (!conversationId) return;
    
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`/api/v1/temp-documents/list/${conversationId}?include_stats=true`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch documents: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Transform server response to local state format
      const transformedDocs: DocumentState[] = data.documents.map((doc: any) => ({
        tempDocId: doc.temp_doc_id,
        filename: doc.filename,
        status: doc.status === 'ready' ? 'ready' : doc.status,
        isIncluded: doc.is_included || false,
        metadata: doc.metadata || {},
        error: doc.status === 'error' ? doc.error : undefined
      }));
      
      setDocuments(transformedDocs);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch documents';
      setError(errorMessage);
      console.error('Failed to refresh documents:', err);
    } finally {
      setLoading(false);
    }
  }, [conversationId]);

  // Upload a new document
  const uploadDocument = useCallback(async (
    file: File, 
    options: UploadOptions = {}
  ): Promise<DocumentState | null> => {
    if (!conversationId) {
      setError('No conversation ID provided');
      return null;
    }

    const {
      ttlHours = 2,
      autoInclude = true,
      enableInMemoryRag = true
    } = options;

    try {
      setError(null);
      
      // Create optimistic document state
      const tempDocId = `temp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const optimisticDoc: DocumentState = {
        tempDocId,
        filename: file.name,
        status: 'uploading',
        isIncluded: autoInclude,
        metadata: {
          file_size: file.size,
          upload_timestamp: new Date().toISOString(),
          in_memory_rag_enabled: enableInMemoryRag
        },
        uploadProgress: 0
      };

      // Add optimistic document to state
      setDocuments(prev => [...prev, optimisticDoc]);

      // Prepare form data
      const formData = new FormData();
      formData.append('file', file);
      formData.append('conversation_id', conversationId);
      formData.append('ttl_hours', ttlHours.toString());
      formData.append('auto_include', autoInclude.toString());
      formData.append('enable_in_memory_rag', enableInMemoryRag.toString());

      // Upload with progress tracking
      const response = await fetch('/api/v1/temp-documents/upload-with-progress', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      // Handle SSE progress updates
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body reader available');
      }

      const decoder = new TextDecoder();
      let finalResult: DocumentState | null = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.substring(6));
              
              // Update progress
              if (data.progress !== undefined) {
                setDocuments(prev => prev.map(doc => 
                  doc.tempDocId === tempDocId 
                    ? { ...doc, uploadProgress: data.progress, status: data.status === 'error' ? 'error' : 'uploading' }
                    : doc
                ));
              }

              // Handle completion
              if (data.status === 'success' && data.result) {
                const result = data.result;
                finalResult = {
                  tempDocId: result.temp_doc_id,
                  filename: file.name,
                  status: 'ready',
                  isIncluded: autoInclude,
                  metadata: result.metadata || {},
                  uploadProgress: 100
                };

                setDocuments(prev => prev.map(doc => 
                  doc.tempDocId === tempDocId ? finalResult! : doc
                ));
              }

              // Handle errors
              if (data.status === 'error') {
                setDocuments(prev => prev.map(doc => 
                  doc.tempDocId === tempDocId 
                    ? { ...doc, status: 'error', error: data.error }
                    : doc
                ));
              }
            } catch (parseError) {
              console.warn('Failed to parse SSE data:', parseError);
            }
          }
        }
      }

      return finalResult;

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Upload failed';
      setError(errorMessage);
      
      // Update optimistic document with error
      setDocuments(prev => prev.map(doc => 
        doc.tempDocId === tempDocId 
          ? { ...doc, status: 'error', error: errorMessage }
          : doc
      ));
      
      console.error('Document upload failed:', err);
      return null;
    }
  }, [conversationId]);

  // Delete a document
  const deleteDocument = useCallback(async (tempDocId: string): Promise<boolean> => {
    try {
      setError(null);
      
      // Optimistically remove from state
      setDocuments(prev => prev.filter(doc => doc.tempDocId !== tempDocId));

      const response = await fetch(`/api/v1/temp-documents/delete/${tempDocId}?conversation_id=${conversationId}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        // Restore document on failure
        await refreshDocuments();
        throw new Error(`Delete failed: ${response.statusText}`);
      }

      return true;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Delete failed';
      setError(errorMessage);
      console.error('Document deletion failed:', err);
      return false;
    }
  }, [conversationId, refreshDocuments]);

  // Toggle document inclusion
  const toggleDocumentInclusion = useCallback(async (
    tempDocId: string, 
    included: boolean
  ): Promise<boolean> => {
    try {
      setError(null);
      
      // Optimistically update state
      setDocuments(prev => prev.map(doc => 
        doc.tempDocId === tempDocId 
          ? { ...doc, isIncluded: included }
          : doc
      ));

      const response = await fetch(`/api/v1/temp-documents/preferences/${tempDocId}?conversation_id=${conversationId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          is_included: included
        })
      });

      if (!response.ok) {
        // Revert optimistic update
        setDocuments(prev => prev.map(doc => 
          doc.tempDocId === tempDocId 
            ? { ...doc, isIncluded: !included }
            : doc
        ));
        throw new Error(`Update failed: ${response.statusText}`);
      }

      return true;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Update failed';
      setError(errorMessage);
      console.error('Document toggle failed:', err);
      return false;
    }
  }, [conversationId]);

  // Clear all documents
  const clearDocuments = useCallback(async (): Promise<void> => {
    try {
      setError(null);
      
      const response = await fetch(`/api/v1/temp-documents/cleanup/${conversationId}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        setDocuments([]);
      } else {
        throw new Error(`Cleanup failed: ${response.statusText}`);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Cleanup failed';
      setError(errorMessage);
      console.error('Document cleanup failed:', err);
    }
  }, [conversationId]);

  // Get active documents
  const getActiveDocuments = useCallback((): DocumentState[] => {
    return documents.filter(doc => doc.isIncluded && doc.status === 'ready');
  }, [documents]);

  // Get total document count
  const getTotalDocuments = useCallback((): number => {
    return documents.length;
  }, [documents]);

  // Load documents on mount and conversation change
  useEffect(() => {
    if (conversationId) {
      refreshDocuments();
    }
  }, [conversationId, refreshDocuments]);

  return {
    documents,
    loading,
    error,
    uploadDocument,
    deleteDocument,
    toggleDocumentInclusion,
    refreshDocuments,
    clearDocuments,
    getActiveDocuments,
    getTotalDocuments
  };
};