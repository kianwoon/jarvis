import { useState, useEffect, useCallback } from 'react';

interface RAGModeState {
  priorityMode: boolean;
  hasActiveDocuments: boolean;
  fallbackToPersistent: boolean;
  strategy: 'temp_priority' | 'parallel_fusion' | 'persistent_only';
}

interface RAGModeHook {
  ragMode: RAGModeState;
  togglePriorityMode: () => void;
  setStrategy: (strategy: RAGModeState['strategy']) => void;
  setFallbackToPersistent: (enabled: boolean) => void;
  isInMemoryRAGActive: () => boolean;
  getRAGDescription: () => string;
  updateActiveDocumentCount: (count: number) => void;
}

interface RAGModeConfig {
  defaultPriorityMode?: boolean;
  defaultStrategy?: RAGModeState['strategy'];
  defaultFallbackToPersistent?: boolean;
  persistPreferences?: boolean;
}

export const useRAGMode = (
  conversationId: string,
  config: RAGModeConfig = {}
): RAGModeHook => {
  const {
    defaultPriorityMode = true,
    defaultStrategy = 'temp_priority',
    defaultFallbackToPersistent = true,
    persistPreferences = true
  } = config;

  const [ragMode, setRAGMode] = useState<RAGModeState>({
    priorityMode: defaultPriorityMode,
    hasActiveDocuments: false,
    fallbackToPersistent: defaultFallbackToPersistent,
    strategy: defaultStrategy
  });

  // Storage key for persisting preferences
  const storageKey = `rag-mode-${conversationId}`;

  // Load persisted preferences
  useEffect(() => {
    if (persistPreferences && conversationId) {
      try {
        const saved = localStorage.getItem(storageKey);
        if (saved) {
          const parsed = JSON.parse(saved);
          setRAGMode(prev => ({
            ...prev,
            priorityMode: parsed.priorityMode ?? defaultPriorityMode,
            fallbackToPersistent: parsed.fallbackToPersistent ?? defaultFallbackToPersistent,
            strategy: parsed.strategy ?? defaultStrategy
          }));
        }
      } catch (error) {
        console.warn('Failed to load RAG mode preferences:', error);
      }
    }
  }, [conversationId, storageKey, persistPreferences, defaultPriorityMode, defaultFallbackToPersistent, defaultStrategy]);

  // Save preferences when they change
  useEffect(() => {
    if (persistPreferences && conversationId) {
      try {
        const toSave = {
          priorityMode: ragMode.priorityMode,
          fallbackToPersistent: ragMode.fallbackToPersistent,
          strategy: ragMode.strategy
        };
        localStorage.setItem(storageKey, JSON.stringify(toSave));
      } catch (error) {
        console.warn('Failed to save RAG mode preferences:', error);
      }
    }
  }, [ragMode.priorityMode, ragMode.fallbackToPersistent, ragMode.strategy, storageKey, persistPreferences, conversationId]);

  // Toggle priority mode
  const togglePriorityMode = useCallback(() => {
    setRAGMode(prev => ({
      ...prev,
      priorityMode: !prev.priorityMode
    }));
  }, []);

  // Set RAG strategy
  const setStrategy = useCallback((strategy: RAGModeState['strategy']) => {
    setRAGMode(prev => ({
      ...prev,
      strategy
    }));
  }, []);

  // Set fallback to persistent RAG
  const setFallbackToPersistent = useCallback((enabled: boolean) => {
    setRAGMode(prev => ({
      ...prev,
      fallbackToPersistent: enabled
    }));
  }, []);

  // Update active document count
  const updateActiveDocumentCount = useCallback((count: number) => {
    setRAGMode(prev => ({
      ...prev,
      hasActiveDocuments: count > 0
    }));
  }, []);

  // Check if in-memory RAG is active
  const isInMemoryRAGActive = useCallback((): boolean => {
    return ragMode.priorityMode && ragMode.hasActiveDocuments;
  }, [ragMode.priorityMode, ragMode.hasActiveDocuments]);

  // Get human-readable description of current RAG mode
  const getRAGDescription = useCallback((): string => {
    if (!ragMode.hasActiveDocuments) {
      return 'No documents uploaded - using standard knowledge base';
    }

    if (!ragMode.priorityMode) {
      return 'Document priority mode disabled - using standard knowledge base';
    }

    switch (ragMode.strategy) {
      case 'temp_priority':
        return ragMode.fallbackToPersistent
          ? 'Prioritizing uploaded documents with knowledge base fallback'
          : 'Using uploaded documents only';
      
      case 'parallel_fusion':
        return 'Blending results from uploaded documents and knowledge base';
      
      case 'persistent_only':
        return 'Using standard knowledge base only';
      
      default:
        return 'Using hybrid document search';
    }
  }, [ragMode]);

  return {
    ragMode,
    togglePriorityMode,
    setStrategy,
    setFallbackToPersistent,
    isInMemoryRAGActive,
    getRAGDescription,
    updateActiveDocumentCount
  };
};