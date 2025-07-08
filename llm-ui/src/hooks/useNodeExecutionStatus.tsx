import { useState, useEffect, useCallback } from 'react';

export interface NodeExecutionStatus {
  status: 'idle' | 'running' | 'success' | 'error' | 'warning';
  message?: string;
  progress?: number;
  startTime?: Date;
  endTime?: Date;
  duration?: number;
  output?: any;
  error?: string;
}

export interface UseNodeExecutionStatusOptions {
  nodeId: string;
  onStatusChange?: (status: NodeExecutionStatus) => void;
  autoReset?: boolean;
  resetDelay?: number;
}

export const useNodeExecutionStatus = (options: UseNodeExecutionStatusOptions) => {
  const { nodeId, onStatusChange, autoReset = false, resetDelay = 3000 } = options;

  const [status, setStatus] = useState<NodeExecutionStatus>({
    status: 'idle'
  });

  const updateStatus = useCallback((newStatus: Partial<NodeExecutionStatus>) => {
    setStatus(prev => {
      const updated = { ...prev, ...newStatus };
      
      // Calculate duration if ending
      if (newStatus.status && ['success', 'error', 'warning'].includes(newStatus.status) && prev.startTime && !updated.endTime) {
        updated.endTime = new Date();
        updated.duration = updated.endTime.getTime() - prev.startTime.getTime();
      }
      
      onStatusChange?.(updated);
      return updated;
    });
  }, [onStatusChange]);

  const startExecution = useCallback((message?: string) => {
    updateStatus({
      status: 'running',
      message,
      progress: 0,
      startTime: new Date(),
      endTime: undefined,
      duration: undefined,
      output: undefined,
      error: undefined
    });
  }, [updateStatus]);

  const updateProgress = useCallback((progress: number, message?: string) => {
    updateStatus({
      progress: Math.max(0, Math.min(100, progress)),
      message
    });
  }, [updateStatus]);

  const completeExecution = useCallback((output?: any, message?: string) => {
    updateStatus({
      status: 'success',
      progress: 100,
      output,
      message: message || 'Execution completed successfully'
    });
  }, [updateStatus]);

  const failExecution = useCallback((error: string, output?: any) => {
    updateStatus({
      status: 'error',
      error,
      output,
      message: `Execution failed: ${error}`
    });
  }, [updateStatus]);

  const warnExecution = useCallback((warning: string, output?: any) => {
    updateStatus({
      status: 'warning',
      error: warning,
      output,
      message: `Execution completed with warnings: ${warning}`
    });
  }, [updateStatus]);

  const resetStatus = useCallback(() => {
    setStatus({
      status: 'idle'
    });
  }, []);

  // Auto-reset after completion
  useEffect(() => {
    if (autoReset && ['success', 'error', 'warning'].includes(status.status)) {
      const timer = setTimeout(resetStatus, resetDelay);
      return () => clearTimeout(timer);
    }
  }, [status.status, autoReset, resetDelay, resetStatus]);

  // Get visual properties based on status
  const getStatusColor = useCallback(() => {
    switch (status.status) {
      case 'running':
        return '#2196f3';
      case 'success':
        return '#4caf50';
      case 'error':
        return '#f44336';
      case 'warning':
        return '#ff9800';
      default:
        return '#9e9e9e';
    }
  }, [status.status]);

  const getStatusIcon = useCallback(() => {
    switch (status.status) {
      case 'running':
        return '⚡';
      case 'success':
        return '✅';
      case 'error':
        return '❌';
      case 'warning':
        return '⚠️';
      default:
        return '⭕';
    }
  }, [status.status]);

  const isExecuting = status.status === 'running';
  const isCompleted = ['success', 'error', 'warning'].includes(status.status);
  const isIdle = status.status === 'idle';

  return {
    status,
    startExecution,
    updateProgress,
    completeExecution,
    failExecution,
    warnExecution,
    resetStatus,
    updateStatus,
    getStatusColor,
    getStatusIcon,
    isExecuting,
    isCompleted,
    isIdle
  };
};