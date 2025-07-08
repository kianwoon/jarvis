import { useState, useCallback, useRef, useEffect } from 'react';

export interface UseResizableTextFieldOptions {
  minHeight?: number;
  maxHeight?: number;
  resizable?: boolean;
  autoResize?: boolean;
  defaultHeight?: number;
}

export interface UseResizableTextFieldState {
  height: number;
  isResizing: boolean;
  canResize: boolean;
}

export interface UseResizableTextFieldActions {
  setHeight: (height: number) => void;
  startResize: () => void;
  stopResize: () => void;
  autoResizeToContent: (content: string) => void;
  resetHeight: () => void;
}

export interface UseResizableTextFieldReturn {
  state: UseResizableTextFieldState;
  actions: UseResizableTextFieldActions;
  fieldProps: {
    rows: number;
    sx: any;
  };
  resizeHandleProps: {
    onMouseDown: (e: React.MouseEvent) => void;
    style: React.CSSProperties;
  };
}

const ROW_HEIGHT = 24; // Approximate height of one row in pixels

export const useResizableTextField = (
  options: UseResizableTextFieldOptions = {},
  fieldId?: string
): UseResizableTextFieldReturn => {
  const {
    minHeight = 3,
    maxHeight = 20,
    resizable = true,
    autoResize = true,
    defaultHeight = 4
  } = options;

  const [height, setHeightState] = useState(() => {
    // Try to load saved height from localStorage
    if (fieldId && typeof window !== 'undefined') {
      const saved = localStorage.getItem(`resizable-field-${fieldId}`);
      if (saved) {
        const savedHeight = parseInt(saved, 10);
        if (!isNaN(savedHeight) && savedHeight >= minHeight && savedHeight <= maxHeight) {
          return savedHeight;
        }
      }
    }
    return defaultHeight;
  });

  const [isResizing, setIsResizing] = useState(false);
  const resizeStartY = useRef<number>(0);
  const resizeStartHeight = useRef<number>(0);

  // Save height to localStorage when it changes
  useEffect(() => {
    if (fieldId && typeof window !== 'undefined') {
      localStorage.setItem(`resizable-field-${fieldId}`, height.toString());
    }
  }, [fieldId, height]);

  const setHeight = useCallback((newHeight: number) => {
    const clampedHeight = Math.max(minHeight, Math.min(maxHeight, newHeight));
    setHeightState(clampedHeight);
  }, [minHeight, maxHeight]);

  const startResize = useCallback(() => {
    setIsResizing(true);
  }, []);

  const stopResize = useCallback(() => {
    setIsResizing(false);
  }, []);

  const autoResizeToContent = useCallback((content: string) => {
    if (!autoResize) return;

    // Count lines in content
    const lines = content.split('\n').length;
    // Add some extra space for comfortable editing
    const desiredHeight = Math.max(minHeight, Math.min(maxHeight, lines + 1));
    setHeight(desiredHeight);
  }, [autoResize, minHeight, maxHeight, setHeight]);

  const resetHeight = useCallback(() => {
    setHeight(defaultHeight);
  }, [defaultHeight, setHeight]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (!resizable) return;
    
    e.preventDefault();
    e.stopPropagation();
    
    resizeStartY.current = e.clientY;
    resizeStartHeight.current = height;
    startResize();

    const handleMouseMove = (e: MouseEvent) => {
      const deltaY = e.clientY - resizeStartY.current;
      const deltaRows = Math.round(deltaY / ROW_HEIGHT);
      const newHeight = resizeStartHeight.current + deltaRows;
      setHeight(newHeight);
    };

    const handleMouseUp = () => {
      stopResize();
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [resizable, height, startResize, stopResize, setHeight]);

  return {
    state: {
      height,
      isResizing,
      canResize: resizable
    },
    actions: {
      setHeight,
      startResize,
      stopResize,
      autoResizeToContent,
      resetHeight
    },
    fieldProps: {
      rows: height,
      sx: {
        '& .MuiInputBase-root': {
          alignItems: 'flex-start',
          transition: 'height 0.2s ease',
          overflow: 'auto',
          resize: 'none'
        },
        '& .MuiInputBase-input': {
          overflow: 'auto',
          scrollbarWidth: 'thin',
          '&::-webkit-scrollbar': {
            width: '6px'
          },
          '&::-webkit-scrollbar-track': {
            background: 'rgba(0,0,0,0.1)'
          },
          '&::-webkit-scrollbar-thumb': {
            background: 'rgba(0,0,0,0.3)',
            borderRadius: '3px'
          }
        }
      }
    },
    resizeHandleProps: {
      onMouseDown: handleMouseDown,
      style: {
        position: 'absolute' as const,
        bottom: 0,
        left: '50%',
        transform: 'translateX(-50%)',
        width: '60px',
        height: '4px',
        backgroundColor: isResizing ? 'rgba(156, 39, 176, 0.8)' : 'rgba(156, 39, 176, 0.4)',
        borderRadius: '2px',
        cursor: 'ns-resize',
        transition: 'background-color 0.2s ease',
        zIndex: 10
      }
    }
  };
};