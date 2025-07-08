import { useState, useCallback } from 'react';

export interface UseResizableTextFieldOptions {
  minHeight?: number;
  maxHeight?: number;
  initialHeight?: number;
}

export const useResizableTextField = (options: UseResizableTextFieldOptions = {}) => {
  const {
    minHeight = 40,
    maxHeight = 200,
    initialHeight = 40
  } = options;

  const [height, setHeight] = useState(initialHeight);

  const handleResize = useCallback((event: React.ChangeEvent<HTMLTextAreaElement>) => {
    const textarea = event.target;
    textarea.style.height = 'auto';
    const scrollHeight = textarea.scrollHeight;
    const newHeight = Math.min(Math.max(scrollHeight, minHeight), maxHeight);
    setHeight(newHeight);
    textarea.style.height = `${newHeight}px`;
  }, [minHeight, maxHeight]);

  const resetHeight = useCallback(() => {
    setHeight(initialHeight);
  }, [initialHeight]);

  return {
    height,
    handleResize,
    resetHeight,
    textAreaProps: {
      style: { height: `${height}px`, resize: 'none' },
      onChange: handleResize
    }
  };
};