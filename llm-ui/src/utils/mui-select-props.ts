import { SelectProps } from '@mui/material';

// Common MenuProps for Select components to fix dropdown issues in accordions
// The key is to NOT use disablePortal when in ReactFlow nodes
export const selectMenuProps: SelectProps['MenuProps'] = {
  // Keep portal enabled to render outside ReactFlow's stacking context
  disablePortal: false,
  // Ensure the dropdown appears with very high z-index
  PaperProps: {
    sx: {
      zIndex: 10000, // Very high z-index to appear above ReactFlow
    },
  },
  // Position settings
  anchorOrigin: {
    vertical: 'bottom',
    horizontal: 'left',
  },
  transformOrigin: {
    vertical: 'top',
    horizontal: 'left',
  },
  // Keep menu in DOM for better performance
  keepMounted: false,
};

// Event handlers to prevent accordion from expanding/collapsing when using Select
export const selectEventHandlers = {
  onClick: (e: React.MouseEvent) => {
    e.stopPropagation();
  },
  onMouseDown: (e: React.MouseEvent) => {
    e.stopPropagation();
  },
  onFocus: (e: React.FocusEvent) => {
    e.stopPropagation();
  },
  onKeyDown: (e: React.KeyboardEvent) => {
    e.stopPropagation();
  },
};

// Container div style with proper positioning and z-index
export const selectContainerStyle: React.CSSProperties = {
  position: 'relative',
  zIndex: 1000,
};