import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Box, Alert, AlertTitle, Button } from '@mui/material';
import { Refresh as RefreshIcon } from '@mui/icons-material';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  };

  public static getDerivedStateFromError(error: Error): State {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({
      error,
      errorInfo
    });
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  public render() {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default error UI
      return (
        <Box 
          sx={{ 
            p: 3, 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center',
            minHeight: '300px'
          }}
        >
          <Alert 
            severity="error" 
            sx={{ maxWidth: 600 }}
            action={
              <Button
                color="inherit"
                size="small"
                onClick={this.handleReset}
                startIcon={<RefreshIcon />}
              >
                Try Again
              </Button>
            }
          >
            <AlertTitle>Something went wrong</AlertTitle>
            An error occurred while loading this component. Please try refreshing the page or contact support if the problem persists.
            
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <Box sx={{ mt: 2, fontFamily: 'monospace', fontSize: '0.8em' }}>
                <strong>Error:</strong> {this.state.error.message}
              </Box>
            )}
          </Alert>
        </Box>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;