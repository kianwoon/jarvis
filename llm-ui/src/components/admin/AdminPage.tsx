import React, { useEffect, useState, useMemo } from 'react';
import { Box, ThemeProvider, createTheme } from '@mui/material';
import CssBaseline from '@mui/material/CssBaseline';
import NavigationBar from '../shared/NavigationBar';
import DocumentAdminPage from './DocumentAdminPage';

const AdminPage: React.FC = () => {
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    // Load dark mode preference from localStorage
    const savedDarkMode = localStorage.getItem('jarvis-dark-mode');
    if (savedDarkMode) {
      const isDark = JSON.parse(savedDarkMode);
      setDarkMode(isDark);
      document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
      document.body.setAttribute('data-theme', isDark ? 'dark' : 'light');
    }
  }, []);

  // Listen for theme changes from localStorage
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'jarvis-dark-mode' && e.newValue) {
        const newDarkMode = JSON.parse(e.newValue);
        setDarkMode(newDarkMode);
        document.body.setAttribute('data-theme', newDarkMode ? 'dark' : 'light');
        document.documentElement.setAttribute('data-theme', newDarkMode ? 'dark' : 'light');
      }
    };
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  const theme = useMemo(() => createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#2196f3',
      },
      secondary: {
        main: '#f50057',
      },
      background: {
        default: darkMode ? '#121212' : '#f5f5f5',
        paper: darkMode ? '#1e1e1e' : '#ffffff',
      },
    },
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          body: {
            margin: 0,
            padding: 0,
            height: '100vh',
            overflow: 'hidden',
            backgroundColor: darkMode ? '#121212' : '#f5f5f5',
          },
          '#root': {
            height: '100vh',
            display: 'flex',
            flexDirection: 'column',
            backgroundColor: darkMode ? '#121212' : '#f5f5f5',
          },
        },
      },
    },
  }), [darkMode]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
        {/* Navigation Bar with tab 8 (Admin) active */}
        <NavigationBar currentTab={8} />
        
        {/* Main Admin Content */}
        <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', bgcolor: 'background.default' }}>
          <DocumentAdminPage />
        </Box>
      </Box>
    </ThemeProvider>
  );
};

export default AdminPage;