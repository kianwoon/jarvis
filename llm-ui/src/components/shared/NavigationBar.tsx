import React from 'react';
import {
  Box,
  Tabs,
  Tab
} from '@mui/material';

interface NavigationBarProps {
  currentTab: number;
}

const NavigationBar: React.FC<NavigationBarProps> = ({ currentTab }) => {
  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    // Navigate to the appropriate page based on tab selection
    switch (newValue) {
      case 0:
        if (currentTab !== 0) {
          window.location.href = '/';
        }
        break;
      case 1:
        if (currentTab !== 1) {
          window.location.href = '/multi-agent.html';
        }
        break;
      case 2:
        if (currentTab !== 2) {
          window.location.href = '/workflow.html';
        }
        break;
      case 3:
        if (currentTab !== 3) {
          window.location.href = '/meta-task.html';
        }
        break;
      case 4:
        if (currentTab !== 4) {
          window.location.href = '/settings.html';
        }
        break;
      case 5:
        if (currentTab !== 5) {
          window.location.href = '/knowledge-graph.html';
        }
        break;
      case 6:
        if (currentTab !== 6) {
          window.location.href = '/idc.html';
        }
        break;
    }
  };

  return (
    <Box sx={{ borderBottom: 1, borderColor: 'divider', bgcolor: 'background.paper' }}>
      <Tabs 
        value={currentTab}
        onChange={handleTabChange} 
        aria-label="jarvis modes"
        centered
        sx={{
          '& .MuiTab-root': {
            fontSize: '1rem',
            fontWeight: 600,
            textTransform: 'none',
            minWidth: 120,
            padding: '12px 24px',
            '&.Mui-selected': {
              color: 'primary.main',
              fontWeight: 700
            }
          }
        }}
      >
        <Tab 
          label="Standard Chat" 
          id="tab-0"
          aria-controls="tabpanel-0"
        />
        <Tab 
          label="Multi-Agent" 
          id="tab-1"
          aria-controls="tabpanel-1"
        />
        <Tab 
          label="Workflow" 
          id="tab-2"
          aria-controls="tabpanel-2"
        />
        <Tab 
          label="Meta-Tasks" 
          id="tab-3"
          aria-controls="tabpanel-3"
        />
        <Tab 
          label="Settings" 
          id="tab-4"
          aria-controls="tabpanel-4"
        />
        <Tab 
          label="Knowledge Graph" 
          id="tab-5"
          aria-controls="tabpanel-5"
        />
        <Tab 
          label="IDC" 
          id="tab-6"
          aria-controls="tabpanel-6"
        />
      </Tabs>
    </Box>
  );
};

export default NavigationBar;