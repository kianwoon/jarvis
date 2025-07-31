import React from 'react'
import ReactDOM from 'react-dom/client'
import { SnackbarProvider } from 'notistack'
import WorkflowApp from './WorkflowApp'
import './index.css'
import './styles/workflow-animations.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <SnackbarProvider maxSnack={3}>
      <WorkflowApp />
    </SnackbarProvider>
  </React.StrictMode>,
)
