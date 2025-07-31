import React from 'react'
import ReactDOM from 'react-dom/client'
import { SnackbarProvider } from 'notistack'
import MultiAgentApp from './MultiAgentApp.tsx'
import './App.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <SnackbarProvider maxSnack={3}>
      <MultiAgentApp />
    </SnackbarProvider>
  </React.StrictMode>,
)
