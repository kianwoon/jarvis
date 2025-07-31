import React from 'react'
import ReactDOM from 'react-dom/client'
import { SnackbarProvider } from 'notistack'
import KnowledgeGraphViewer from './components/KnowledgeGraphViewer'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <SnackbarProvider maxSnack={3}>
      <KnowledgeGraphViewer />
    </SnackbarProvider>
  </React.StrictMode>,
)
