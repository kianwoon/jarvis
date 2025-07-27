import React from 'react'
import ReactDOM from 'react-dom/client'
import KnowledgeGraphViewer from './components/KnowledgeGraphViewer'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <KnowledgeGraphViewer />
  </React.StrictMode>,
)