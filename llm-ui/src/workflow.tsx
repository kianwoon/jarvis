import React from 'react'
import ReactDOM from 'react-dom/client'
import WorkflowApp from './WorkflowApp'
import './index.css'
import './styles/workflow-animations.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <WorkflowApp />
  </React.StrictMode>,
)