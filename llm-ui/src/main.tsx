import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

console.log('🚀 MAIN.TSX LOADING...');

const rootElement = document.getElementById('root');
console.log('📍 ROOT ELEMENT:', rootElement);

if (rootElement) {
  console.log('✅ ROOT FOUND, RENDERING REACT APP...');
  ReactDOM.createRoot(rootElement).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
} else {
  console.error('❌ ROOT ELEMENT NOT FOUND!');
}