CREATE TABLE IF NOT EXISTS settings (
    id SERIAL PRIMARY KEY,
    category VARCHAR(50) NOT NULL UNIQUE, -- e.g., 'llm', 'langchain', 'langgraph', 'endpoint'
    settings JSONB NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
); 