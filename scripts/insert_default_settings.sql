-- Insert default settings for langgraph
INSERT INTO settings (category, settings)
VALUES ('langgraph', '{
    "enabled": false,
    "config": {
        "max_iterations": 10,
        "timeout": 300
    }
}'::jsonb)
ON CONFLICT (category) DO UPDATE
SET settings = EXCLUDED.settings;

-- Insert default settings for endpoint
INSERT INTO settings (category, settings)
VALUES ('endpoint', '{
    "base_url": "http://localhost:8000",
    "timeout": 30,
    "retry_attempts": 3
}'::jsonb)
ON CONFLICT (category) DO UPDATE
SET settings = EXCLUDED.settings; 