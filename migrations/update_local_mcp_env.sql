-- Migration: Update Local MCP Server with environment variables
-- This script updates the Local MCP server with environment variables
-- Note: These are already present in the database, but this script ensures they're properly set

-- First, let's check if the Local MCP server exists
DO $$
BEGIN
    -- Check if Local MCP server exists
    IF EXISTS (SELECT 1 FROM mcp_servers WHERE name = 'Local MCP' AND config_type = 'http') THEN
        -- Update the environment variables for the Local MCP server
        UPDATE mcp_servers 
        SET env = jsonb_build_object(
            'API_KEY', 'ma_enterprise_09463011bf9d4bec929de5be819bd896',
            'JIRA_URL', 'https://lyw.atlassian.net',
            'JIRA_USER', 'kianwoon.wong@beyondsoft.com',
            'JIRA_TOKEN', 'ATATT3xFfGF0TU4_sk47NuWTPbWyDtb9qyI-uiMK5ut0HDZzDec5SHZTZx70VPJpyxg5xgNaIYPZ3WSEdWhoSRtNnMK_tjbjXMw5V8KL4sdMSiO5-NB4yA_sT-TvFzm3vjkjfPWZlzjM4rjXB5gsJXyNsP-bSmiV9WpGzzbGvVLtUcVET8RcLQU=B0827E0B',
            'MS_GRAPH_TOKEN', 'eyJ0eXAiOiJKV1QiLCJub25jZSI6ImQxZll0V0hvRWlGaWVYb3RRZWdTeVFYOFV6b2hncXlwUzJ4ZjU0d2NDbVUiLCJhbGciOiJSUzI1NiIsIng1dCI6IkNOdjBPSTNSd3FsSEZFVm5hb01Bc2hDSDJYRSIsImtpZCI6IkNOdjBPSTNSd3FsSEZFVm5hb01Bc2hDSDJYRSJ9.eyJhdWQiOiJodHRwczovL2dyYXBoLm1pY3Jvc29mdC5jb20iLCJpc3MiOiJodHBwczovL3N0cy53aW5kb3dzLm5ldC9mZGExNWIwMy03ZDBiLTQ2MDQtYjZhMC0wMGEwNzEyYWJjZjUvIiwiaWF0IjoxNzQ2NzA0NDA4LCJuYmYiOjE3NDY3MDQ0MDgsImV4cCI6MTc0NjcwODMwOCwiYWlvIjoiazJSZ1lIQStkRzM5aWM5Q0pRODN6WDIxMzkvVEZ3QT0iLCJhcHBfZGlzcGxheW5hbWUiOiJLbm93bGVkZ2UgQmFzZSBCdWlsZGVyIiwiYXBwaWQiOiJhNGIxMWEzOS1lZTllLTQyYjYtYWIzMC03ODhjY2VmMTRkODkiLCJhcHBpZGFjciI6IjEiLCJpZHAiOiJodHRwczovL3N0cy53aW5kb3dzLm5ldC9mZGExNWIwMy03ZDBiLTQ2MDQtYjZhMC0wMGEwNzEyYWJjZjUvIiwiaWR0eXAiOiJhcHAiLCJvaWQiOiIyMWYxNjQ2Mi01MDI2LTQ0NDAtYWQ0Ny04NzY0NGJmZjgzZDIiLCJyaCI6IjEuQVhFQUExdWhfUXQ5QkVhMm9BQ2djU3E4OVFNQUFBQUFBQUFBd0FBQUFBQUFBQUNIQUFCeEFBLiIsInN1YiI6IjIxZjE2NDYyLTUwMjYtNDQ0MC1hZDQ3LTg3NjQ0YmZmODNkMiIsInRlbmFudF9yZWdpb25fc2NvcGUiOiJBUyIsInRpZCI6ImZkYTE1YjAzLTdkMGItNDYwNC1iNmEwLTAwYTA3MTJhYmNmNSIsInV0aSI6IjdNMTM5bUVENVVPSGZjNjJpNGhJQUEiLCJ2ZXIiOiIxLjAiLCJ3aWRzIjpbIjA5OTdhMWQwLTBkMWQtNGFjYi1iNDA4LWQ1Y2E3MzEyMWU5MCJdLCJ4bXNfaWRyZWwiOiI3IDE0IiwieG1zX3RjZHQiOjE2NTM5NjQzMDJ9.LVBvjO8wSHHHEG9FUL4bN57vCMp3vWEh2UrW50zz18EL000Y8RLsZ11djm4a8m4iyY8HaWk8Vp-jQ8MvNS0m_FEkMMjclJLK4CsCLT1JZWzOgzH_j0geOUybCEslXNlTDhJfa9XQPbLMd-8CUTFWOwDKb_BinTgZkuDhRQi16OmAtSe5ct565UgGSMhDIeo3Vlvu03iGMPLDrAxmc2oHIMWyhLkVNfc1kiyxf1bAnpccUVzb5QtGR9JhQoKLRLjJbSwgERz-usfwA_f1El0LwpFrt83MqpXE3tn_60_Ppxgc0AfH5bIDYJR3hOwBL1QoQK8uQgDHFFJjoh23EIS_3w',
            'GOOGLE_SEARCH_API_KEY', 'REPLACE_WITH_YOUR_API_KEY',
            'GOOGLE_SEARCH_ENGINE_ID', 'REPLACE_WITH_YOUR_SEARCH_ENGINE_ID',
            'GOOGLE_SEARCH_DEFAULT_RESULT', '15',
            'CORS_ALLOWED_ORIGINS', 'http://localhost:3000,http://localhost:3001,http://localhost:5173,http://localhost:8000'
        ),
        updated_at = NOW()
        WHERE name = 'Local MCP' AND config_type = 'http';
        
        RAISE NOTICE 'Updated environment variables for Local MCP server';
    ELSE
        RAISE NOTICE 'Local MCP server not found. Please ensure it exists first.';
    END IF;
END $$;

-- Add a comment to document the purpose of this configuration
COMMENT ON COLUMN mcp_servers.env IS 'Environment variables for MCP servers. Contains sensitive API keys and tokens that should be masked in UI.';