"""
OAuth management endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from app.core.db import get_db
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class OAuthCredentials(BaseModel):
    mcp_server_id: int
    service_name: str
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None

class OAuthResponse(BaseModel):
    id: int
    mcp_server_id: int
    service_name: str
    has_access_token: bool
    has_refresh_token: bool
    client_id: Optional[str]
    created_at: datetime
    updated_at: datetime

@router.get("/credentials/{mcp_server_id}")
async def get_oauth_credentials(
    mcp_server_id: int, 
    show_sensitive: bool = False,
    db: Session = Depends(get_db)
):
    """Get OAuth credentials for an MCP server"""
    from sqlalchemy import text
    
    if show_sensitive:
        # Return full credentials including sensitive data
        query = text("""
        SELECT id, mcp_server_id, service_name, 
               client_id, client_secret, access_token, refresh_token,
               created_at, updated_at
        FROM oauth_credentials
        WHERE mcp_server_id = :mcp_server_id
        """)
        
        result = db.execute(query, {"mcp_server_id": mcp_server_id}).first()
        if not result:
            raise HTTPException(status_code=404, detail="OAuth credentials not found")
        
        return {
            "id": result.id,
            "mcp_server_id": result.mcp_server_id,
            "service_name": result.service_name,
            "client_id": result.client_id,
            "client_secret": result.client_secret,
            "access_token": result.access_token,
            "refresh_token": result.refresh_token,
            "created_at": result.created_at,
            "updated_at": result.updated_at
        }
    else:
        # Return sanitized response
        query = text("""
        SELECT id, mcp_server_id, service_name, 
               access_token IS NOT NULL as has_access_token,
               refresh_token IS NOT NULL as has_refresh_token,
               client_id, created_at, updated_at
        FROM oauth_credentials
        WHERE mcp_server_id = :mcp_server_id
        """)
        
        result = db.execute(query, {"mcp_server_id": mcp_server_id}).first()
        if not result:
            raise HTTPException(status_code=404, detail="OAuth credentials not found")
        
        return OAuthResponse(**dict(result._mapping))

@router.post("/credentials")
async def update_oauth_credentials(creds: OAuthCredentials, db: Session = Depends(get_db)):
    """Update OAuth credentials for an MCP server"""
    from sqlalchemy import text
    
    # Check if credentials exist
    check_query = text("""
    SELECT id FROM oauth_credentials 
    WHERE mcp_server_id = :mcp_server_id AND service_name = :service_name
    """)
    existing = db.execute(check_query, {
        "mcp_server_id": creds.mcp_server_id, 
        "service_name": creds.service_name
    }).first()
    
    if existing:
        # Update existing
        update_query = text("""
        UPDATE oauth_credentials
        SET access_token = :access_token, refresh_token = :refresh_token, 
            client_id = :client_id, client_secret = :client_secret,
            updated_at = CURRENT_TIMESTAMP
        WHERE mcp_server_id = :mcp_server_id AND service_name = :service_name
        RETURNING id
        """)
        result = db.execute(update_query, {
            "access_token": creds.access_token,
            "refresh_token": creds.refresh_token,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "mcp_server_id": creds.mcp_server_id,
            "service_name": creds.service_name
        }).first()
    else:
        # Insert new
        insert_query = text("""
        INSERT INTO oauth_credentials 
        (mcp_server_id, service_name, access_token, refresh_token, client_id, client_secret)
        VALUES (:mcp_server_id, :service_name, :access_token, :refresh_token, :client_id, :client_secret)
        RETURNING id
        """)
        result = db.execute(insert_query, {
            "mcp_server_id": creds.mcp_server_id,
            "service_name": creds.service_name,
            "access_token": creds.access_token,
            "refresh_token": creds.refresh_token,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret
        }).first()
    
    db.commit()
    
    return {"message": "OAuth credentials updated successfully", "id": result[0]}

@router.get("/auth-url/{service_name}")
async def get_oauth_url(service_name: str, client_id: str):
    """Generate OAuth authorization URL"""
    
    if service_name.lower() == "gmail":
        # Gmail OAuth URL
        scopes = "https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/gmail.send"
        redirect_uri = "http://localhost:3000/oauth/callback"  # Frontend callback
        
        auth_url = (
            f"https://accounts.google.com/o/oauth2/v2/auth?"
            f"client_id={client_id}&"
            f"redirect_uri={redirect_uri}&"
            f"response_type=code&"
            f"scope={scopes}&"
            f"access_type=offline&"
            f"prompt=consent"
        )
        
        return {"auth_url": auth_url}
    
    raise HTTPException(status_code=400, detail=f"OAuth not supported for {service_name}")

@router.post("/refresh/{mcp_server_id}")
async def refresh_oauth_token(mcp_server_id: int, db: Session = Depends(get_db)):
    """Refresh OAuth access token for an MCP server"""
    from app.core.oauth_token_manager import oauth_token_manager
    from app.core.db import MCPServer
    
    # Check if using MCPServer oauth_credentials field
    server = db.query(MCPServer).filter(MCPServer.id == mcp_server_id).first()
    if server and server.oauth_credentials:
        # Use the oauth_token_manager for MCPServer-based OAuth
        try:
            # Invalidate cached token to force refresh
            oauth_token_manager.invalidate_token(mcp_server_id, "gmail")
            
            # Get fresh token (will trigger refresh if needed)
            oauth_creds = oauth_token_manager.get_valid_token(
                server_id=mcp_server_id,
                service_name="gmail"
            )
            
            if oauth_creds and oauth_creds.get("access_token"):
                logger.info(f"Successfully refreshed OAuth token for MCP server {mcp_server_id}")
                return {
                    "message": "Token refreshed successfully",
                    "access_token": oauth_creds.get("access_token"),
                    "expires_at": oauth_creds.get("expires_at"),
                    "expires_in": 3600  # Default 1 hour
                }
            else:
                raise HTTPException(status_code=400, detail="Failed to refresh token")
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            raise HTTPException(status_code=500, detail=f"Error refreshing token: {str(e)}")
    
    # Fallback to oauth_credentials table (legacy)
    from sqlalchemy import text
    import requests
    
    # Get existing credentials
    query = text("""
    SELECT id, mcp_server_id, service_name, 
           client_id, client_secret, access_token, refresh_token
    FROM oauth_credentials
    WHERE mcp_server_id = :mcp_server_id
    """)
    
    result = db.execute(query, {"mcp_server_id": mcp_server_id}).first()
    if not result:
        raise HTTPException(status_code=404, detail="OAuth credentials not found")
    
    if not result.refresh_token:
        raise HTTPException(status_code=400, detail="No refresh token available")
    
    if result.service_name.lower() == "gmail":
        # Gmail token refresh
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            "client_id": result.client_id,
            "client_secret": result.client_secret,
            "refresh_token": result.refresh_token,
            "grant_type": "refresh_token"
        }
        
        response = requests.post(token_url, data=data)
        
        if response.status_code == 200:
            tokens = response.json()
            new_access_token = tokens['access_token']
            
            # Update the access token in database
            update_query = text("""
            UPDATE oauth_credentials
            SET access_token = :access_token, updated_at = CURRENT_TIMESTAMP
            WHERE id = :id
            """)
            
            db.execute(update_query, {
                "access_token": new_access_token,
                "id": result.id
            })
            db.commit()
            
            logger.info(f"Successfully refreshed OAuth token for MCP server {mcp_server_id}")
            
            return {
                "message": "Token refreshed successfully",
                "access_token": new_access_token,
                "expires_in": tokens.get('expires_in', 3600)
            }
        else:
            logger.error(f"Failed to refresh token: {response.text}")
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Failed to refresh token: {response.text}"
            )
    
    raise HTTPException(status_code=400, detail=f"Token refresh not supported for {result.service_name}")