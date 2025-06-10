"""
OAuth Token Manager with Redis caching and automatic refresh
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import requests

from app.core.redis_base import RedisCache
from app.core.db import SessionLocal
from sqlalchemy import text

logger = logging.getLogger(__name__)


class OAuthTokenManager(RedisCache):
    """Manages OAuth tokens with Redis caching and automatic refresh"""
    
    def __init__(self):
        super().__init__(key_prefix="oauth_tokens:")
        self.refresh_threshold_minutes = 10  # Refresh if token expires in less than 10 minutes
        
    def _get_cache_key(self, server_id: int, service_name: str) -> str:
        """Get Redis cache key for OAuth tokens"""
        return f"{service_name}:{server_id}"
    
    def get_valid_token(self, server_id: int, service_name: str = "gmail") -> Optional[Dict[str, Any]]:
        """
        Get a valid OAuth token, refreshing if necessary
        
        Returns:
            Dict with access_token and other OAuth credentials, or None if failed
        """
        cache_key = self._get_cache_key(server_id, service_name)
        
        # Try to get from Redis cache first
        cached_data = self.get(cache_key)
        if cached_data:
            # Check if token is still valid
            expires_at = cached_data.get("expires_at")
            if expires_at:
                expires_at_dt = datetime.fromisoformat(expires_at)
                time_until_expiry = expires_at_dt - datetime.utcnow()
                
                if time_until_expiry > timedelta(minutes=self.refresh_threshold_minutes):
                    logger.debug(f"Using cached token for {service_name}, expires in {time_until_expiry}")
                    return cached_data
                else:
                    logger.info(f"Token for {service_name} expires in {time_until_expiry}, refreshing...")
        
        # Get credentials from database - use raw SQL for oauth_credentials table
        db = SessionLocal()
        try:
            # Query the oauth_credentials table directly
            result = db.execute(text("""
                SELECT client_id, client_secret, access_token, refresh_token, token_expiry
                FROM oauth_credentials
                WHERE mcp_server_id = :server_id AND service_name = :service_name
                ORDER BY updated_at DESC
                LIMIT 1
            """), {"server_id": server_id, "service_name": service_name}).first()
            
            if not result:
                logger.error(f"No OAuth credentials found for server {server_id}, service {service_name}")
                return None
            
            oauth_creds = {
                "client_id": result.client_id,
                "client_secret": result.client_secret,
                "access_token": result.access_token,
                "refresh_token": result.refresh_token,
                "expires_at": result.token_expiry.isoformat() if result.token_expiry else None
            }
            
            # Check if we need to refresh
            if self._should_refresh_token(oauth_creds):
                # Refresh the token
                new_creds = self._refresh_oauth_token(
                    client_id=oauth_creds.get("client_id"),
                    client_secret=oauth_creds.get("client_secret"),
                    refresh_token=oauth_creds.get("refresh_token"),
                    service_name=service_name
                )
                
                if new_creds:
                    # Update database
                    db.execute(text("""
                        UPDATE oauth_credentials
                        SET access_token = :access_token,
                            token_expiry = NOW() + make_interval(secs => :expires_in),
                            updated_at = NOW()
                        WHERE mcp_server_id = :server_id AND service_name = :service_name
                    """), {
                        "access_token": new_creds["access_token"],
                        "expires_in": new_creds.get("expires_in", 3600),
                        "server_id": server_id,
                        "service_name": service_name
                    })
                    db.commit()
                    
                    # Update oauth_creds with new token
                    oauth_creds["access_token"] = new_creds["access_token"]
                    oauth_creds["expires_at"] = new_creds["expires_at"]
                    
                    # Cache in Redis with expiration
                    self._cache_token(cache_key, oauth_creds, new_creds.get("expires_in", 3600))
                    
                    logger.info(f"Successfully refreshed and cached {service_name} token")
                    return oauth_creds
                else:
                    logger.error(f"Failed to refresh {service_name} token")
                    return oauth_creds  # Return existing creds as fallback
            else:
                # Token is still valid, cache it
                self._cache_token(cache_key, oauth_creds, 3600)  # Default 1 hour
                return oauth_creds
                
        except Exception as e:
            logger.error(f"Error managing OAuth token: {e}")
            return None
        finally:
            db.close()
    
    def _should_refresh_token(self, oauth_creds: Dict[str, Any]) -> bool:
        """Check if token should be refreshed based on last update time"""
        # Don't refresh if we don't have a refresh token
        if not oauth_creds.get("refresh_token"):
            return False
            
        # Check if we have token expiry information
        expires_at = oauth_creds.get("expires_at")
        if expires_at:
            try:
                expires_at_dt = datetime.fromisoformat(expires_at)
                time_until_expiry = expires_at_dt - datetime.utcnow()
                # Refresh if less than threshold
                return time_until_expiry < timedelta(minutes=self.refresh_threshold_minutes)
            except:
                pass
                
        # If no expiry info, don't refresh (assume token is valid)
        return False
    
    def _refresh_oauth_token(self, client_id: str, client_secret: str, 
                           refresh_token: str, service_name: str) -> Optional[Dict[str, Any]]:
        """
        Refresh OAuth token using refresh token
        
        Returns:
            Dict with new access_token and expires_in, or None if failed
        """
        if service_name == "gmail":
            token_url = "https://oauth2.googleapis.com/token"
        else:
            # Add other services as needed
            logger.error(f"Unknown service: {service_name}")
            return None
        
        try:
            response = requests.post(
                token_url,
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token"
                }
            )
            
            if response.ok:
                token_data = response.json()
                
                # Calculate expiration time
                expires_in = token_data.get("expires_in", 3600)
                expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
                
                return {
                    "access_token": token_data.get("access_token"),
                    "expires_in": expires_in,
                    "expires_at": expires_at.isoformat(),
                    "token_type": token_data.get("token_type", "Bearer"),
                    "scope": token_data.get("scope"),
                    "token_uri": token_url  # Include token_uri for Gmail MCP server
                }
            else:
                logger.error(f"Token refresh failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            return None
    
    def _cache_token(self, cache_key: str, oauth_creds: Dict[str, Any], expires_in: int):
        """Cache token in Redis with appropriate expiration"""
        # Store token with some buffer before actual expiration
        cache_duration = max(expires_in - 300, 60)  # At least 1 minute, but 5 minutes less than actual expiry
        
        cache_data = {
            "access_token": oauth_creds.get("access_token"),
            "refresh_token": oauth_creds.get("refresh_token"),
            "client_id": oauth_creds.get("client_id"),
            "client_secret": oauth_creds.get("client_secret"),
            "expires_at": oauth_creds.get("expires_at"),
            "token_uri": oauth_creds.get("token_uri", "https://oauth2.googleapis.com/token"),
            "cached_at": datetime.utcnow().isoformat()
        }
        
        self.set(cache_key, cache_data, expire=cache_duration)
    
    def invalidate_token(self, server_id: int, service_name: str = "gmail"):
        """Invalidate cached token (useful when token is revoked)"""
        cache_key = self._get_cache_key(server_id, service_name)
        self.delete(cache_key)
        logger.info(f"Invalidated cached token for {service_name} server {server_id}")


# Global instance
oauth_token_manager = OAuthTokenManager()