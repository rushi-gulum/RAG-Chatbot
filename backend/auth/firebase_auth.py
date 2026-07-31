"""
Firebase Authentication Integration for Backend
"""
import os
import json
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
import dotenv
dotenv.load_dotenv()  # Load environment variables
# Import Firebase Admin (gracefully handle if not installed)
try:
    import firebase_admin
    from firebase_admin import credentials, auth
    firebase_available = True
except ImportError:
    firebase_available = False
    logging.warning("Firebase Admin SDK not installed. Authentication will be disabled.")

logger = logging.getLogger(__name__)

class FirebaseAuth:
    """Firebase Authentication Handler"""
    
    def __init__(self):
        self.app = None
        self.enabled = False
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        if not firebase_available:
            logger.warning("Firebase Admin SDK not available")
            return
        
        try:
            # Check if already initialized
            if firebase_admin._apps:
                self.app = firebase_admin.get_app()
                self.enabled = True
                logger.info("Using existing Firebase app")
                return
            
            # Get service account key path from environment
            service_key_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY")
            
            if service_key_path and os.path.exists(service_key_path):
                # Initialize with service account file
                cred = credentials.Certificate(service_key_path)
                self.app = firebase_admin.initialize_app(cred)
                self.enabled = True
                logger.info("Firebase initialized with service account file")
            else:
                # Try to initialize with environment variables
                firebase_config = {
                    "type": "service_account",
                    "project_id": os.getenv("FIREBASE_PROJECT_ID"),
                    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
                    "private_key": os.getenv("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n"),
                    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
                    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL")
                }
                
                # Check if we have the minimum required config
                if all([firebase_config["project_id"], firebase_config["private_key"], firebase_config["client_email"]]):
                    cred = credentials.Certificate(firebase_config)
                    self.app = firebase_admin.initialize_app(cred)
                    self.enabled = True
                    logger.info("Firebase initialized with environment variables")
                else:
                    logger.warning("Firebase configuration not found. Authentication will be disabled.")
                    
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {str(e)}")
            self.enabled = False
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify Firebase ID token and return user information
        
        Args:
            token: Firebase ID token
            
        Returns:
            User information if valid, None if invalid
        """
        if not self.enabled:
            logger.warning("Firebase not enabled - skipping token verification")
            return None
        
        try:
            # Verify the token
            decoded_token = auth.verify_id_token(token)
            
            user_info = {
                "uid": decoded_token["uid"],
                "email": decoded_token.get("email"),
                "email_verified": decoded_token.get("email_verified", False),
                "name": decoded_token.get("name"),
                "picture": decoded_token.get("picture"),
                "auth_time": decoded_token.get("auth_time"),
                "exp": decoded_token.get("exp")
            }
            
            return user_info
            
        except auth.InvalidIdTokenError:
            logger.warning("Invalid Firebase ID token")
            return None
        except auth.ExpiredIdTokenError:
            logger.warning("Expired Firebase ID token")
            return None
        except Exception as e:
            logger.error(f"Error verifying Firebase token: {str(e)}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get authentication system status"""
        return {
            "firebase_available": firebase_available,
            "firebase_enabled": self.enabled,
            "app_name": self.app.name if self.app else None
        }

# Global Firebase Auth instance
firebase_auth = FirebaseAuth()

# Security scheme for FastAPI
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict[str, Any]]:
    """
    FastAPI dependency to get current authenticated user
    
    Args:
        credentials: HTTP Bearer credentials
        
    Returns:
        User information if authenticated, None otherwise
    """
    if not credentials:
        return None
    
    # Extract token from Bearer token
    token = credentials.credentials
    
    # Verify token with Firebase
    user_info = firebase_auth.verify_token(token)
    
    return user_info

async def require_auth(user: Optional[Dict[str, Any]] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    FastAPI dependency that requires authentication
    
    Args:
        user: Current user from get_current_user dependency
        
    Returns:
        User information if authenticated
        
    Raises:
        HTTPException: If user is not authenticated
    """
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return user

async def optional_auth(user: Optional[Dict[str, Any]] = Depends(get_current_user)) -> Optional[Dict[str, Any]]:
    """
    FastAPI dependency for optional authentication
    
    Args:
        user: Current user from get_current_user dependency
        
    Returns:
        User information if authenticated, None otherwise
    """
    return user

def get_user_id(user: Dict[str, Any]) -> str:
    """Extract user ID from user info"""
    return user.get("uid", "")

def get_user_email(user: Dict[str, Any]) -> str:
    """Extract user email from user info"""
    return user.get("email", "")