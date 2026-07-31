"""
Security middleware for FastAPI application
"""
import time
from typing import Dict, Any
from fastapi import Request, Response
import logging

logger = logging.getLogger(__name__)

# Simple security middleware that always works
class SecurityMiddleware:
    """Custom security middleware with fallback support"""
    
    def __init__(self, app, allowed_origins: list = None):
        self.app = app
        self.allowed_origins = allowed_origins or ["http://localhost:3000"]
    
    async def __call__(self, scope, receive, send):
        """ASGI application interface"""
        if scope["type"] != "http":
            # Pass through non-HTTP requests
            await self.app(scope, receive, send)
            return
        
        # Create a simple request wrapper for HTTP requests
        start_time = time.time()
        
        # Define a custom send wrapper to add headers
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Add security headers to the response
                headers = list(message.get("headers", []))
                
                # Security headers
                headers.extend([
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"x-xss-protection", b"1; mode=block"),
                    (b"referrer-policy", b"strict-origin-when-cross-origin"),
                    (b"permissions-policy", b"geolocation=(), microphone=(), camera=()"),
                ])
                
                # Add process time
                process_time = time.time() - start_time
                headers.append((b"x-process-time", str(process_time).encode()))
                
                # Update message with new headers
                message["headers"] = headers
                
                # Log security-relevant requests
                method = scope.get("method", "")
                path = scope.get("path", "")
                if method in ["POST", "PUT", "DELETE", "PATCH"]:
                    client = scope.get("client", ["unknown", 0])
                    logger.info(
                        f"Security-relevant request: {method} {path} "
                        f"from {client[0]} in {process_time:.4f}s"
                    )
            
            await send(message)
        
        # Call the application with our custom send wrapper
        await self.app(scope, receive, send_wrapper)

# Rate limiter setup
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    
    limiter = Limiter(key_func=get_remote_address)

    def setup_rate_limiting(app):
        """Setup rate limiting for the FastAPI app"""
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        return limiter
        
except ImportError:
    logger.warning("SlowAPI not available - rate limiting disabled")
    limiter = None
    
    def setup_rate_limiting(app):
        """Fallback rate limiting setup"""
        return None