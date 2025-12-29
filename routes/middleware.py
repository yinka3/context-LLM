from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import is_configured

ALLOWED_PREFIXES = (
    "/setup",
    "/static",
    "/docs",
    "/openapi.json",
)

ALLOWED_EXACT = (
    "/",
)

class SetupGuardMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        
        if path in ALLOWED_EXACT or path.startswith(ALLOWED_PREFIXES):
            return await call_next(request)
        
        if not is_configured():
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "Vestige is not configured. Please complete setup first.",
                    "setup_url": "/setup"
                }
            )
        
        return await call_next(request)