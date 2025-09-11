# app/main.py
"""
FastAPI application entry point for AI Assistant Platform.
Configures routes, middleware, and application lifecycle.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.staticfiles import StaticFiles

from app.config import settings

# -----------------------------------------------------------------------------
# Existing routers
from app.api.endpoints import rag as rag_router
from app.api.endpoints import files as files_router
from app.api.endpoints import agent as agent_router
from app.api.endpoints import agent_actions as agent_actions_router  # type: ignore

# OAuth routes
from app.api.endpoints import google_oauth as google_oauth_router  # type: ignore

# Utilities for auth-status helper
from app.services.token_store import TokenStore
from app.services.session import get_current_user_id

# NEW: expose tool list (debug) from the same module the agent uses
from app.tools.real_tools import get_all_tools

# -----------------------------------------------------------------------------


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    # Initialize DB here (not at import time)
    try:
        # If you have an init_db() function, import/use it here.
        init_db()  # noqa: F821  # allowed to fail if not defined
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"DB init failed: {e}")
        # continue even if DB init fails
    try:
        yield
    finally:
        logger.info("Application shutdown complete")


# FastAPI application instance
app = FastAPI(
    title=settings.app_name,
    description="AI Assistant for Enterprise with RAG, LangChain, and multi-modal processing",
    version=settings.app_version,
    debug=settings.debug,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# --- STATIC SITE MOUNT (absolute path, single mount) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_ROOT / "web"

if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    logging.getLogger(__name__).info("Mounted /web from %s", WEB_DIR)
else:
    logging.getLogger(__name__).warning("Static 'web' folder NOT found at %s", WEB_DIR)

# --- Routers ---
# These three end up under /api/v1/...
app.include_router(rag_router.router, prefix=settings.api_v1_prefix)
app.include_router(files_router.router, prefix=settings.api_v1_prefix)
app.include_router(agent_router.router, prefix=settings.api_v1_prefix)  # -> /api/v1/agent/*

# OAuth + agent actions keep their own prefixes defined in the modules
app.include_router(google_oauth_router.router)     # -> /auth/google/*
app.include_router(agent_actions_router.router)    # -> /api/agent/*

# --- Trusted hosts ---
_clean_hosts = [h for h in settings.allowed_hosts if h and h != "*"]
if _clean_hosts:
    if "testserver" not in _clean_hosts:
        _clean_hosts.append("testserver")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_clean_hosts)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Health + utility endpoints

@app.get("/")
async def root():
    """Root endpoint - basic health check."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "status": "healthy",
    }


@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "debug": settings.debug,
    }


@app.get(settings.api_v1_prefix)
async def api_v1_root():
    """API v1 root endpoint."""
    return {
        "message": "AI Assistant Platform API v1",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "api": settings.api_v1_prefix,
        },
    }

# Used by the frontend to check if the user has connected Google yet
@app.get("/api/me/google-auth-status")
def google_auth_status(request: Request):
    user_id = get_current_user_id(request)
    tok = TokenStore.get(user_id) if user_id else None
    return {"authed": bool(tok), "email": (tok or {}).get("email")}



@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "status_code": 404},
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500},
    )

# -----------------------------------------------------------------------------
# Development server runner

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info",
    )
