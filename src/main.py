"""FastAPI application for DataSentience data center optimization platform.

Provides RESTful API endpoints for predictive maintenance analysis using
NVIDIA NIM models deployed on AWS SageMaker. Supports health checks, query
processing, and multi-agent orchestration for data center telemetry analysis.
"""

import logging
import time
import asyncio
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.agent import agent, InputValidationError
from src.indexer import index_all_data
from src.config import config
from src.startup_checks import validate_models
from src.multi_agent_system import MultiAgentOrchestrator

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Prometheus metrics
nim_requests_total = Counter(
    "nim_requests_total",
    "Total number of NIM API requests",
    ["model", "status"]
)

nim_latency_seconds = Histogram(
    "nim_latency_seconds",
    "NIM API request latency in seconds",
    ["model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

nim_tokens_total = Counter(
    "nim_tokens_total",
    "Total number of tokens processed",
    ["model", "type"]
)

query_requests_total = Counter(
    "query_requests_total",
    "Total number of query requests",
    ["endpoint", "status"]
)

# Initialize FastAPI application
app = FastAPI(
    title="DataSentience API",
    version="1.0.0",
    description="AI-powered data center optimization platform"
)

# Initialize rate limiter
if config.REDIS_URL:
    limiter = Limiter(
        key_func=get_remote_address,
        storage_uri=config.REDIS_URL,
        strategy="fixed-window-elastic-expiry"
    )
else:
    limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize multi-agent orchestrator (lazy-loaded in startup_event)
app.state.orchestrator = None

# CORS configuration
cors_origins = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",  # Alternative dev port
    "http://localhost:8080",  # Alternative dev port
    "http://localhost",       # Docker frontend
    config.FRONTEND_URL       # Production frontend
]

# Remove empty strings and add wildcard for local development
if config.is_local():
    cors_origins = ["*"]
else:
    cors_origins = [origin for origin in cors_origins if origin]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup with non-blocking pattern.

    Uses asyncio.to_thread() to move sync operations off the event loop,
    ensuring /ping endpoint responds immediately for SageMaker health checks.
    """
    import sys
    import os

    logger.info("=" * 80)
    logger.info("STARTUP: DataSentience Application Starting")
    logger.info("=" * 80)

    # Environment diagnostics
    logger.info("STARTUP: Python version: %s", sys.version)
    logger.info("STARTUP: Platform: %s", sys.platform)
    logger.info("STARTUP: Working directory: %s", os.getcwd())
    logger.info("STARTUP: Environment variables:")
    logger.info("  - NVIDIA_API_KEY: %s", "SET" if config.NVIDIA_API_KEY else "NOT SET")
    logger.info("  - NVIDIA_API_URL: %s", config.NVIDIA_API_URL or "NOT SET")
    logger.info("  - MODEL_NAME: %s", config.MODEL_NAME or "NOT SET")
    logger.info("  - LOG_LEVEL: %s", config.LOG_LEVEL)

    # Check orchestrator initialization
    logger.info("STARTUP: Multi-agent orchestrator initialized: %s", app.state.orchestrator is not None)

    # State for readiness probe
    app.state.is_ready = False

    async def initialize_app():
        """Run startup tasks in the background using asyncio.to_thread().

        This ensures each heavy sync operation runs in a thread pool,
        preventing event loop blocking that causes health check timeouts.
        """
        try:
            # Use asyncio.to_thread() for sync operations to prevent blocking
            logger.info("STARTUP: Validating NVIDIA models...")
            await asyncio.to_thread(validate_models)
            logger.info("STARTUP: Model validation successful")

            logger.info("STARTUP: Loading secrets...")
            await asyncio.to_thread(config.load_secrets)
            logger.info("STARTUP: Secrets loaded.")

            logger.info("STARTUP: Validating configuration...")
            await asyncio.to_thread(config.validate)
            logger.info("STARTUP: Configuration validation successful")

            # Index vector store (required for agent queries)
            logger.info("STARTUP: Indexing data into vector store...")
            await asyncio.to_thread(index_all_data)
            logger.info("STARTUP: Vector store indexing complete")

            # Initialize multi-agent orchestrator
            logger.info("STARTUP: Initializing multi-agent orchestrator...")
            app.state.orchestrator = MultiAgentOrchestrator()
            logger.info("STARTUP: Multi-agent orchestrator initialized.")

            app.state.is_ready = True
            logger.info("=" * 80)
            logger.info("STARTUP: Application startup complete - READY")
            logger.info("=" * 80)

        except Exception as error:
            logger.error("=" * 80)
            logger.error("STARTUP: Background initialization failed - %s", str(error))
            logger.error("STARTUP: Exception type: %s", type(error).__name__)
            import traceback
            logger.error("STARTUP: Traceback:\n%s", traceback.format_exc())
            logger.error("=" * 80)
            # The readiness probe will fail but /ping still works

    # Schedule initialization and return immediately - /ping available now
    asyncio.create_task(initialize_app())


@app.exception_handler(InputValidationError)
async def validation_exception_handler(request, exc):
    """Handle input validation errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for agent queries."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Question about data center operations"
    )
    agent_type: Optional[str] = Field(
        None,
        description="Agent routing: 'orchestrator', 'agent1', or None for legacy"
    )
    question_type: Optional[str] = Field(
        None,
        description="Question type for specialized data payloads"
    )

    @validator('question')
    def question_not_empty(cls, value):
        """Validate question is not empty or whitespace."""
        if not value.strip():
            raise ValueError("Question cannot be empty")
        return value.strip()


class QueryResponse(BaseModel):
    """Response model for agent queries."""
    answer: str
    chart_data: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str
    nvidia_configured: bool
    mode: str


@app.get("/ping")
def ping():
    """SageMaker health check endpoint.

    Returns empty 200 response immediately. Must respond in <2 seconds
    for SageMaker health check to pass. This allows background initialization
    to complete without blocking health checks.
    """
    return Response(status_code=status.HTTP_200_OK)


@app.get("/readyz")
def readyz(response: Response):
    """Readiness probe for SageMaker.

    Returns 200 if initialization complete, 503 if still starting.
    """
    if app.state.is_ready:
        return {"status": "ready"}
    else:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready"}


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint for monitoring and load balancers."""
    return HealthResponse(
        status="healthy",
        nvidia_configured=bool(config.NVIDIA_API_KEY),
        mode="nvidia_api" if config.NVIDIA_API_KEY else "local_analysis"
    )


@app.post("/invocations", response_model=QueryResponse)
@limiter.limit("100/hour")
async def invocations_endpoint(request: Request, query_request: QueryRequest):
    """SageMaker invocations endpoint with readiness protection.

    Checks readiness state before processing queries to prevent serving
    requests before initialization completes (vector store indexing, etc.).
    """
    # Check readiness to prevent serving before initialization complete
    if not app.state.is_ready:
        return Response(
            content='{"detail":"Service not ready. Background initialization is in progress or has failed."}',
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            media_type="application/json",
            headers={"Retry-After": "15"},
        )

    # Validate origin header for production security
    if not config.is_local():
        origin = request.headers.get("origin")
        allowed_origins = [config.FRONTEND_URL, "http://datasentience-showcase.s3-website-us-east-1.amazonaws.com", "http://datasentience-final-1762168304.s3-website-us-east-1.amazonaws.com"]
        if origin and origin not in allowed_origins:
            logger.warning("Unauthorized origin access attempt: %s", origin)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: invalid origin"
            )

    start_time = time.time()
    status_label = "success"

    try:
        # Fast path: Use single agent for agent1 queries (bypass slow orchestrator)
        if query_request.agent_type == 'agent1':
            logger.info("Using fast path: single agent for question: {}...".format(query_request.question[:50]))
            # Run synchronous agent.query in thread pool to avoid blocking
            result = await asyncio.to_thread(agent.query, query_request.question)
            
            # Track metrics
            latency = time.time() - start_time
            nim_latency_seconds.labels(model=config.REASONING_MODEL).observe(latency)
            query_requests_total.labels(endpoint="invocations", status="success").inc()
            
            # Handle both dict and string responses
            if isinstance(result, dict):
                return QueryResponse(
                    answer=result.get("answer", ""),
                    chart_data=result.get("chart_data")
                )
            else:
                return QueryResponse(answer=result)
        
        # Use multi-agent orchestrator for orchestrator queries
        logger.info("Attempting multi-agent orchestration for question: {}...".format(query_request.question[:50]))

        try:
            result = await app.state.orchestrator.orchestrate_investigation(
                query_request.question,
                {}  # Default metrics
            )
            logger.debug("Orchestrator returned result type: %s", type(result))
            logger.debug("Orchestrator result keys: %s", list(result.keys()) if isinstance(result, dict) else 'Not a dict')
            logger.debug("Orchestrator status: %s", result.get('status') if isinstance(result, dict) else 'No status')

        except Exception as orch_error:
            logger.error("Orchestrator exception: %s: %s", type(orch_error).__name__, str(orch_error))
            logger.error("Orchestrator exception details: %s", repr(orch_error))
            # Force fallback on exception
            result = {"status": "error", "error": "Orchestrator exception: {}".format(str(orch_error))}

        if result.get("status") == "success":
            logger.info("Multi-agent orchestration successful.")
            combined_analysis = result.get("combined_analysis", "No combined_analysis field found")
            logger.debug("Combined analysis preview: %s...", str(combined_analysis)[:200])

            # Extract ROI analysis from Agent 3 for UI consistency (backend-as-source-of-truth)
            roi_data = None
            agent3_data = result.get("agents", {}).get("agent_3", {})
            if agent3_data and agent3_data.get("response"):
                roi_analysis = agent3_data["response"].findings.get("roi_analysis", {})
                if roi_analysis:
                    roi_data = {
                        "monthly_savings": roi_analysis.get("total_monthly_savings", 15000),
                        "annual_savings": roi_analysis.get("total_annual_savings", 180000),
                        "roi_percentage": roi_analysis.get("roi_percentage", 300),
                        "payback_months": roi_analysis.get("payback_months", 6.0),
                        "implementation_cost": roi_analysis.get("total_implementation_cost", 25000)
                    }
                    logger.debug("Extracted ROI data: %s", roi_data)

            # Track metrics
            latency = time.time() - start_time
            nim_latency_seconds.labels(model=config.REASONING_MODEL).observe(latency)
            query_requests_total.labels(endpoint="invocations", status="success").inc()

            return QueryResponse(answer=combined_analysis, chart_data=roi_data)
        else:
            # Fallback to single agent if orchestrator fails
            error_msg = result.get("error", "Multi-agent investigation failed, falling back to single agent.")
            logger.error("Orchestrator failed: {}".format(error_msg))
            logger.info("Falling back to single agent for question: {}...".format(query_request.question[:50]))

            result = agent.query(query_request.question)

            # Track metrics
            latency = time.time() - start_time
            nim_latency_seconds.labels(model=config.REASONING_MODEL).observe(latency)
            query_requests_total.labels(endpoint="invocations", status="success").inc()

            # Handle both dict and string responses
            if isinstance(result, dict):
                return QueryResponse(
                    answer=result.get("answer", ""),
                    chart_data=result.get("chart_data")
                )
            else:
                return QueryResponse(answer=result)

    except InputValidationError as error:
        status_label = "validation_error"
        query_requests_total.labels(endpoint="invocations", status=status_label).inc()
        logger.warning("Invalid query: %s", str(error))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error)
        )

    except RuntimeError as error:
        status_label = "runtime_error"
        query_requests_total.labels(endpoint="invocations", status=status_label).inc()
        logger.error("Query processing failed: %s", str(error))
        import traceback
        logger.error("Traceback: %s", traceback.format_exc())
        # Provide detailed error information for troubleshooting
        error_detail = str(error)
        if "Circuit breaker" in error_detail:
            error_detail = "Service temporarily unavailable. Please try again in a moment."
        elif "vector store" in error_detail.lower() or "index" in error_detail.lower():
            error_detail = "Data indexing may be incomplete. Please try again in a moment."
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail or "Failed to process query"
        )
    except Exception as error:
        status_label = "unknown_error"
        query_requests_total.labels(endpoint="invocations", status=status_label).inc()
        logger.error("Unexpected error: %s", str(error))
        import traceback
        logger.error("Traceback: %s", traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error: {}".format(str(error))
        )


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/query", response_model=QueryResponse)
async def query_agent_endpoint(request: QueryRequest):
    """Process a query through the AI agent with intelligent routing."""
    try:
        # Fast path: Use single agent for agent1 queries (bypass slow orchestrator)
        if request.agent_type == 'agent1':
            logger.info("Using fast path: single agent for question: {}...".format(request.question[:50]))
            # Run synchronous agent.query in thread pool to avoid blocking
            result = await asyncio.to_thread(agent.query, request.question)
            if isinstance(result, dict):
                return QueryResponse(answer=result.get("answer", ""), chart_data=result.get("chart_data"))
            else:
                return QueryResponse(answer=result)
        
        # Use multi-agent orchestrator for orchestrator queries
        logger.info("Attempting multi-agent orchestration for question: {}...".format(request.question[:50]))
        result = await app.state.orchestrator.orchestrate_investigation(
            request.question,
            {}  # Default metrics
        )

        if result.get("status") == "success":
            logger.info("Multi-agent orchestration successful.")
            combined_analysis = result.get("combined_analysis", "No combined_analysis field found")
            return QueryResponse(answer=combined_analysis)
        else:
            # Fallback to legacy only if orchestrator fails
            error_msg = result.get("error", "Multi-agent investigation failed, falling back to single agent.")
            logger.error("Orchestrator failed: {}".format(error_msg))
            single_agent_result = agent.query(request.question)
            if isinstance(single_agent_result, dict):
                return QueryResponse(answer=single_agent_result.get("answer", ""), chart_data=single_agent_result.get("chart_data"))
            else:
                return QueryResponse(answer=single_agent_result)

    except Exception as error:
        logger.error("Query processing failed: {}".format(str(error)))
        import traceback
        logger.error("Traceback: {}".format(traceback.format_exc()))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query: {}".format(str(error))
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)