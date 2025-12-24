"""
Enhanced BioRAG FastAPI Application - Phase 1 (CORRECTED)
Production-ready server with comprehensive health monitoring, CORS, and auth routes
"""
import sys
import os

# Add parent directory to path to allow Backend imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix Windows console encoding issues
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
import json
import logging
from typing import Dict, Any
from datetime import datetime
import asyncio
import httpx

from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr
from fastapi import FastAPI, HTTPException, status, BackgroundTasks, APIRouter, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ------------------------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------------------------
load_dotenv()
load_dotenv("enhanced.env")
load_dotenv(".env")

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.getenv("LOG_FILE", "logs/biorag.log"))
    ]
)

logger = logging.getLogger("biorag.main")

# ------------------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Enhanced BioRAG API",
    description="Phase 1: Biomedical RAG with paragraph generation and citation validation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ------------------------------------------------------------------------------
# CORS
# ------------------------------------------------------------------------------
cors_origins = os.getenv("CORS_ORIGINS", '["http://localhost:3000","http://localhost:8080","http://127.0.0.1:3000"]')

try:
    origins = json.loads(cors_origins)
except Exception:
    origins = ["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Health Monitoring
# ------------------------------------------------------------------------------
health_status: Dict[str, Any] = {
    "elasticsearch_available": False,
    "elasticsearch_last_check": None,
    "faiss_available": False,
    "generator_loaded": False,
    "hybrid_search": False,
    "services_initialized": False,
}

# ------------------------------------------------------------------------------
# Service Management
# ------------------------------------------------------------------------------
_services: Dict[str, Any] = {}

async def get_services() -> Dict[str, Any]:
    """Initialize services with proper error handling"""
    if _services:
        return _services

    try:
        logger.info("Initializing core services...")

        # Import services
        from Backend.services.document_service import DocumentService
        from Backend.services.embedding_service import EmbeddingService
        from Backend.services.vector_search_service import EnhancedVectorSearchService
        from Backend.services.rag_service import EnhancedRAGService
        from Backend.services.evaluation_service import EvaluationService

        # Initialize services
        _services["document"] = DocumentService()
        _services["embedding"] = EmbeddingService()
        
        _services["vector_search"] = EnhancedVectorSearchService(
            es_host=os.getenv("ES_HOST", "http://localhost:9200"),
            es_user=os.getenv("ES_USER", "elastic"),
            es_password=os.getenv("ES_PASSWORD", "")
        )
        
        # USE LIGHTWEIGHT SERVICE - Fast and reliable for demos
        from Backend.services.lightweight_rag_service import LightweightRAGService
        _services["rag"] = LightweightRAGService()
        logger.info("⚡ Using Lightweight RAG service (fast, real PubMed data, no heavy models)")
        
        _services["evaluation"] = EvaluationService(_services["rag"])

        # Don't initialize vector search - not needed for lightweight mode
        # await _services["vector_search"].initialize()

        # Update health status
        health_status["generator_loaded"] = True  # Lightweight mode doesn't need models
        health_status["services_initialized"] = True

        logger.info("✅ All services initialized successfully")
        return _services

    except ImportError as e:
        logger.error(f"❌ Service import failed: {e}")
        logger.error("Ensure all service files are in Backend/services/ directory")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service import failed: {e}",
        )
    except Exception as e:
        logger.exception("❌ Service initialization failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service initialization failed: {e}",
        )

# ------------------------------------------------------------------------------
# Health Check Utilities
# ------------------------------------------------------------------------------
async def check_elasticsearch_health() -> bool:
    """Check Elasticsearch connectivity"""
    es_host = os.getenv("ES_HOST", "http://localhost:9200")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{es_host}/_cluster/health")
            ok = resp.status_code == 200
            health_status["elasticsearch_last_check"] = datetime.utcnow().isoformat()
            
            if ok:
                logger.info(f"✅ Elasticsearch OK: {resp.json().get('status', 'unknown')}")
            else:
                logger.warning(f"⚠️ Elasticsearch HTTP {resp.status_code}")
            return ok
            
    except Exception as e:
        logger.warning(f"⚠️ Elasticsearch check failed: {e}")
        health_status["elasticsearch_last_check"] = datetime.utcnow().isoformat()
        return False

# ------------------------------------------------------------------------------
# Startup Event
# ------------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    logger.info("🚀 Starting Enhanced BioRAG API...")

    # Check Elasticsearch
    es_ok = await check_elasticsearch_health()
    health_status["elasticsearch_available"] = es_ok
    
    # Check FAISS availability
    try:
        import faiss  # noqa: F401
        health_status["faiss_available"] = True
        logger.info("✅ FAISS available")
    except ImportError:
        health_status["faiss_available"] = False
        logger.info("⚠️ FAISS not available (install faiss-cpu for dense search)")
    
    health_status["hybrid_search"] = (
        health_status["elasticsearch_available"] and health_status["faiss_available"]
    )
    
    logger.info("✅ Startup complete")

# ------------------------------------------------------------------------------
# Core API Routes
# ------------------------------------------------------------------------------
@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "message": "Enhanced BioRAG API - Phase 1",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "index": "POST /index",
            "query": "POST /query OR POST /api/query", 
            "evaluate": "POST /evaluate",
            "auth": ["POST /api/auth/signup", "POST /api/auth/login"],
        },
    }

@app.get("/health")
async def health() -> Dict[str, Any]:
    es_ok = await check_elasticsearch_health()
    health_status["elasticsearch_available"] = es_ok
    health_status["hybrid_search"] = es_ok and health_status["faiss_available"]
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "elasticsearch_available": health_status["elasticsearch_available"],
            "faiss_available": health_status["faiss_available"],
            "generator_loaded": health_status["generator_loaded"],
            "hybrid_search": health_status["hybrid_search"],
            "services_initialized": health_status["services_initialized"],
        },
        "environment": {
            "es_host": os.getenv("ES_HOST", "http://localhost:9200"),
            "debug": os.getenv("DEBUG", "true").lower() == "true",
            "cors_enabled": True,
            "database_path": os.getenv("DATABASE_PATH", "users.db"),
        },
        "version": "1.0.0-phase1",
    }

# ------------------------------------------------------------------------------
# Background Tasks
# ------------------------------------------------------------------------------
async def index_documents_task(document_service, embedding_service, vector_search_service, topics, max_docs):
    """Background task for document indexing"""
    try:
        logger.info(f"📄 Starting indexing for {len(topics)} topics...")
        total_indexed = 0
        
        for topic in topics:
            logger.info(f"📚 Indexing topic: {topic}")
            docs = await document_service.search_and_fetch(topic, max_docs)
            
            for doc in docs:
                chunks = await embedding_service.embed_document(doc)
                if chunks:
                    await vector_search_service.add_documents(chunks)
                    total_indexed += len(chunks)
        
        logger.info(f"✅ Indexing completed: {total_indexed} chunks indexed")
        
    except Exception as e:
        logger.exception(f"❌ Indexing failed: {e}")

# ------------------------------------------------------------------------------
# RAG Endpoints
# ------------------------------------------------------------------------------
@app.post("/index")
async def index_documents(request: Dict[str, Any], background_tasks: BackgroundTasks):
    try:
        services = await get_services()
        topics = request.get("topics", ["cancer treatment", "diabetes management"])
        max_docs = min(int(request.get("max_documents", 20)), 100)
        
        if not topics or not isinstance(topics, list):
            raise HTTPException(status_code=400, detail="Topics must be a non-empty list")
        
        background_tasks.add_task(
            index_documents_task,
            services["document"],
            services["embedding"], 
            services["vector_search"],
            topics,
            max_docs,
        )
        
        return {
            "message": f"Indexing started for {len(topics)} topics",
            "topics": topics,
            "max_documents_per_topic": max_docs,
            "status": "processing",
            "estimated_time": f"{len(topics) * max_docs * 2} seconds"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Index request failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

@app.post("/query")
async def process_query(request: Dict[str, Any]):
    """Main query processing endpoint"""
    try:
        services = await get_services()
        query = request.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        if len(query) > 1000:
            raise HTTPException(status_code=400, detail="Query too long (max 1000 characters)")
        
        max_results = min(int(request.get("max_results", 10)), 50)
        user_id = request.get("user_id")
        
        logger.info(f"🔍 Processing query: {query[:50]}...")
        
        result = await services["rag"].process_query(
            query=query, 
            max_results=max_results, 
            user_id=user_id
        )
        
        confidence = float(result.get("confidence_score", 0.0))
        logger.info(f"✅ Query processed (confidence={confidence:.2f}, citations={len(result.get('citations', []))})")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Query processing failed")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {e}")

# CRITICAL FIX: Add /api/query endpoint for frontend compatibility
@app.post("/api/query")
async def api_process_query(request: Dict[str, Any]):
    """API query endpoint - forwards to main query handler"""
    logger.info("📡 API query endpoint called")
    return await process_query(request)

@app.post("/evaluate")
async def run_evaluation(request: Dict[str, Any]):
    try:
        services = await get_services()
        test_queries = request.get("test_queries", [
            "What is the mechanism of action of metformin?",
            "How effective is immunotherapy for lung cancer?", 
            "What are the side effects of ACE inhibitors?",
            "What is the role of CRISPR in gene therapy?",
            "How does COVID-19 affect the cardiovascular system?"
        ])
        
        if not test_queries or not isinstance(test_queries, list):
            raise HTTPException(status_code=400, detail="test_queries must be a non-empty list")
        
        results = await services["evaluation"].run_comprehensive_evaluation(test_queries)
        
        return {
            "evaluation_results": results,
            "timestamp": datetime.utcnow().isoformat(),
            "num_queries": len(test_queries)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Evaluation failed")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")

# ------------------------------------------------------------------------------
# Authentication Routes
# ------------------------------------------------------------------------------
auth_router = APIRouter(prefix="/api/auth", tags=["auth"])

class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    name: str = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

@auth_router.post("/signup")
async def signup(payload: SignupRequest) -> Dict[str, Any]:
    try:
        # FIXED: Correct import path
        from Backend.services.auth_service import get_auth_service
        
        auth = get_auth_service()
        user, token = await auth.create_user(
            email=payload.email, 
            password=payload.password,
            name=payload.name
        )
        
        logger.info(f"✅ User registered: {payload.email}")
        
        return {
            "success": True,
            "user": user, 
            "token": token, 
            "access_token": token,
            "token_type": "bearer"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"❌ Signup failed for {payload.email}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {e}")

@auth_router.post("/login") 
async def login(payload: LoginRequest) -> Dict[str, Any]:
    try:
        # FIXED: Correct import path
        from Backend.services.auth_service import get_auth_service
        
        auth = get_auth_service()
        user, token = await auth.authenticate_user(
            email=payload.email, 
            password=payload.password
        )
        
        logger.info(f"✅ User authenticated: {payload.email}")
        
        return {
            "success": True,
            "user": user, 
            "token": token,
            "access_token": token, 
            "token_type": "bearer"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.exception(f"❌ Login failed for {payload.email}")
        raise HTTPException(status_code=500, detail=f"Authentication failed: {e}")

# Mount auth router
app.include_router(auth_router)

# ------------------------------------------------------------------------------
# Error Handlers
# ------------------------------------------------------------------------------
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "Check /docs for available endpoints",
            "available_endpoints": ["/health", "/", "/index", "/query", "/api/query", "/evaluate", "/api/auth/*"],
        },
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"❌ Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error", 
            "message": "Please check server logs for details"
        },
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "Request failed",
            "detail": exc.detail
        }
    )

# ------------------------------------------------------------------------------
# Application Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "Backend.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True,
        loop="auto"
    )
