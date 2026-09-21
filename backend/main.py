import sys
import os
import logging

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.search import RAGSearch
from backend.router.rag import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Search API",
    description="Hybrid RAG system using Nomic, BM25, RRF, Reranker and Ollama",
    version="1.0.0",)

#CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",
        "http://127.0.0.1:5173",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Routers
# ==========================================

app.include_router(
    router
)


# ==========================================
# Health check
# ==========================================

@app.get("/")
def root():

    return {
        "message": "RAG API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }

#rag_search = None


'''@app.on_event("startup")
def load_rag_pipeline():
    global rag_search

    logger.info("[INFO] Loading RAG pipeline at startup")

    rag_search = RAGSearch(
        llm_model="llama3.2"
    )

    logger.info("[INFO] RAG pipeline ready")


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(router)'''