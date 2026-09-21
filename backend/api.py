import sys
import os
import logging

# Add project root to path so `src` is importable from inside backend/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.search import RAGSearch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_search = None


@app.on_event("startup")
def load_rag_pipeline():
    global rag_search
    logger.info("[INFO] Loading RAG pipeline at startup")
    rag_search = RAGSearch(llm_model="llama3.2")
    logger.info("[INFO] RAG pipeline ready")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info("[INFO] Received question: '%s'", request.question)
    try:
        answer = rag_search.search_and_summarize(request.question, top_k=request.top_k)
        return QueryResponse(answer=answer)
    except Exception:
        logger.exception("[error] Failed to process question: '%s'", request.question)
        raise HTTPException(status_code=500, detail="Failed to generate an answer")