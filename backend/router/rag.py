from fastapi import APIRouter, HTTPException,status
from pydantic import BaseModel,Field
import logging

from src.search import RAGSearch


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)

#rwquest model
class QueryRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask the RAG system"
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of documents passed to the LLM"
    )

#RESPONSE MODEL
class QueryResponse(BaseModel):
    answer: str

#RAG SEARCH INSTANCE
rag_search = RAGSearch()

#API ENDPOINT
@router.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):


    try:
       

        if not request.question.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty",
            )

        
        answer = rag_search.search_and_summarize(
                request.question,
                top_k=request.top_k,
            )

        return QueryResponse(answer=answer)

    except Exception as e:

        logger.exception(
            "RAG request failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(e)}"
        )