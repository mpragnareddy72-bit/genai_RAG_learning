import logging
import os

from langchain_ollama import ChatOllama

try:
    from .vectorstores import QdrantVectorStore
except ImportError:
    from vectorstores import QdrantVectorStore

try:
    from .hybrid_search import HybridSearch
    from .reranker import Reranker
except ImportError:
    from hybrid_search import HybridSearch
    from reranker import Reranker



logger = logging.getLogger(__name__)


class RAGSearch:

    def __init__(
        self,
        llm_model: str = "llama3.2",
        embedding_model: str = "nomic-embed-text",
        collection_name: str = "documents",
        persist_dir: str = "qdrant_store",
        reranker_model: str = "BAAI/bge-reranker-base",
    ):

        logger.info("Initializing RAGSearch")

        # --------------------------------
        # 1. Vector store
        # --------------------------------

        self.vectorstore = QdrantVectorStore(
            collection_name=collection_name,
            persist_dir=persist_dir,
            embedding_model=embedding_model,
        )

        # --------------------------------
        # 2. Hybrid search
        # --------------------------------

        self.hybrid_search = HybridSearch(
            self.vectorstore
        )

        # --------------------------------
        # 3. Reranker
        # --------------------------------

        self.reranker = Reranker(
            model_name=reranker_model
        )

        # --------------------------------
        # 4. LLM
        # --------------------------------

        self.llm = ChatOllama(
            model=llm_model,
            temperature=0.1,
            base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        )

        logger.info(
            "RAGSearch initialized successfully"
        )

    # ==================================================
    # RETRIEVAL
    # ==================================================

    def _retrieve(
        self,
        query: str,
        top_k: int = 5
    ):

        # --------------------------------
        # Hybrid search
        # --------------------------------

        hybrid_results = self.hybrid_search.search(
            query=query,
            semantic_top_k=5,
            bm25_top_k=5,
            final_top_k=5,
        )

        logger.info(
            "Hybrid results=%d",
            len(hybrid_results)
        )

        # --------------------------------
        # Reranking
        # --------------------------------

        reranked_results = self.reranker.rerank(
            query=query,
            results=hybrid_results,
            top_k=top_k,
        )

        logger.info(
            "Reranked results=%d",
            len(reranked_results)
        )

        return reranked_results

    # ==================================================
    # GENERATION
    # ==================================================

    def _generate(
        self,
        query: str,
        results: list
    ):

        if not results:

            return "I could not find relevant information in the documents."

        # Build context
        context = "\n\n".join(
            result.get("metadata", {}).get("text", "")
            for result in results
        )

        prompt = f"""
You are a question-answering assistant for a Retrieval-Augmented Generation (RAG) system.

Your task is to answer the user's question using ONLY the information provided in the context.

Rules:
1. Use only the provided context.
2. Do not use your own knowledge or make assumptions.
3. If the context contains the answer, answer the question directly and clearly.
4. If the context does not contain enough information to answer the question, say:
   "The information is not available in the documents."
5. Do not invent, guess, or infer missing facts.
6. For factual questions, preserve the values exactly as they appear in the context.


    Context:
    {context}

    Question:
    {query}

    Answer:
    """

        response = self.llm.invoke(prompt)

        return response.content

    # ==================================================
    # MAIN METHOD
    # ==================================================

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5
    ):

        logger.info(
            "RAG query started | query=%s",
            query
        )

        # Retrieve
        results = self._retrieve(
            query=query,
            top_k=top_k
        )

        # Generate answer
        answer = self._generate(
            query=query,
            results=results
        )

        logger.info(
            "RAG query completed"
        )

        return answer