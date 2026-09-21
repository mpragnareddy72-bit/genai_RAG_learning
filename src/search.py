import logging

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
            semantic_top_k=10,
            bm25_top_k=10,
            final_top_k=10,
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
    You are a helpful question-answering assistant.

    Answer the user's question using ONLY the provided context.

    If the answer cannot be found in the context,
    say that the information is not available in the documents.

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