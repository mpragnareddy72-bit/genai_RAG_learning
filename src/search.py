import os
import time
import logging
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from langfuse import observe, get_client
try:
    from .vectorstores import QdrantVectorStore
except ImportError:
    from  vectorstores import QdrantVectorStore  

# --------------------------------------------------
# Env
# --------------------------------------------------

load_dotenv()

# --------------------------------------------------
# Python Logger
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Langfuse
# --------------------------------------------------

langfuse = get_client()


class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "qdrant_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama3.2",
        collection_name: str = "documents",
        ollama_base_url: str = "http://localhost:11434",
    ):
        logger.info("Initializing RAGSearch | collection=%s | embedding_model=%s | llm_model=%s",
                    collection_name, embedding_model, llm_model)

        self.vectorstore = QdrantVectorStore(
            collection_name=collection_name,
            persist_dir=persist_dir,
            embedding_model=embedding_model,
        )

        self.llm = ChatOllama(model=llm_model, base_url=ollama_base_url,temperature=0.1)
        logger.info("[INFO] Ollama LLM initialized | model=%s | base_url=%s", llm_model, ollama_base_url)

    # --------------------------------------------------
    # Retrieval step (timed + traced separately)
    # --------------------------------------------------
    @observe(name="retrieve_context")
    def _retrieve(self, query: str, top_k: int) -> str:
        logger.info("[INFO] Retrieving context | query='%s' | top_k=%d", query, top_k)
        t0 = time.perf_counter()

        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r.get("metadata")]
        context = "\n\n".join(texts)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[INFO] Retrieval completed | results=%d | chars=%d | time_ms=%.2f",
            len(results), len(context), elapsed_ms,
        )

        langfuse.update_current_span(
            input={"query": query, "top_k": top_k},
            output={
                "results_returned": len(results),
                "context_length_chars": len(context),
                "retrieval_time_ms": round(elapsed_ms, 2),
            },
        )
        return context

    # --------------------------------------------------
    # Generation step (timed + traced separately)
    # --------------------------------------------------
    @observe(name="generate_answer")
    def _generate(self, query: str, context: str) -> str:
        logger.info("[INFO] Generating answer | query='%s' | context_chars=%d", query, len(context))
        t0 = time.perf_counter()

        prompt = (
            f"Summarize the following context for the query: '{query}'\n\n"
            f"Context:\n{context}\n\nSummary:"
        )
        response = self.llm.invoke([prompt])
        answer = response.content

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("[INFO] Generation completed | answer_chars=%d | time_ms=%.2f", len(answer), elapsed_ms)

        langfuse.update_current_span(
            input={"query": query, "context_length_chars": len(context)},
            output={
                "answer_length_chars": len(answer),
                "generation_time_ms": round(elapsed_ms, 2),
            },
        )
        return answer

    # --------------------------------------------------
    # Full RAG query (top-level trace: question -> answer, total time)
    # --------------------------------------------------
    @observe(name="search_and_summarize")
    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        logger.info("Starting RAG query | query='%s' | top_k=%d", query, top_k)
        t0 = time.perf_counter()

        context = self._retrieve(query, top_k=top_k)
        if not context:
            logger.warning("[warn] No relevant documents found for query: '%s'", query)
            answer = "No relevant documents found."
            elapsed_ms = (time.perf_counter() - t0) * 1000
            langfuse.update_current_span(
                input={"query": query, "top_k": top_k},
                output={"answer": answer, "total_time_ms": round(elapsed_ms, 2)},
            )
            return answer

        answer = self._generate(query, context)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "[INFO] RAG query completed | query='%s' | total_time_ms=%.2f",
            query, elapsed_ms,
        )

        # Top-level span: full question -> answer trace with total latency
        langfuse.update_current_span(
            input={"query": query, "top_k": top_k},
            output={
                "answer": answer,
                "total_time_ms": round(elapsed_ms, 2),
            },
        )
        return answer


# Example usage
if __name__ == "__main__":
    logger.info("[INFO] Starting RAG search pipeline")
    try:
        rag_search = RAGSearch(llm_model="llama3.2")

        print("RAG Search ready. Type your question (or 'exit' to quit).\n")

        while True:
            query = input("You: ").strip()

            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                logger.info("[INFO] User ended session")
                break

            try:
                summary = rag_search.search_and_summarize(query, top_k=3)
                print(f"\nAnswer: {summary}\n")
            except Exception:
                logger.exception("[error] Failed to answer query: '%s'", query)
                print("Something went wrong answering that question. Check logs.\n")

    except Exception:
        logger.exception("[error] RAG search pipeline failed to start")

    finally:
        langfuse.flush()