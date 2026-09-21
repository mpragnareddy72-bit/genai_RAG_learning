import logging
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class Reranker:

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
    ):
        logger.info(
            "[INFO] Loading reranker model | model=%s",
            model_name,
        )

        self.model = CrossEncoder(model_name)

        logger.info(
            "[INFO] Reranker model loaded successfully"
        )

    def rerank(self, query: str, results: list, top_k: int = 5):

        if not results:
            logger.warning(
                "[WARNING] No results to rerank | query=%s",
                query
            )
            return []

        # Create query-document pairs
        pairs = [
            (
                query,
                result.get("metadata", {}).get("text", "")
            )
            for result in results
        ]

        # Calculate reranker scores
        scores = self.model.predict(pairs)

        # Attach reranker score
        reranked_results = []

        for result, score in zip(results, scores):
            reranked_results.append({
                "chunk_id": result.get("metadata", {}).get("chunk_id"),
                "score": float(score),
                "metadata": result.get("metadata", {})
            })

        # Sort by reranker score
        reranked_results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        print("\n===== RERANKER RESULTS =====")

        for r in reranked_results:
            print(
                "SCORE:",
                r["score"],
                "|",
                r["metadata"].get("text", "")
            )

        final_results = reranked_results[:top_k]

        logger.info(
            "Reranking completed | input=%d | returned=%d",
            len(results),
            len(final_results)
        )

        return final_results