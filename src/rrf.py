import logging

logger = logging.getLogger(__name__)


class ReciprocalRankFusion:

    def __init__(self, k=60):
        self.k = k

    def fuse(self, result_lists, top_k=5):
        scores = {}
        documents = {}

        for results in result_lists:

            for rank, result in enumerate(results, start=1):

                metadata = result["metadata"]

                chunk_id = metadata.get("chunk_id")

                if not chunk_id:
                    continue

                documents[chunk_id] = metadata

                if chunk_id not in scores:
                    scores[chunk_id] = 0.0

                scores[chunk_id] += 1 / (self.k + rank)

        ranked_chunks = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        final_results = []

        for chunk_id, score in ranked_chunks[:top_k]:

            final_results.append({
                "score": score,
                "metadata": documents[chunk_id]
            })

        logger.info(
            "[INFO] RRF fusion completed | input_lists=%d | results=%d",
            len(result_lists),
            len(final_results),
        )

        return final_results