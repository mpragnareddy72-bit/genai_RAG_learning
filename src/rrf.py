import logging

logger = logging.getLogger(__name__)


class ReciprocalRankFusion:

    def __init__(self, k: int=60):
        self.k = k

    def fuse(self, result_lists, top_k: int =5):
        scores = {}
        documents = {}

        for results in result_lists:

            for rank, result in enumerate(results, start=1):

                metadata = result.get("metadata", {})

                chunk_id = metadata.get("chunk_id")

                if not chunk_id:
                    logger.warning(
                        "[WARNING] Missing chunk_id in metadata | metadata=%s",
                        metadata
                    )
                    continue
                #save document metadata for the chunk_id
                documents[chunk_id] = metadata

                #calxulate rrf score
                rrf_score = 1.0 / (self.k + rank)

                if chunk_id not in scores:
                    scores[chunk_id] = 0.0

                scores[chunk_id] = (
                    scores.get(chunk_id, 0.0) + rrf_score
                )

        #sort by rrf score
        ranked_chunks = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        final_results = []

        for chunk_id, score in ranked_chunks[:top_k]:

            final_results.append({
                "chunk_id": chunk_id,
                "score": score,
                "metadata": documents[chunk_id]
            })

        print("\n===== RRF RESULTS =====")

        for r in final_results:
            print(r["metadata"].get("text", ""))

        logger.info(
            "[INFO] RRF fusion completed | input_lists=%d | results=%d",
            len(result_lists),
            len(final_results),
        )

        return final_results