import logging
from rank_bm25 import BM25Okapi
try:
    from .vectorstores import QdrantVectorStore
except ImportError:
    from vectorstores import QdrantVectorStore


from rrf import ReciprocalRankFusion

logger = logging.getLogger(__name__)


class BM25Search:

    def __init__(self, documents):
        """
        documents: list of dictionaries containing chunk text and metadata
        """

        self.documents = documents

        # Extract text from every chunk
        self.texts = [
            doc["metadata"].get("text", "")
            for doc in documents
        ]

        # Tokenize each document
        tokenized_documents = [
            text.lower().split()
            for text in self.texts
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_documents)

        logger.info(
            "[INFO] BM25 index created | documents=%d",
            len(self.documents)
        )

    def search(self, query, top_k=5):
        """
        Search documents using BM25 keyword matching.
        """

        # Tokenize query
        tokenized_query = query.lower().split()

        # Calculate BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get indices of highest-scoring documents
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for index in ranked_indices:
            results.append({
                "score": float(scores[index]),
                "metadata": self.documents[index]["metadata"]
            })

        return results

if __name__ == "__main__":
    logger.info("[INFO] Starting BM25 + semantic search test")

    vectorstore = QdrantVectorStore(
        collection_name="documents",
        persist_dir="qdrant_store",
        embedding_model="nomic-embed-text",
    )

    # Get all chunks from Qdrant
    documents = vectorstore.get_all_documents()

    print(f"Total documents/chunks: {len(documents)}")

    # Create BM25 index
    bm25_search = BM25Search(documents)
    # --------------------------------------------------
    # Hybrid search using RRF
    # --------------------------------------------------

    query = "What is population of afganistan"

    # Semantic search
    nomic_results = vectorstore.query(
        query,
        top_k=10
    )

    # Keyword search
    bm25_results = bm25_search.search(
        query,
        top_k=10
    )

    # RRF
    rrf = ReciprocalRankFusion(k=60)

    rrf_results = rrf.fuse(
        [nomic_results, bm25_results],
        top_k=5
    )

    print(f"\nRRF Hybrid results for: {query}\n")

    for i, result in enumerate(rrf_results, start=1):

        print(f"Result {i}")
        print(f"RRF Score: {result['score']}")

        text = result["metadata"].get("text", "")

        if "afghanistan" in text.lower():
            print("AFGHANISTAN CHUNK")

        print(text[:500])
        print("-" * 80)

    '''query = "What is population of afganistan"

    # --------------------------------------------------
    # Check whether Afghanistan exists in Qdrant
    # --------------------------------------------------

    print("\nSearching Qdrant documents for Afghanistan...\n")

    for doc in documents:
        text = doc["metadata"].get("text", "")

        if "afghanistan" in text.lower():
            print("FOUND AFGHANISTAN CHUNK:")
            print(text[:1000])
            print("-" * 80)

    # --------------------------------------------------
    # Nomic / Qdrant semantic search
    # --------------------------------------------------

    print(f"\nNomic/Qdrant results for: {query}\n")

    results = vectorstore.query(
        query,
        top_k=5
    )

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Score: {result['score']}")
        print(result["metadata"].get("text", "")[:500])
        print("-" * 80)

    # --------------------------------------------------
    # BM25 keyword search
    # --------------------------------------------------

    results = bm25_search.search(
        query,
        top_k=5
    )

    print(f"\nBM25 results for: {query}\n")

    for i, result in enumerate(results, start=1):
        print(f"Result {i}")
        print(f"Score: {result['score']}")
        print(f"Text: {result['metadata'].get('text', '')[:500]}")
        print("-" * 80)'''