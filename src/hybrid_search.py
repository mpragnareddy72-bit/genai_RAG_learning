import logging
from rank_bm25 import BM25Okapi
try:
    from .vectorstores import QdrantVectorStore
except ImportError:
    from vectorstores import QdrantVectorStore

try:
    from .rrf import ReciprocalRankFusion
except ImportError:
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
            doc.get("metadata", {}).get("text", "")
            for doc in documents
        ]

        # Tokenize each document
        tokenized_documents = [
            text.lower().split()
            for text in self.texts
        ]

        # Build BM25 index(skip if there's nothing to index yet)
        if tokenized_documents:
            self.bm25 = BM25Okapi(tokenized_documents)
        else:
            self.bm25 = None
            logger.warning("[WARN] No documents found — BM25 index not built")


        logger.info(
            "[INFO] BM25 index created | documents=%d",
            len(self.documents)
        )

    def search(self, query: str, top_k: int = 5):
        """
        Search documents using BM25 keyword matching.
        """

        if self.bm25 is None:
            return []

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

        print("\n===== BM25 SEARCH =====")

        for r in results:
            print(r["metadata"].get("text", ""))

        for index in ranked_indices:
            document = self.documents[index]
            results.append({
                "score": float(scores[index]),
                "metadata": document.get("metadata", {})
            })

        return results

class HybridSearch:

    def __init__(self, vectorstore):

        self.vectorstore = vectorstore

        # Get all documents from Qdrant
        documents = vectorstore.get_all_documents()

        

        # Create BM25 index
        self.bm25_search = BM25Search(documents)

        # --------------------------------------------------
        # Hybrid search using RRF
        # --------------------------------------------------

        self.rrf = ReciprocalRankFusion(k=60)

        logger.info(
            "HybridSearch initialized | documents=%d",
            len(documents)
        )
       

    def search(
        self,
        query: str,
        semantic_top_k: int = 10,
        bm25_top_k: int = 10,
        final_top_k: int = 10,
    ):

        logger.info(
            "Hybrid search started | query=%s",
            query
        )

        logger.info(
            "Hybrid search started | query=%s",
            query
        )

        # --------------------------------
        # 1. Nomic semantic search
        # --------------------------------

        semantic_results = self.vectorstore.query(
            query,
            top_k=semantic_top_k
        )

        logger.info(
            "Nomic results=%d",
            len(semantic_results)
        )

        # --------------------------------
        # 2. BM25 keyword search
        # --------------------------------

        bm25_results = self.bm25_search.search(
            query,
            top_k=bm25_top_k
        )

        logger.info(
            "BM25 results=%d",
            len(bm25_results)
        )

        # --------------------------------
        # 3. RRF
        # --------------------------------

        rrf_results = self.rrf.fuse(
            [
                semantic_results,
                bm25_results
            ],
            top_k=final_top_k
        )

        logger.info(
            "Hybrid search completed | results=%d",
            len(rrf_results)
        )

        return rrf_results