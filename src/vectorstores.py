import os
import uuid
import logging
from typing import List, Any, Optional, Set

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
#from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaEmbeddings


from langfuse import observe, get_client

try:
    from .embeddig import EmbeddingPipeline 
except ImportError:
    from embeddig import EmbeddingPipeline 

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


def chunk_id_to_point_id(chunk_id: str) -> str:
    """
    Qdrant point IDs must be an unsigned int or a UUID - not an arbitrary
    string. We deterministically derive a UUID from the sha256 chunk_id
    so the *same* chunk always maps to the *same* point id, which is
    what makes upsert-based dedup possible.
    """
    return str(uuid.UUID(bytes=bytes.fromhex(chunk_id[:32])))


@observe(name="qdrant_vector_store")
class QdrantVectorStore:
    def __init__(
        self,
        collection_name: str = "documents",
        persist_dir: str = "qdrant_store",
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.info("Loading embedding model: %s", embedding_model)
        self.model = OllamaEmbeddings(model=embedding_model,base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
        #self.embedding_dim = self.model.get_embedding_dimension()
        # Get embedding dimension from the actual model
        test_embedding = self.model.embed_query("test")
        self.embedding_dim = len(test_embedding)
        logger.info("[INFO] Loaded embedding model: %s (dim=%d)", embedding_model, self.embedding_dim)

        if url:
            logger.info("Connecting to remote Qdrant at %s", url)
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            os.makedirs(persist_dir, exist_ok=True)
            logger.info("Using local persistent Qdrant store at %s", persist_dir)
            self.client = QdrantClient(path=persist_dir)

        self._ensure_collection()

    # --------------------------------------------------
    # Collection setup
    # --------------------------------------------------
    @observe(name="ensure_collection")
    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing:
            logger.info("[INFO] Collection '%s' already exists", self.collection_name)
            return

        logger.info("[INFO] Creating collection '%s' (dim=%d)", self.collection_name, self.embedding_dim)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=self.embedding_dim,
                distance=qmodels.Distance.COSINE,
            ),
        )
        logger.info("[INFO] Collection '%s' created", self.collection_name)

    # --------------------------------------------------
    # Dedup helper
    # --------------------------------------------------
    @observe(name="get_existing_chunk_ids")
    def get_existing_chunk_ids(self) -> Set[str]:
        logger.info("Scanning collection '%s' for existing chunk_ids", self.collection_name)
        existing_ids: Set[str] = set()
        next_offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=next_offset,
                with_payload=["chunk_id"],
                with_vectors=False,
            )
            for p in points:
                cid = p.payload.get("chunk_id") if p.payload else None
                if cid:
                    existing_ids.add(cid)
            if next_offset is None:
                break

        logger.info("[INFO] Found %d existing chunk_ids in store", len(existing_ids))
        langfuse.update_current_span(output={"existing_chunk_ids": len(existing_ids)})
        return existing_ids

    # --------------------------------------------------
    # Build from raw documents (chunk + embed + store, with dedup)
    # --------------------------------------------------
    @observe(name="build_from_documents")
    def build_from_documents(self, documents: List[Any]):
        logger.info("[INFO] Building vector store from %d raw documents", len(documents))

        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        #chunks = emb_pipe.chunk_documents(documents)
        #logger.info("[INFO] Chunking completed | Total chunks: %d", len(chunks))
        # --------------------------------------------------
        # Separate JSON documents from normal documents
        # --------------------------------------------------

        json_documents = []
        normal_documents = []

        for document in documents:
            file_type = document.metadata.get("file_type")

            if file_type == "json":
                json_documents.append(document)
            else:
                normal_documents.append(document)

        logger.info(
            "[INFO] Document split | JSON records: %d | Normal documents: %d",
            len(json_documents),
            len(normal_documents),
        )

        # --------------------------------------------------
        # Chunk normal documents
        # --------------------------------------------------

        chunks = []

        if normal_documents:
            normal_chunks = emb_pipe.chunk_documents(normal_documents)
            chunks.extend(normal_chunks)

            logger.info(
                "[INFO] Normal document chunking completed | chunks=%d",
                len(normal_chunks),
            )

        # --------------------------------------------------
        # JSON documents are already individual records
        # Do NOT character-split them
        # --------------------------------------------------

        if json_documents:
            chunks.extend(json_documents)

            logger.info(
                "[INFO] JSON records kept intact | records=%d",
                len(json_documents),
            )

        logger.info(
            "[INFO] Final documents/chunks ready for embedding | total=%d",
            len(chunks),
        )

        #duplication
        existing_ids = self.get_existing_chunk_ids()
        new_chunks = [c for c in chunks if c.metadata.get("chunk_id") not in existing_ids]
        skipped = len(chunks) - len(new_chunks)
        logger.info(
            "[INFO] Dedup complete | New chunks: %d | Skipped (already stored): %d",
            len(new_chunks), skipped,
        )

        if not new_chunks:
            logger.info("[INFO] Nothing new to embed or store. Exiting.")
            langfuse.update_current_span(
                input={"documents": len(documents)},
                output={"new_chunks": 0, "skipped_duplicates": skipped},
            )
            return

        embeddings = emb_pipe.embed_chunks(new_chunks)
        logger.info("[INFO] Embedding generation completed | New embeddings: %d", len(embeddings))

        self.add_precomputed(new_chunks, np.array(embeddings).astype("float32"))

        langfuse.update_current_span(
            input={"documents": len(documents), "embedding_model": self.embedding_model},
            output={
                "total_chunks_seen": len(chunks),
                "new_chunks_stored": len(new_chunks),
                "skipped_duplicates": skipped,
            },
        )

    # --------------------------------------------------
    # Store precomputed chunks + embeddings directly
    # (use this if you already ran EmbeddingPipeline yourself,
    #  e.g. from embeddig.py's __main__, and don't want to re-embed)
    # --------------------------------------------------
    @observe(name="add_precomputed")
    def add_precomputed(self, chunks: List[Any], embeddings: np.ndarray):
        if len(chunks) != len(embeddings):
            logger.error(
                "[error] chunks (%d) and embeddings (%d) length mismatch",
                len(chunks), len(embeddings),
            )
            raise ValueError("chunks and embeddings must be the same length")

        existing_ids = self.get_existing_chunk_ids()
        points = []
        skipped = 0

        for chunk, vector in zip(chunks, embeddings):
            chunk_id = chunk.metadata.get("chunk_id")
            if not chunk_id:
                logger.warning("[warn] Chunk missing chunk_id, skipping")
                continue
            if chunk_id in existing_ids:
                skipped += 1
                continue

            points.append(
                qmodels.PointStruct(
                    id=chunk_id_to_point_id(chunk_id),
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": chunk_id,
                        "text": chunk.page_content,
                        "source": chunk.metadata.get("source", "unknown"),
                    },
                )
            )

        if not points:
            logger.info("[INFO] All %d chunks already present, nothing upserted", skipped)
            return

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(
            "[INFO] Upserted %d new vectors to Qdrant (skipped %d duplicates)",
            len(points), skipped,
        )
        langfuse.update_current_span(
            input={"chunks_received": len(chunks)},
            output={"upserted": len(points), "skipped_duplicates": skipped},
        )

    def get_all_documents(self):
        """
        Retrieve all stored chunks from Qdrant.
        Used to build the BM25 keyword index.
        """
        points = []
        offset = None

        while True:
            batch, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            points.extend(batch)

            if offset is None:
                break

        documents = []

        for point in points:
            documents.append({
                "id": str(point.id),
                "metadata": point.payload or {}
            })

        logger.info(
            "[INFO] Retrieved all documents from Qdrant | count=%d",
            len(documents)
        )

        return documents

    # --------------------------------------------------
    # Search(retrieve step in rag) and query helpers
    # --------------------------------------------------
    @observe(name="search")
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        logger.info("Searching collection '%s' | top_k=%d", self.collection_name, top_k)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding[0].tolist(),
            limit=top_k,
        )
        points = results.points
        logger.info("[INFO] Search returned %d results", len(points))
        langfuse.update_current_span(
            input={"top_k": top_k},
            output={"results_returned": len(points)},
        )
        return [
            {"id": r.id, "score": r.score, "metadata": r.payload}
            for r in points
        ]

    @observe(name="query")
    def query(self, query_text: str, top_k: int = 5):
        logger.info("[INFO] Querying vector store for: '%s'", query_text)
        query_emb = np.array([self.model.embed_query(query_text)], dtype="float32")
        return self.search(query_emb, top_k=top_k)


# Example usage
if __name__ == "__main__":
    from data_loader import load_all_documents

    logger.info("[INFO] Starting Qdrant vector store pipeline")
    try:
        docs = load_all_documents("data")
        logger.info("[INFO] Loaded %d documents successfully", len(docs))

        store = QdrantVectorStore(collection_name="documents", persist_dir="qdrant_store")
        store.build_from_documents(docs)

        results = store.query("what is generative AI", top_k=3)
        for r in results:
            print(r)

    except Exception:
        logger.exception("[error] Vector store pipeline failed")

    finally:
        langfuse.flush()