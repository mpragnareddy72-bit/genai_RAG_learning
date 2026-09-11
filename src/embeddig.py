import hashlib
from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from langfuse import observe, get_client

from sentence_transformers import SentenceTransformer
import numpy as np
from data_loader import load_all_documents

# --------------------------------------------------
# Python Logger
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Langfuse
# --------------------------------------------------

langfuse = get_client()


class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info( "Loading embedding model: %s",model_name)
        self.model = SentenceTransformer(model_name)
        logger.info(f"[INFO] Loaded embedding model: {model_name}")
      

# --------------------------------------------------
# Chunking(documents to chunking)
# --------------------------------------------------

    @observe(name="chunk_documents")
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        logger.info(
            "Starting document chunking | Documents: %d",
            len(documents)
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        for chunk in chunks:
            chunk_id = hashlib.sha256(
                chunk.page_content.encode("utf-8")
            ).hexdigest()

            chunk.metadata["chunk_id"] = chunk_id
        logger.info("[INFO] Chunk id's generated successfully for %d chunks", len(chunks))

    
        chunk_lengths = [len(chunk.page_content)for chunk in chunks]
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        logger.info( "[INFO] Document chunking completed | " "Documents: %d | Chunks: %d | ""Chunk size: %d | Overlap: %d",
            len(documents),len(chunks),self.chunk_size,self.chunk_overlap)
        if chunk_lengths:
            logger.info(
                "Chunk statistics | Min: %d | Max: %d | Average: %.2f",
                min(chunk_lengths),
                max(chunk_lengths),
                sum(chunk_lengths) / len(chunk_lengths)
            )

            # Add information to the current Langfuse observation
            langfuse.update_current_span(
                input={
                    "documents": len(documents),
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap
                },
                output={
                    "chunks": len(chunks),
                    "min_chunk_length": min(chunk_lengths),
                    "max_chunk_length": max(chunk_lengths),
                    "average_chunk_length": sum(chunk_lengths) / len(chunk_lengths)
                }
            )
        return chunks
    
# --------------------------------------------------
# Embedding(chunks to embeddings)
# --------------------------------------------------
    @observe(name="generate_embeddings")
    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        logger.info("Starting embedding generation | Chunks: %d",len(chunks))

        #1. Extract text from chunks
        texts = [chunk.page_content for chunk in chunks]
        logger.info(f"[INFO] Generating embeddings for {len(texts)} chunks...")

        # 2. Add information to Langfuse
        langfuse.update_current_span(
        input={
            "number_of_chunks": len(chunks),
            "number_of_texts": len(texts),
            "model": self.model_name
        }
    )

        logger.info("Generating embeddings | Model: %s",self.model_name)

        # 3. Generate embeddings using the SentenceTransformer model
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # 4. Get embedding information
        embedding_shape = embeddings.shape
        embedding_dimension = embeddings.shape[1]

        logger.info(
            "Embedding generation completed | "
            "Chunks: %d | Shape: %s | Dimension: %d",
            len(texts),
            embedding_shape,
            embedding_dimension
        )

        # 5. Send result information to Langfuse
        langfuse.update_current_span(
            output={
                "number_of_embeddings": len(embeddings),
                "embedding_shape": str(embedding_shape),
                "embedding_dimension": embedding_dimension,
                "sample_embedding": embeddings[0][:10].tolist()
                if len(embeddings) > 0 else None
            }
        )
        
        return embeddings

# Example usage
if __name__ == "__main__":

    logger.info("[INFO]Starting embedding pipeline")
    try:
        docs = load_all_documents("data")
        logger.info("[INFO] Loaded %d documents successfully", len(docs))

        #initialize embedding pipeline
        emb_pipe = EmbeddingPipeline()

        # Chunk documents
        chunks = emb_pipe.chunk_documents(docs)
        logger.info("[INFO] Document chunking completed")

        embeddings = emb_pipe.embed_chunks(chunks)

        logger.info("[INFO] PIPELINE COMPLETED | Total Chunks: %d | Embeddings shape: %s",len(chunks),embeddings.shape)
        #logger.info("[INFO] Example embedding: %s", embeddings[0] if len(embeddings) > 0 else None)

    except Exception:
        logger.exception(
            "Embedding pipeline failed"
        )

    finally:
        langfuse.flush()