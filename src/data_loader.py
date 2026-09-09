from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

from typing import List, Any
import logging
import os

from dotenv import load_dotenv

from langfuse import observe, get_client

# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Langfuse
# ---------------------------------------------------------

langfuse = get_client()


# ---------------------------------------------------------
# Document Loader
# ---------------------------------------------------------

@observe(name="load_all_documents")

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """
    logger.info("Starting document ingestion")
    logger.info("Input data directory: %s", data_dir)

    # -----------------------------------------------------
    # Data path
    # -----------------------------------------------------

    # Use project root data folder
    data_path = Path(data_dir).resolve()
    logger.info("Data path: %s", data_path)

    if not data_path.exists():
        logger.error("Data directory does not exist: %s", data_path)
        raise FileNotFoundError(
            f"Data directory does not exist: {data_path}"
        )
    
    documents = []

    # PDF files
    pdf_files = list(data_path.glob('**/*.pdf'))
    logger.info("[debug] Found %d PDF files: %s", len(pdf_files), [str(f) for f in pdf_files])
    for pdf_file in pdf_files:
        logger.info("[debug] Loading PDF: %s", pdf_file)
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            logger.info("[debug]successfully Loaded %d PDF docs from %s", len(loaded), pdf_file.name)
            documents.extend(loaded)
        except Exception as e:
            logger.error("[error] Failed to load PDF %s: %s", pdf_file, e)

    # TXT files
    txt_files = list(data_path.glob('**/*.txt'))
    logger.info("[debug] Found %d TXT files: %s", len(txt_files), [str(f) for f in txt_files])
    for txt_file in txt_files:
        logger.info("[debug] Loading TXT: %s", txt_file)
        try:
            loader = TextLoader(str(txt_file))
            loaded = loader.load()
            logger.info("[debug] successfullyLoaded %d TXT docs from %s", len(loaded), txt_file.name)
            documents.extend(loaded)
        except Exception as e:
            logger.error("[error] Failed to load TXT %s: %s", txt_file, e)

    # CSV files
    csv_files = list(data_path.glob('**/*.csv'))
    logger.info("[debug] Found %d CSV files: %s", len(csv_files), [str(f) for f in csv_files])
    for csv_file in csv_files:
        logger.info("[debug] Loading CSV: %s", csv_file)
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            logger.info("[debug] successfully Loaded %d CSV docs from %s", len(loaded), csv_file.name)
            documents.extend(loaded)
        except Exception as e:
            logger.error("[error] Failed to load CSV %s: %s", csv_file, e)

    # Excel files
    xlsx_files = list(data_path.glob('**/*.xlsx'))
    logger.info("[debug] Found %d Excel files: %s", len(xlsx_files), [str(f) for f in xlsx_files])
    for xlsx_file in xlsx_files:
        logger.info("[debug] Loading Excel: %s", xlsx_file)
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            loaded = loader.load()
            logger.info("[debug] successfully Loaded %d Excel docs from %s", len(loaded), xlsx_file.name)
            documents.extend(loaded)
        except Exception as e:
            logger.error("[error] Failed to load Excel %s: %s", xlsx_file, e)

    # Word files
    docx_files = list(data_path.glob('**/*.docx'))
    logger.info("[debug] Found %d Word files: %s", len(docx_files), [str(f) for f in docx_files])
    for docx_file in docx_files:
        logger.info("[debug] Loading Word: %s", docx_file)
        try:
            loader = Docx2txtLoader(str(docx_file))
            loaded = loader.load()
            logger.info("[debug] successfully Loaded %d Word docs from %s", len(loaded), docx_file.name)
            documents.extend(loaded)
        except Exception as e:
            logger.error("[error] Failed to load Word %s: %s", docx_file, e)

    # JSON files
    json_files = list(data_path.glob('**/*.json'))
    logger.info("[debug] Found %d JSON files: %s", len(json_files), [str(f) for f in json_files])
    for json_file in json_files:
        logger.info("[debug] Loading JSON: %s", json_file)
        try:
            loader = JSONLoader(file_path=str(json_file),jq_schema=".[]",text_content=False)
            loaded = loader.load()
            logger.info("[debug] Loaded %d JSON docs from %s", len(loaded), json_file)
            documents.extend(loaded)
        except Exception as e:
            logger.error("[error] Failed to load JSON %s: %s", json_file, e)

    logger.info("[debug] Total loaded documents: %d", len(documents))
    return documents

# Example usage
if __name__ == "__main__":

    logger.info("Starting application")

    try:

        docs = load_all_documents("data")

        logger.info(
            "Loaded %d documents successfully",
            len(docs),
        )

        print(f"Loaded {len(docs)} documents.")

        if docs:
            print("Example document:")
            print(docs[0])

    except Exception:
        logger.exception("Application failed")

    finally:
        # Send pending Langfuse events
        langfuse.flush()