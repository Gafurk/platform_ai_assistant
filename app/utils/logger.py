import logging
import os
from datetime import datetime


def setup_logging():
    """Configure structured logging for the application."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"app_{timestamp}.log")

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler for errors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# Initialize logger
logger = setup_logging()


def log_chat_request(session_id: str, message: str, source: str):
    """Log chat request."""
    logger.info(f"CHAT | session={session_id} | source={source} | msg_len={len(message)}")


def log_chat_error(session_id: str, error_type: str, error_msg: str):
    """Log chat error."""
    logger.error(f"CHAT_ERROR | session={session_id} | type={error_type} | msg={error_msg}")


def log_rag_search(query: str, intent: str, language: str, results_count: int):
    """Log RAG search operation."""
    logger.info(f"RAG_SEARCH | intent={intent} | lang={language} | results={results_count} | query_len={len(query)}")


def log_qdrant_error(error: str):
    """Log Qdrant connection error."""
    logger.error(f"QDRANT_ERROR | {error}")


def log_document_upload(filename: str, size_kb: int, status: str):
    """Log document upload."""
    logger.info(f"DOC_UPLOAD | filename={filename} | size_kb={size_kb} | status={status}")


def log_ingestion_start(filename: str):
    """Log ingestion start."""
    logger.info(f"INGEST_START | filename={filename}")


def log_ingestion_complete(filename: str, chunks_count: int, status: str):
    """Log ingestion completion."""
    logger.info(f"INGEST_DONE | filename={filename} | chunks={chunks_count} | status={status}")


def log_ingestion_error(filename: str, error: str):
    """Log ingestion error."""
    logger.error(f"INGEST_ERROR | filename={filename} | error={error}")
