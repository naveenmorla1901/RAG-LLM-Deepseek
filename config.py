import os
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "document_cache"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Ensure directories exist
for directory in [DATA_DIR, CACHE_DIR, CHROMA_DIR]:
    directory.mkdir(exist_ok=True)

# ChromaDB settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
    allow_reset=True
)

# Document processing settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 100
MAX_CSV_ROWS = 10000  # Limit for CSV processing

# RAG settings
TOP_K_RESULTS = 5
RELEVANCE_THRESHOLD = 0.3
MAX_TOKENS = 1024

def get_chroma_client(persist_directory: Optional[str] = None) -> chromadb.Client:
    """Get ChromaDB client with configured settings"""
    try:
        if persist_directory:
            directory = Path(persist_directory)
            directory.mkdir(exist_ok=True)
        else:
            directory = CHROMA_DIR
            
        client = chromadb.PersistentClient(
            path=str(directory),
            settings=CHROMA_SETTINGS
        )
        
        logger.info(f"✓ ChromaDB client initialized at {directory}")
        return client
        
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB client: {e}")
        raise

def get_or_create_collection(
    client: chromadb.Client,
    collection_name: str = "loan_documents"
) -> chromadb.Collection:
    """Get or create a ChromaDB collection"""
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(f"✓ Collection '{collection_name}' ready")
        return collection
        
    except Exception as e:
        logger.error(f"Failed to get/create collection: {e}")
        raise

def validate_environment():
    """Validate required environment variables"""
    required_vars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_REGION',
        'DEEPSEEK_API_KEY'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
    
    logger.info("✓ Environment variables validated")

# Validate environment on import
try:
    validate_environment()
except Exception as e:
    logger.error(f"Environment validation failed: {e}")
    raise