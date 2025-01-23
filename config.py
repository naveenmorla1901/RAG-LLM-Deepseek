"""
Configuration Module

This module manages the configuration settings for the loan processing system.
It handles environment variables, directory structures, and component settings.

Key Components:
- Environment validation
- Directory management
- ChromaDB configuration
- Processing parameters
- RAG settings
"""

import os
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging system
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define project directory structure
PROJECT_ROOT = Path(__file__).parent  # Use this file's location as root
DATA_DIR = PROJECT_ROOT / "data"  # Raw and processed data storage
CACHE_DIR = PROJECT_ROOT / "document_cache"  # Document processing cache
CHROMA_DIR = PROJECT_ROOT / "chroma_db"  # Vector database storage

# Create necessary directories if they don't exist
for directory in [DATA_DIR, CACHE_DIR, CHROMA_DIR]:
    directory.mkdir(exist_ok=True)

# ChromaDB configuration settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,  # Disable usage tracking
    is_persistent=True,  # Enable persistent storage
    allow_reset=True  # Allow collection reset
)

# Document processing parameters
CHUNK_SIZE = 500  # Characters per text chunk
CHUNK_OVERLAP = 50  # Overlap between chunks
BATCH_SIZE = 100  # Number of items to process in batch
MAX_CSV_ROWS = 10000  # Maximum rows to process per CSV batch

# RAG (Retrieval Augmented Generation) settings
TOP_K_RESULTS = 5  # Number of similar documents to retrieve
RELEVANCE_THRESHOLD = 0.3  # Minimum similarity score
MAX_TOKENS = 1024  # Maximum tokens for LLM response

def get_chroma_client(persist_directory: Optional[str] = None) -> chromadb.Client:
    """
    Initialize and return a ChromaDB client.
    
    This function creates a persistent ChromaDB client with the specified
    or default storage location.
    
    Args:
        persist_directory: Optional custom directory for ChromaDB storage
        
    Returns:
        Initialized ChromaDB client
        
    Raises:
        Exception: If client initialization fails
    """
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
    """
    Get existing ChromaDB collection or create new one.
    
    Args:
        client: ChromaDB client instance
        collection_name: Name of the collection
        
    Returns:
        ChromaDB collection object
        
    Raises:
        Exception: If collection creation/retrieval fails
    """
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity for vectors
        )
        
        logger.info(f"✓ Collection '{collection_name}' ready")
        return collection
        
    except Exception as e:
        logger.error(f"Failed to get/create collection: {e}")
        raise

def validate_environment():
    """
    Validate required environment variables.
    
    This function checks for the presence of all required environment
    variables needed for the system to function.
    
    Required variables:
    - AWS_ACCESS_KEY_ID: AWS credentials
    - AWS_SECRET_ACCESS_KEY: AWS credentials
    - AWS_REGION: AWS region for Bedrock
    - DEEPSEEK_API_KEY: DeepSeek API access
    
    Raises:
        ValueError: If any required variables are missing
    """
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

# Validate environment when module is imported
try:
    validate_environment()
except Exception as e:
    logger.error(f"Environment validation failed: {e}")
    raise