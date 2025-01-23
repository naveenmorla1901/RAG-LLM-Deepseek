"""
Bedrock Embeddings Module

This module provides functionality for generating text embeddings using AWS Bedrock.
It includes caching, error handling, and vector normalization features.

Key Features:
- LRU caching for efficient embedding retrieval
- Automatic retry configuration
- Vector normalization
- Comprehensive error handling
"""

import boto3
import json
import logging
from typing import List, Union
from botocore.config import Config
import numpy as np
from functools import lru_cache
import hashlib

# Configure logging system
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BedrockEmbeddings:
    """
    A class for generating and managing text embeddings using AWS Bedrock.
    
    This class handles:
    - Text embedding generation
    - Caching of results
    - Vector normalization
    - Error handling and retries
    
    Attributes:
        model_id: AWS Bedrock model identifier
        client: Boto3 client for Bedrock
        get_embedding: Cached embedding function
    """

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v1",
        region_name: str = None,
        cache_size: int = 1000
    ):
        """
        Initialize Bedrock embeddings with caching.
        
        Args:
            model_id: Bedrock model ID for embeddings
            region_name: AWS region for Bedrock service
            cache_size: Maximum number of embeddings to cache
            
        The class automatically configures retry behavior and caching
        to optimize performance and reliability.
        """
        self.model_id = model_id
        
        # Configure boto3 client with retries
        config = Config(
            retries = dict(
                max_attempts = 3  # Retry failed requests up to 3 times
            )
        )
        
        # Initialize Bedrock client
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name,
            config=config
        )
        
        # Setup LRU cache for embeddings
        self.get_embedding = lru_cache(maxsize=cache_size)(self._get_embedding)
        logger.info("✓ Bedrock embeddings initialized")

    def _get_text_hash(self, text: str) -> str:
        """
        Generate a consistent hash for text to use as cache key.
        
        Args:
            text: Input text to hash
            
        Returns:
            MD5 hash of the input text
        
        This method ensures consistent cache keys for identical text inputs.
        """
        return hashlib.md5(text.encode()).hexdigest()

    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text without using cache.
        
        Args:
            text: Input text to embed
            
        Returns:
            Normalized embedding vector as list of floats
            
        Raises:
            Exception: If embedding generation fails
            
        This is the core method that interacts with AWS Bedrock to
        generate embeddings. It includes normalization of the output vector.
        """
        try:
            # Prepare request payload
            body = json.dumps({
                "inputText": text
            })
            
            # Request embedding from Bedrock
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body
            )
            
            # Extract and process embedding
            response_body = json.loads(response['body'].read())
            embedding = response_body['embedding']
            
            # Normalize the embedding vector
            embedding = np.array(embedding)
            normalized = embedding / np.linalg.norm(embedding)
            return normalized.tolist()
            
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts using caching.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of normalized embedding vectors
            
        Raises:
            Exception: If embedding generation fails
            
        This method handles batch processing of texts, utilizing
        the cache for efficiency.
        """
        try:
            embeddings = []
            for text in texts:
                # Leverage cached embeddings
                embedding = self.get_embedding(text)
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Get embedding for a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Normalized embedding vector
            
        Raises:
            Exception: If embedding generation fails
            
        This method is optimized for single query embedding,
        useful for similarity searches.
        """
        try:
            return self.get_embedding(text)
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise

    def clear_cache(self):
        """
        Clear the embedding cache.
        
        This method can be useful to free memory or force
        regeneration of embeddings.
        """
        self.get_embedding.cache_clear()
        logger.info("Embedding cache cleared")

# Create a default instance for easy import
default_embeddings = BedrockEmbeddings()
