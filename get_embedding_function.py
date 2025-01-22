import boto3
import json
import logging
from typing import List, Union
from botocore.config import Config
import numpy as np
from functools import lru_cache
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BedrockEmbeddings:
    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v1",
        region_name: str = None,
        cache_size: int = 1000
    ):
        """Initialize Bedrock embeddings with caching
        
        Args:
            model_id: Bedrock model ID
            region_name: AWS region name
            cache_size: Size of the LRU cache for embeddings
        """
        self.model_id = model_id
        
        # Configure boto3 client
        config = Config(
            retries = dict(
                max_attempts = 3
            )
        )
        
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name,
            config=config
        )
        
        # Initialize cache
        self.get_embedding = lru_cache(maxsize=cache_size)(self._get_embedding)
        logger.info("✓ Bedrock embeddings initialized")

    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text to use as cache key"""
        return hashlib.md5(text.encode()).hexdigest()

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text (without caching)"""
        try:
            # Prepare the request body
            body = json.dumps({
                "inputText": text
            })
            
            # Make the request to Bedrock
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            embedding = response_body['embedding']
            
            # Normalize the embedding
            embedding = np.array(embedding)
            normalized = embedding / np.linalg.norm(embedding)
            return normalized.tolist()
            
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        try:
            embeddings = []
            for text in texts:
                # Use cached embedding if available
                embedding = self.get_embedding(text)
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a query text"""
        try:
            return self.get_embedding(text)
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise

    def clear_cache(self):
        """Clear the embedding cache"""
        self.get_embedding.cache_clear()
        logger.info("Embedding cache cleared")

# Create a default instance
default_embeddings = BedrockEmbeddings()