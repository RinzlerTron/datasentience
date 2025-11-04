"""Application configuration container.

LOCAL MODE: When NVIDIA_API_KEY is not set, the agent automatically falls
back to local analysis using pattern matching and heuristics. This allows
internal demonstrations without external API dependencies.

PRODUCTION MODE: Set NVIDIA_API_KEY for full LLM-powered analysis.
AWS SAGEMAKER: API key loaded from AWS Secrets Manager.
"""

import os
import json
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file (local development)
load_dotenv()

logger = logging.getLogger(__name__)


def get_secret_from_aws(secret_name, region_name="us-east-1"):
    """Retrieve secret from AWS Secrets Manager.

    Args:
        secret_name: Name of the secret in Secrets Manager
        region_name: AWS region where secret is stored

    Returns:
        Secret value as string, or None if retrieval fails
    """
    try:
        import boto3
        from botocore.exceptions import ClientError

        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )

        try:
            get_secret_value_response = client.get_secret_value(
                SecretId=secret_name
            )
        except ClientError as error:
            logger.error("Failed to retrieve secret from AWS Secrets Manager: %s", str(error))
            return None

        # Secrets Manager returns either SecretString or SecretBinary
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
            try:
                # Try to parse as JSON
                secret_dict = json.loads(secret)
                return secret_dict.get('NVIDIA_API_KEY')
            except json.JSONDecodeError:
                # Return as plain string
                return secret
        else:
            # Binary secret
            return get_secret_value_response['SecretBinary'].decode('utf-8')

    except ImportError:
        logger.warning("boto3 not installed, cannot retrieve from Secrets Manager")
        return None
    except Exception as error:
        logger.error("Unexpected error retrieving secret: %s", str(error))
        return None


class Config:
    """Application configuration container.
    
    All configuration values are loaded from environment variables.
    Defaults are provided for local development.
    
    Attributes:
        ENVIRONMENT: Deployment environment (local, staging, production)
        NVIDIA_API_KEY: API key for NVIDIA NIM inference service
        NVIDIA_API_URL: NVIDIA NIM API endpoint
        PORT: Server port for FastAPI application
        FRONTEND_URL: Frontend URL for CORS configuration
        LOG_LEVEL: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
        MAX_TOKENS: Maximum tokens for LLM responses
        TEMPERATURE: LLM temperature for response generation
        SEARCH_TOP_K: Number of vector search results to retrieve
        REQUEST_TIMEOUT: API request timeout in seconds
    """
    
    # Environment configuration
    ENVIRONMENT = os.getenv("ENVIRONMENT", "local")

    # NVIDIA NIM configuration
    # Try to load from Secrets Manager first if SECRET_ARN is provided
    SECRET_ARN = os.getenv("SECRET_ARN")
    SECRET_NAME = os.getenv("SECRET_NAME", "datasentience/nvidia-api-key")

    NVIDIA_API_KEY: Optional[str] = None # Will be loaded dynamically

    def __init__(self):
        self.NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY") # Load from env first

    def load_secrets(self):
        """Loads NVIDIA_API_KEY from AWS Secrets Manager if configured."""
        if self.SECRET_ARN or self.SECRET_NAME:
            secret_id = self.SECRET_ARN if self.SECRET_ARN else self.SECRET_NAME
            _api_key_from_secrets = get_secret_from_aws(secret_id)
            if _api_key_from_secrets:
                self.NVIDIA_API_KEY = _api_key_from_secrets
                logger.info("Loaded NVIDIA_API_KEY from AWS Secrets Manager")


    NVIDIA_API_URL = os.getenv(
        "NVIDIA_API_URL",
        "https://integrate.api.nvidia.com/v1/chat/completions"
    )
    
    # Server configuration
    PORT = int(os.getenv("PORT", "8000"))
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
    
    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Model parameters
    REASONING_MODEL = os.getenv("REASONING_MODEL", "nvidia/llama-3.1-nemotron-nano-8b-v1")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
    MODEL_NAME = REASONING_MODEL
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_P = float(os.getenv("TOP_P", "0.7"))
    
    # Vector search configuration
    SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "8"))
    
    # Request configuration
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

    # Redis configuration
    REDIS_URL = os.getenv("REDIS_URL")

    # Embedding configuration
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
    
    @classmethod
    def is_production(cls):
        """Check if running in production environment.
        
        Returns:
            bool: True if ENVIRONMENT is 'production'
        """
        return cls.ENVIRONMENT == "production"
    
    @classmethod
    def is_local(cls):
        """Check if running in local development environment.
        
        Returns:
            bool: True if ENVIRONMENT is 'local'
        """
        return cls.ENVIRONMENT == "local"
    
    @classmethod
    def validate(cls):
        """Validate required configuration values.
        
        Raises:
            ValueError: If required configuration is missing in production
        """
        if cls.is_production() and not cls.NVIDIA_API_KEY:
            raise ValueError(
                "NVIDIA_API_KEY is required in production environment. "
                "Set the NVIDIA_API_KEY environment variable."
            )


# Create singleton config instance
config = Config()

