"""Startup validation checks for NVIDIA NIM model configuration.

This module validates that configured models are on the approved whitelist
for the AWS & NVIDIA Hackathon competition requirements.
"""

import os
import logging

logger = logging.getLogger(__name__)


REASONING_WHITELIST = [
    "nvidia/llama-3.1-nemotron-nano-8b-v1",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/llama-3.1-nemotron-51b-instruct"
]

EMBEDDING_WHITELIST = [
    "nvidia/nv-embedqa-e5-v5",
    "nvidia/nv-embed-v2"
]


def validate_models():
    """Validate configured models against competition whitelist.

    Checks that REASONING_MODEL and EMBEDDING_MODEL environment variables
    contain approved NVIDIA NIM model identifiers. Logs the selected models
    and raises ValueError if invalid models are configured.

    Returns:
        Tuple of (reasoning_model, embedding_model)

    Raises:
        ValueError: If either model is not in the approved whitelist
    """
    reasoning_model = os.getenv("REASONING_MODEL", "nvidia/llama-3.1-nemotron-nano-8b-v1")
    embedding_model = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")

    # Validate reasoning model
    if reasoning_model not in REASONING_WHITELIST:
        raise ValueError(
            "Reasoning model {0} not in whitelist. Allowed models: {1}".format(
                reasoning_model,
                ", ".join(REASONING_WHITELIST)
            )
        )

    # Validate embedding model
    if embedding_model not in EMBEDDING_WHITELIST:
        raise ValueError(
            "Embedding model {0} not in whitelist. Allowed models: {1}".format(
                embedding_model,
                ", ".join(EMBEDDING_WHITELIST)
            )
        )

    # Log approved models
    logger.info("=" * 60)
    logger.info("NVIDIA NIM Model Configuration")
    logger.info("=" * 60)
    logger.info("Using reasoning model: {0}".format(reasoning_model))
    logger.info("Using embedding model: {0}".format(embedding_model))
    logger.info("=" * 60)

    print("=" * 60)
    print("NVIDIA NIM Model Configuration")
    print("=" * 60)
    print("Using reasoning model: {0}".format(reasoning_model))
    print("Using embedding model: {0}".format(embedding_model))
    print("=" * 60)

    return reasoning_model, embedding_model


def get_embedding_dimension(embedding_model):
    """Get embedding dimension for specified model.

    Args:
        embedding_model: NVIDIA embedding model identifier

    Returns:
        Embedding dimension (1024 or 4096)
    """
    if "nv-embed-v2" in embedding_model:
        return 4096
    else:
        return 1024


if __name__ == "__main__":
    """Run validation as standalone script for testing."""
    logging.basicConfig(level=logging.INFO)

    try:
        reasoning, embedding = validate_models()
        dim = get_embedding_dimension(embedding)
        print("\nValidation passed!")
        print("Embedding dimension: {0}".format(dim))
    except ValueError as error:
        print("\nValidation failed: {0}".format(error))
        exit(1)
