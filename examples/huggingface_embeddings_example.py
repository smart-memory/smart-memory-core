"""
Example: Using HuggingFace Embeddings with SmartMemory

This example demonstrates how to use HuggingFace models for embeddings
instead of OpenAI. Supports both API-based and local model inference.
"""

import os

from smartmemory.plugins.embedding import EmbeddingService
from smartmemory.smart_memory import SmartMemory


def example_huggingface_api():
    """Example using HuggingFace Inference API (requires API key)"""
    print("\n=== HuggingFace API Example ===")
    print("⚠️  WARNING: This example requires a valid HuggingFace API key.")
    print("   Replace 'your_hf_api_key_here' with your actual API key from https://huggingface.co/settings/tokens")
    print()

    # Set environment variables
    os.environ['EMBEDDING_PROVIDER'] = 'huggingface'
    os.environ['HUGGINGFACE_API_KEY'] = 'your_hf_api_key_here'  # ⚠️ REPLACE WITH YOUR ACTUAL API KEY
    os.environ['HUGGINGFACE_MODEL'] = 'sentence-transformers/all-MiniLM-L6-v2'

    # Create embedding service
    config = {
        'provider': 'huggingface',
        'huggingface_api_key': os.environ.get('HUGGINGFACE_API_KEY'),
        'huggingface_model': 'sentence-transformers/all-MiniLM-L6-v2'
    }

    embedding_service = EmbeddingService(config)

    # Generate embedding
    text = "This is a test sentence for embedding generation."
    embedding = embedding_service.embed(text)

    print(f"Generated embedding with shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")


def example_huggingface_local():
    """Example using local HuggingFace model (no API key needed)"""
    print("\n=== HuggingFace Local Model Example ===")

    # Set environment variables
    os.environ['EMBEDDING_PROVIDER'] = 'huggingface'
    os.environ['HUGGINGFACE_MODEL'] = 'sentence-transformers/all-MiniLM-L6-v2'

    # Create embedding service (no API key = local inference)
    config = {
        'provider': 'huggingface',
        'huggingface_model': 'sentence-transformers/all-MiniLM-L6-v2'
    }

    embedding_service = EmbeddingService(config)

    # Generate embedding
    text = "Local HuggingFace model inference example."
    embedding = embedding_service.embed(text)

    print(f"Generated embedding with shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")


def example_smartmemory_with_huggingface():
    """Example using SmartMemory with HuggingFace embeddings"""
    print("\n=== SmartMemory with HuggingFace Example ===")

    # Set environment variables
    os.environ['EMBEDDING_PROVIDER'] = 'huggingface'
    os.environ['HUGGINGFACE_MODEL'] = 'sentence-transformers/all-MiniLM-L6-v2'

    # Initialize SmartMemory (will automatically use HuggingFace embeddings)
    memory = SmartMemory()

    # Ingest some memories (full pipeline)
    memory.ingest("Python is a high-level programming language.")
    memory.ingest("I learned about neural networks today.")
    memory.ingest("To train a model: load data, define model, train, evaluate.")

    # Search using semantic similarity (powered by HuggingFace embeddings)
    results = memory.search("programming languages", top_k=2)

    print("\nSearch results for 'programming languages':")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.content}")


def compare_embedding_providers():
    """Compare different embedding providers"""
    print("\n=== Comparing Embedding Providers ===")

    text = "Machine learning is a subset of artificial intelligence."

    # Test different providers
    providers = [
        {'provider': 'openai', 'model': 'text-embedding-ada-002'},
        {'provider': 'huggingface', 'huggingface_model': 'sentence-transformers/all-MiniLM-L6-v2'},
        {'provider': 'ollama', 'model': 'llama2', 'ollama_url': 'http://localhost:11434'}
    ]

    for config in providers:
        try:
            service = EmbeddingService(config)
            embedding = service.embed(text)
            print(f"\n{config['provider'].upper()}:")
            print(f"  Embedding dimension: {embedding.shape[0]}")
            print(f"  First 3 values: {embedding[:3]}")
        except Exception as e:
            print(f"\n{config['provider'].upper()}: Failed - {e}")


if __name__ == "__main__":
    print("HuggingFace Embeddings Examples")
    print("=" * 50)

    # Note: Install required packages first:
    # pip install transformers torch sentence-transformers

    try:
        # Run examples
        example_huggingface_local()
        # example_huggingface_api()  # Uncomment if you have HF API key
        # example_smartmemory_with_huggingface()  # Uncomment to test with SmartMemory
        # compare_embedding_providers()  # Uncomment to compare providers

    except ImportError as e:
        print(f"\nMissing dependencies: {e}")
        print("\nInstall required packages:")
        print("  pip install transformers torch sentence-transformers")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
