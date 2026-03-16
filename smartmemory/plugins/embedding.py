"""Embedding service with token cost tracking (CFS-1a).

Provides embedding generation with:
- Multi-provider support (OpenAI, Ollama, HuggingFace)
- Redis caching for cost reduction
- Token usage tracking for OpenAI embeddings
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from smartmemory.configuration import MemoryConfig

logger = logging.getLogger(__name__)

# Thread-local storage for last embedding usage (similar to DSPy pattern)
_thread_local = threading.local()


@dataclass
class EmbeddingUsage:
    """Token usage from an embedding call."""

    prompt_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    cached: bool = False


def get_last_embedding_usage() -> dict[str, Any] | None:
    """Get and clear the last embedding usage from this thread.

    Returns:
        Dict with prompt_tokens, total_tokens, model, cached — or None if no recent call.

    """
    usage = getattr(_thread_local, "last_usage", None)
    _thread_local.last_usage = None  # Consume-once
    if usage:
        return {
            "prompt_tokens": usage.prompt_tokens,
            "total_tokens": usage.total_tokens,
            "model": usage.model,
            "cached": usage.cached,
        }
    return None


def _set_last_usage(usage: EmbeddingUsage) -> None:
    """Store usage for later retrieval."""
    _thread_local.last_usage = usage


class EmbeddingService:
    """
    Abstracts embedding computation for different providers (OpenAI, Ollama, etc).
    """

    def __init__(self, config=None):
        if config is None:
            try:
                config = MemoryConfig().vector.get("embedding", {})
            except Exception:
                config = {}
        self.provider = config.get("provider", "openai")
        self.model = config.get("models", "text-embedding-ada-002")
        self.api_key = config.get("openai_api_key")
        self.ollama_url = config.get("ollama_url", "http://localhost:11434")
        self.hf_api_key = config.get("huggingface_api_key")
        self.hf_model = config.get("huggingface_model", "sentence-transformers/all-MiniLM-L6-v2")
        self._hf_tokenizer = None
        self._hf_model_instance = None

    def embed(self, text):
        """Generate embedding for text with token usage tracking.

        Token usage is stored in thread-local and can be retrieved via
        ``get_last_embedding_usage()`` immediately after this call.
        """
        # Try Redis cache first for significant performance improvement
        try:
            from smartmemory.utils.cache import get_cache

            cache = get_cache()

            # Check cache for existing embedding
            cached_embedding = cache.get_embedding(text)
            if cached_embedding is not None:
                logger.debug(f"Cache hit for embedding: {text[:50]}...")
                # Record cache hit for token tracking (CFS-1a)
                _set_last_usage(
                    EmbeddingUsage(
                        prompt_tokens=0,
                        total_tokens=0,
                        model=self.model,
                        cached=True,
                    )
                )
                return np.array(cached_embedding)

            logger.debug(f"Cache miss for embedding: {text[:50]}...")
        except Exception as e:
            logger.debug(f"Redis cache unavailable for embeddings: {e}")
            cache = None

        # Generate embedding via API
        usage = EmbeddingUsage(model=self.model, cached=False)

        if self.provider == "openai":
            if not self.api_key:
                # No API key — fall back to spaCy document vectors (96-dim).
                # These provide real semantic similarity from word2vec averaged
                # over the document, at zero cost and zero latency.
                embedding = self._embed_spacy(text)
                if embedding is not None:
                    _set_last_usage(EmbeddingUsage(model="spacy/en_core_web_sm", cached=False))
                    if cache is not None:
                        try:
                            cache.set_embedding(text, embedding.tolist())
                        except Exception:
                            pass
                    return embedding
                # spaCy unavailable — this shouldn't happen since spaCy is a
                # core dep, but guard against broken installs
                raise RuntimeError(
                    "No embedding provider available. Set OPENAI_API_KEY for API "
                    "embeddings or install spaCy with: python -m spacy download en_core_web_sm"
                )
            else:
                import openai

                openai.api_key = self.api_key
                resp = openai.embeddings.create(input=text, model=self.model)
                embedding = np.array(resp.data[0].embedding)
                # Capture actual token usage from OpenAI response (CFS-1a)
                if hasattr(resp, "usage") and resp.usage:
                    usage.prompt_tokens = getattr(resp.usage, "prompt_tokens", 0)
                    usage.total_tokens = getattr(resp.usage, "total_tokens", 0)
        elif self.provider == "ollama":
            import requests

            url = f"{self.ollama_url}/api/embeddings"
            resp = requests.post(url, json={"models": self.model, "prompt": text})
            resp.raise_for_status()
            embedding = np.array(resp.json()["embedding"])
            # Ollama doesn't report token usage
        elif self.provider == "huggingface":
            embedding = self._embed_huggingface(text)
            # HuggingFace local models don't report token usage
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

        # Store usage for retrieval (CFS-1a)
        _set_last_usage(usage)

        # Cache the result for future use
        if cache is not None:
            try:
                cache.set_embedding(text, embedding.tolist())
                logger.debug(f"Cached embedding for: {text[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to cache embedding: {e}")

        return embedding

    def _embed_huggingface(self, text):
        """
        Generate embeddings using HuggingFace models.
        Supports both API-based and local model inference.
        """
        # Try API-based approach first if API key is provided
        if self.hf_api_key:
            try:
                import requests

                api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.hf_model}"
                headers = {"Authorization": f"Bearer {self.hf_api_key}"}
                response = requests.post(api_url, headers=headers, json={"inputs": text})
                response.raise_for_status()
                embedding = np.array(response.json())
                # HuggingFace API returns shape (1, seq_len, hidden_dim), take mean over seq_len
                if len(embedding.shape) == 3:
                    embedding = embedding[0].mean(axis=0)
                elif len(embedding.shape) == 2:
                    embedding = embedding.mean(axis=0)
                return embedding
            except Exception as e:
                logger.warning(f"HuggingFace API failed, falling back to local model: {e}")

        # Fall back to local model inference
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            # Lazy load model and tokenizer (cache for reuse)
            if self._hf_tokenizer is None or self._hf_model_instance is None:
                logger.info(f"Loading HuggingFace model: {self.hf_model}")
                try:
                    self._hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
                    self._hf_model_instance = AutoModel.from_pretrained(self.hf_model)
                    self._hf_model_instance.eval()  # Set to evaluation mode
                except Exception as load_error:
                    logger.error(f"Failed to load HuggingFace model: {load_error}")
                    raise RuntimeError(
                        f"Failed to load HuggingFace model '{self.hf_model}': {load_error}"
                    ) from load_error

            # Verify tokenizer and model are loaded before using
            if self._hf_tokenizer is None or self._hf_model_instance is None:
                raise RuntimeError("HuggingFace tokenizer or model is None after loading attempt")

            # Tokenize and generate embeddings
            inputs = self._hf_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

            with torch.no_grad():
                outputs = self._hf_model_instance(**inputs)
                # Use mean pooling over token embeddings
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embedding = embeddings[0].cpu().numpy()

            return embedding

        except ImportError as e:
            raise ImportError(
                "HuggingFace embeddings require 'transformers' and 'torch'. "
                "Install with: pip install transformers torch"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to generate HuggingFace embedding: {e}") from e


    # Singleton spaCy nlp instance — loaded once, reused across calls.
    _spacy_nlp = None

    def _embed_spacy(self, text):
        """Generate embedding using spaCy's built-in word vectors (96-dim).

        Uses the already-installed en_core_web_sm model. Averages word vectors
        over the document to produce a fixed-size embedding. Quality is lower
        than transformer-based models but provides real semantic similarity
        at zero cost and zero additional dependencies.
        """
        try:
            if EmbeddingService._spacy_nlp is None:
                import spacy
                EmbeddingService._spacy_nlp = spacy.load("en_core_web_sm")
            doc = EmbeddingService._spacy_nlp(text)
            if doc.vector_norm == 0:
                logger.debug("spaCy produced zero vector (no word vectors for input)")
                return None
            return doc.vector
        except Exception as e:
            logger.debug(f"spaCy embedding failed: {e}")
            return None


def create_embeddings(text):
    return EmbeddingService().embed(text)
