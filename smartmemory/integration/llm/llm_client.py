"""
Unified LLM Client

Consolidates all LLM client implementations across SmartMemory into a single,
maintainable client supporting multiple providers with consistent interfaces.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Dict, List, Optional, Any, Type

from smartmemory.configuration import MemoryConfig
from .providers import OpenAIProvider, AnthropicProvider, AzureOpenAIProvider, BaseLLMProvider
from .response_parser import ResponseParser, StructuredResponse

# Import Claude Agent SDK (optional — replaces legacy claude_cli package)
try:
    from claude_agent_sdk import query as _claude_agent_query, ClaudeAgentOptions, AssistantMessage, TextBlock
    CLAUDE_AGENT_SDK_AVAILABLE = True
except ImportError:
    _claude_agent_query = None  # type: ignore
    ClaudeAgentOptions = None  # type: ignore
    AssistantMessage = None  # type: ignore
    TextBlock = None  # type: ignore
    CLAUDE_AGENT_SDK_AVAILABLE = False


def _run_async(coro):
    """Run an async coroutine synchronously, handling nested event loops."""
    try:
        asyncio.get_running_loop()
        # Already in an async context — run in a new thread to avoid nesting
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=300)
    except RuntimeError:
        # No event loop running — safe to use asyncio.run()
        return asyncio.run(coro)


class ClaudeAgentSDKProvider(BaseLLMProvider):
    """Provider adapter using the official Claude Agent SDK (claude-agent-sdk).

    Uses Claude Code CLI under the hood. No API key needed — authentication
    is handled by the Claude Code installation.

    Args:
        max_turns: Maximum agent turns per query. Default: 1 (single response)
        system_prompt: Optional default system prompt
        cwd: Working directory for Claude Code

    Examples:
        client = LLMClient(provider='claude-agent')
        client = LLMClient(provider='claude-agent', max_turns=1)
    """

    def __init__(self, config=None, **kwargs):
        if not CLAUDE_AGENT_SDK_AVAILABLE:
            raise ImportError(
                "claude-agent-sdk package not installed. Install with: "
                "pip install claude-agent-sdk"
            )
        self.config = config
        self.provider = "claude-agent"
        self.api_key = None
        self.logger = logging.getLogger(f"{__name__}.ClaudeAgentSDKProvider")
        self._max_turns = kwargs.get("max_turns", 1)
        self._system_prompt = kwargs.get("system_prompt")
        self._cwd = kwargs.get("cwd")

    def _initialize_client(self, **_kwargs):
        pass

    def _get_api_key(self):
        return None

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate completion via Claude Agent SDK."""
        system_prompt = self._system_prompt
        user_parts = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                user_parts.append(msg["content"])

        prompt = "\n\n".join(user_parts)

        async def _do_query():
            options = ClaudeAgentOptions(
                system_prompt=system_prompt,
                max_turns=kwargs.get("max_turns", self._max_turns),
            )
            if self._cwd:
                options.cwd = self._cwd

            text_parts = []
            async for message in _claude_agent_query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
            return "\n".join(text_parts)

        content = _run_async(_do_query())

        return {
            "content": content,
            "models": "claude-agent-sdk",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "metadata": {"finish_reason": "stop", "response_id": ""},
        }

    def structured_completion(self, messages, response_model, **kwargs):  # noqa: ARG002
        del messages, response_model, kwargs
        return None

    def get_supported_features(self):
        return ["chat_completion"]

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Unified response format from LLM providers."""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(UTC)


class LLMClient:
    """
    Unified LLM client supporting multiple providers with consistent interfaces.
    
    Consolidates functionality from:
    - smartmemory.ontology.llm_manager.LLMOntologyManager._get_default_llm_client()
    - smartmemory.utils.llm.call_llm()
    - Various scattered OpenAI client initializations
    """

    def __init__(self,
                 provider: str = "openai",
                 config: Optional[MemoryConfig] = None,
                 api_key: Optional[str] = None,
                 **provider_kwargs):
        """
        Initialize unified LLM client.

        Args:
            provider: Provider name ("openai", "anthropic", "azure_openai", "claude-cli")
            config: Configuration object (defaults to global config)
            api_key: Override API key
            **provider_kwargs: Provider-specific configuration (model, timeout, etc.)
        """
        self.provider_name = provider
        self.config = config or MemoryConfig()
        self.response_parser = ResponseParser()

        # Store model override if provided
        self._model_override = provider_kwargs.get("model") or provider_kwargs.get("default_model")

        # Initialize provider using unified config directly
        self.provider = self._create_provider(api_key, **provider_kwargs)

        logger.info(f"Initialized LLM client with provider: {provider}")

    def _create_provider(self, api_key: Optional[str], **kwargs):
        """Create provider instance based on configuration."""
        provider_classes = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "azure_openai": AzureOpenAIProvider,
            "claude-agent": ClaudeAgentSDKProvider,
            "claude_agent": ClaudeAgentSDKProvider,
            # Legacy aliases — map to new Agent SDK provider
            "claude-cli": ClaudeAgentSDKProvider,
            "claude_cli": ClaudeAgentSDKProvider,
        }

        provider_class = provider_classes.get(self.provider_name)
        if not provider_class:
            raise ValueError(f"Unsupported provider: {self.provider_name}")

        # Claude Agent SDK provider doesn't use api_key
        if self.provider_name in ("claude-agent", "claude_agent", "claude-cli", "claude_cli"):
            return provider_class(config=self.config, **kwargs)

        return provider_class(
            config=self.config,
            provider=self.provider_name,
            api_key=api_key,
            **kwargs
        )

    def chat_completion(self,
                        messages: List[Dict[str, str]],
                        model: Optional[str] = None,
                        temperature: Optional[float] = None,
                        max_tokens: Optional[int] = None,
                        **kwargs) -> LLMResponse:
        """
        Generate chat completion using the configured provider.
        
        Consolidates functionality from:
        - LLMOntologyManager._call_llm()
        - utils.llm.call_llm() fallback path
        """
        try:
            response = self.provider.chat_completion(
                messages=messages,
                model=model or self._get_default_model(),
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            return LLMResponse(
                content=response["content"],
                model=response["models"],
                provider=self.provider_name,
                usage=response.get("usage"),
                metadata=response.get("metadata")
            )

        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise

    def structured_completion(self,
                              messages: List[Dict[str, str]],
                              response_model: Type[Any],
                              model: Optional[str] = None,
                              **kwargs) -> StructuredResponse:
        """
        Generate structured completion with Pydantic models parsing.
        
        Consolidates functionality from:
        - utils.llm.call_llm() with response_model
        - utils.llm.run_ontology_llm()
        """
        try:
            # Try structured parsing first
            structured_response = self.provider.structured_completion(
                messages=messages,
                response_model=response_model,
                model=model or self._get_default_model(),
                **kwargs
            )

            if structured_response:
                return StructuredResponse(
                    parsed_data=structured_response["parsed_data"],
                    raw_content=structured_response.get("raw_content"),
                    model=structured_response["models"],
                    provider=self.provider_name,
                    success=True
                )

        except Exception as e:
            logger.warning(f"Structured completion failed, trying fallback: {e}")

        # Fallback to JSON parsing
        return self._structured_fallback(messages, response_model, model, **kwargs)

    def _structured_fallback(self,
                             messages: List[Dict[str, str]],
                             response_model: Type[Any],
                             model: Optional[str],
                             **kwargs) -> StructuredResponse:
        """Fallback to JSON parsing when structured completion fails."""
        # Add JSON instruction to messages
        json_instruction = (
            "Return ONLY a valid JSON object that matches the required schema. "
            "Do not include markdown fences or commentary."
        )

        fallback_messages = messages + [
            {"role": "system", "content": json_instruction}
        ]

        response = self.chat_completion(
            messages=fallback_messages,
            model=model,
            **kwargs
        )

        # Parse JSON response
        parsed_data = self.response_parser.parse_json_response(
            response.content, response_model
        )

        return StructuredResponse(
            parsed_data=parsed_data,
            raw_content=response.content,
            model=response.model,
            provider=self.provider_name,
            success=parsed_data is not None
        )

    def simple_completion(self,
                          prompt: str,
                          model: Optional[str] = None,
                          **kwargs) -> LLMResponse:
        """
        Simple completion for single prompt.
        
        Convenience method for basic LLM calls.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(messages, model, **kwargs)

    def ontology_completion(self,
                            user_content: str,
                            response_model: Type[Any],
                            system_prompt: Optional[str] = None,
                            **kwargs) -> StructuredResponse:
        """
        Ontology-specific completion method.
        
        Consolidates functionality from:
        - utils.llm.run_ontology_llm()
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        return self.structured_completion(
            messages=messages,
            response_model=response_model,
            **kwargs
        )

    def _get_default_model(self) -> str:
        """Get default model for the provider."""
        # First check if model was explicitly set at construction
        if self._model_override:
            return self._model_override

        # Try extractor.llm.model from config
        extractor_config = getattr(self.config, 'extractor', {})
        if isinstance(extractor_config, dict):
            llm_config = extractor_config.get('llm', {})
            if llm_config and llm_config.get('model'):
                return llm_config['model']

        # Try direct llm.model from config
        llm_config = getattr(self.config, 'llm', {})
        if llm_config and llm_config.get('model'):
            return llm_config['model']

        # Provider defaults (fallback)
        defaults = {
            'openai': 'gpt-4',
            'anthropic': 'claude-3-sonnet-20240229',
            'azure_openai': 'gpt-4',
            'claude-cli': 'haiku',
            'claude_cli': 'haiku',
        }

        return defaults.get(self.provider_name, 'gpt-4')

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider."""
        return {
            "provider": self.provider_name,
            "default_model": self._get_default_model(),
            "supported_features": self.provider.get_supported_features(),
            "configuration": {
                "provider": self.provider_name,
                "default_model": self._get_default_model()
            }
        }

    def validate_connection(self) -> bool:
        """Validate connection to the LLM provider."""
        try:
            test_response = self.simple_completion(
                "Hello",
                max_tokens=10
            )
            return test_response.content is not None
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False


# Backward compatibility functions
def get_default_llm_client(provider: str = "openai") -> LLMClient:
    """
    Get default LLM client - backward compatibility for LLMOntologyManager.
    """
    return LLMClient(provider=provider)


def call_llm(*,
             model: str,
             messages: Optional[List[Dict[str, str]]] = None,
             system_prompt: Optional[str] = None,
             user_content: Optional[str] = None,
             response_model: Optional[Type[Any]] = None,
             **kwargs):
    """
    Backward compatibility function for utils.llm.call_llm().
    """
    client = LLMClient()

    # Build messages
    if not messages:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_content:
            messages.append({"role": "user", "content": user_content})

    if response_model:
        response = client.structured_completion(
            messages=messages,
            response_model=response_model,
            model=model,
            **kwargs
        )
        return response.parsed_data, response.raw_content
    else:
        response = client.chat_completion(
            messages=messages,
            model=model,
            **kwargs
        )
        return None, response.content


def run_ontology_llm(*,
                     model: str,
                     user_content: str,
                     response_model: Type[Any],
                     system_prompt: Optional[str] = None,
                     **kwargs):
    """
    Backward compatibility function for utils.llm.run_ontology_llm().
    """
    client = LLMClient()
    response = client.ontology_completion(
        user_content=user_content,
        response_model=response_model,
        system_prompt=system_prompt,
        model=model,
        **kwargs
    )
    return response.parsed_data, response.raw_content
