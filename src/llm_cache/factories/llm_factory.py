import os

from llm_cache.config.app_config import ConfigError, LLMConfig
from llm_cache.llm.groq_llm_provider import GroqLLMProvider
from llm_cache.llm.i_llm_provider import ILLMProvider
from llm_cache.llm.ollama_llm_provider import OllamaLLMProvider

_SUPPORTED_LLM_PROVIDERS = ("ollama", "groq")


def _normalize_provider(provider: str) -> str:
    return provider.strip().lower()


def create_llm_provider(config: LLMConfig) -> ILLMProvider:
    provider = _normalize_provider(config.provider)

    if provider == "ollama":
        if not config.model:
            raise ConfigError(
                "Missing model for Ollama LLM provider. "
                "Example: LLMConfig(provider='ollama', model='gemma3:4b')"
            )

        return OllamaLLMProvider(model_name=config.model)

    if provider == "groq":
        if not config.model:
            raise ConfigError(
                "Missing model for Groq LLM provider. "
                "Example: LLMConfig(provider='groq', model='llama-3.1-8b-instant')"
            )

        api_key_env = config.api_key_env or "GROQ_API_KEY"
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ConfigError(
                f"Missing Groq API key. Set environment variable {api_key_env} "
                "or configure LLMConfig.api_key_env."
            )

        return GroqLLMProvider(model_name=config.model, api_key=api_key)

    raise ConfigError(
        f"Unknown LLM provider '{config.provider}'. "
        f"Supported LLM providers: {', '.join(_SUPPORTED_LLM_PROVIDERS)}"
    )
