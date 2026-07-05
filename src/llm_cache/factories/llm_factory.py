import os

from llm_cache.config.app_config import ConfigError, LLMConfig
from llm_cache.config.provider_options import (
    SUPPORTED_LLM_PROVIDERS,
    normalize_provider_name,
)
from llm_cache.llm.interface import ILLMProvider


def create_llm_provider(config: LLMConfig) -> ILLMProvider:
    provider = normalize_provider_name(config.provider)

    if provider == "llm-provider-spy":
        from llm_cache.test_doubles.llm_provider_spy import LLMProviderSpy

        return LLMProviderSpy()

    if provider == "ollama":
        if not config.model:
            raise ConfigError(
                "Missing model for Ollama LLM provider. "
                "Example: LLMConfig(provider='ollama', model='gemma3:4b')"
            )

        from llm_cache.llm.providers.ollama import OllamaLLMProvider

        return OllamaLLMProvider(model_name=config.model, base_url=config.base_url)

    if provider == "groq":
        if not config.model:
            raise ConfigError(
                "Missing model for Groq LLM provider. "
                "Example: LLMConfig(provider='groq', model='llama-3.1-8b-instant')"
            )

        api_key_env = config.groq_api_key_env or "GROQ_API_KEY"
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ConfigError(
                f"Missing Groq API key. Set environment variable {api_key_env} "
                f"by running: export {api_key_env}='your_api_key'"
            )

        from llm_cache.llm.providers.groq import GroqLLMProvider

        return GroqLLMProvider(
            model_name=config.model,
            api_key=api_key,
            api_key_env=api_key_env,
        )

    raise ConfigError(
        f"Unknown LLM provider '{config.provider}'. "
        f"Supported LLM providers: {', '.join(SUPPORTED_LLM_PROVIDERS)}"
    )
