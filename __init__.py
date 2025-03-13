"""
Pydantic-based LLM interface package
Provides a unified interface to various LLM providers
"""
from .base import (
    LLMProvider, 
    LLMResponse,
    AudioResponse, 
    ReferenceFile,
    Message,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ContentType
)

from .config import get_credentials, load_config

# Import providers
from .provider.openai import OpenAIProvider, OpenAIProviderConfig
from .provider.anthropic import AnthropicProvider, AnthropicProviderConfig
from .provider.aws import AWSProvider, AWSProviderConfig
from .provider.huggingface import HuggingFaceProvider, HuggingFaceProviderConfig

__version__ = "0.1.0"

# Factory function to create an LLM provider
def create_provider(provider_name: str, **kwargs):
    """
    Create an LLM provider by name.
    
    Args:
        provider_name: Name of the provider (openai, anthropic, aws, huggingface)
        **kwargs: Additional parameters for provider initialization
        
    Returns:
        LLMProvider instance
    """
    provider_map = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "aws": AWSProvider,
        "huggingface": HuggingFaceProvider
    }
    
    if provider_name not in provider_map:
        raise ValueError(f"Unsupported provider: {provider_name}")
        
    return provider_map[provider_name](**kwargs)