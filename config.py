"""
Configuration module for LLM interfaces.
Provides centralized credential management and configuration.
"""
from typing import Dict, Any, Optional
import os
import json
from pathlib import Path

# Default configuration path
DEFAULT_CONFIG_PATH = Path.home() / ".llmconfig.json"


def get_credentials(provider: str) -> Dict[str, str]:
    """
    Get credentials for a specific provider.
    Override this function to customize credential retrieval.
    
    Args:
        provider: Name of the provider (openai, anthropic, aws, huggingface)
        
    Returns:
        Dict containing credentials for the provider
    """
    # First check environment variables
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            return {"api_key": api_key}
            
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            return {"api_key": api_key}
            
    elif provider == "aws":
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        region = os.environ.get("AWS_REGION", "us-east-1")
        if access_key and secret_key:
            return {
                "aws_access_key_id": access_key,
                "aws_secret_access_key": secret_key,
                "region_name": region
            }
            
    elif provider == "huggingface":
        api_key = os.environ.get("HUGGINGFACE_API_KEY")
        if api_key:
            return {"api_token": api_key}
    
    # If not in environment variables, try config file
    config = load_config()
    provider_config = config.get(provider, {})
    
    if not provider_config:
        raise ValueError(f"No credentials found for provider: {provider}")
        
    return provider_config