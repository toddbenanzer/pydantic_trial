"""
Anthropic provider implementation.
"""
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

from ..base import LLMProvider, LLMResponse, AudioResponse, ReferenceFile
from ..config import get_credentials

class AnthropicProviderConfig(BaseModel):
    """Configuration for Anthropic provider"""
    api_key: str
    default_model: str = "claude-3-opus-20240229"

class AnthropicProvider(LLMProvider):
    """Anthropic API provider implementation"""
    
    def __init__(
        self, 
        config: Optional[AnthropicProviderConfig] = None,
        api_key: Optional[str] = None,
        default_model: str = "claude-3-opus-20240229"
    ):
        """
        Initialize Anthropic provider.
        
        Args:
            config: Provider configuration
            api_key: Anthropic API key (overrides config)
            default_model: Default model to use (overrides config)
        """
        # Get credentials if not provided
        if not config and not api_key:
            credentials = get_credentials("anthropic")
            api_key = credentials.get("api_key")
        
        # Use config if provided
        if config:
            api_key = api_key or config.api_key
            default_model = default_model or config.default_model
            
        # Create client
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key
        )
        
        self.default_model = default_model
        
    async def generate_text(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        reference_files: Optional[List[ReferenceFile]] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text response from Anthropic.
        
        Args:
            prompt: User prompt
            system_message: System instructions
            reference_files: List of reference files
            model: Model to use (default: claude-3-opus-20240229)
            **kwargs: Additional parameters for API
            
        Returns:
            LLMResponse with generated text
        """
        model = model or self.default_model
        
        # Prepare messages
        messages = [{"role": "user", "content": prompt}]
        
        # Add reference files as text content if provided
        if reference_files:
            combined_prompt = prompt
            
            for ref_file in reference_files:
                # Read file content as text
                try:
                    file_content = ref_file.read_text()
                    combined_prompt += f"\n\nReference file ({ref_file.path.name}):\n{file_content}"
                except Exception as e:
                    print(f"Warning: Could not read file {ref_file.path}: {e}")
            
            # Update the prompt with reference materials
            messages = [{"role": "user", "content": combined_prompt}]
        
        # Make API call
        response = await self.client.messages.create(
            model=model,
            messages=messages,
            system=system_message if system_message else None,
            **kwargs
        )
        
        # Extract and return response
        content = response.content[0].text
        return LLMResponse(
            content=content,
            model=model,
            raw_response=response
        )
    
    async def generate_code(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        reference_files: Optional[List[ReferenceFile]] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate code response from Anthropic.
        
        Args:
            prompt: User prompt for code generation
            system_message: System instructions
            reference_files: List of reference files
            model: Model to use
            **kwargs: Additional parameters for API
            
        Returns:
            LLMResponse with generated code
        """
        # Add code-specific instructions if no system message provided
        if not system_message:
            system_message = "You are a helpful AI assistant focused on generating clean, efficient, and well-documented code. Always provide code in markdown format with appropriate language tags."
        
        # Use the text generation with code-specific instructions
        return await self.generate_text(
            prompt=prompt,
            system_message=system_message,
            reference_files=reference_files,
            model=model,
            **kwargs
        )
    
    async def text_to_speech(
        self, 
        text: str, 
        **kwargs
    ) -> AudioResponse:
        """
        Convert text to speech.
        Not implemented for Anthropic.
        
        Args:
            text: Text to convert to speech
            **kwargs: Additional parameters
            
        Raises:
            NotImplementedError: Feature not supported
        """
        self.not_implemented("Text to speech")
        
    async def add_reference_file(
        self, 
        file_path: Union[str, Path],
        **kwargs
    ) -> ReferenceFile:
        """
        Add reference file.
        Anthropic doesn't support direct file uploads via API, so we'll just create
        a ReferenceFile object for local processing.
        
        Args:
            file_path: Path to file
            **kwargs: Additional parameters
            
        Returns:
            ReferenceFile without file_id
        """
        file_path = Path(file_path)
        
        # Create reference file
        ref_file = ReferenceFile(
            path=file_path,
            content_type=None
        )
        
        return ref_file