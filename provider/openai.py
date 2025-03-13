"""
OpenAI provider implementation.
"""
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import asyncio

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from .base import LLMProvider, LLMResponse, AudioResponse, ReferenceFile, SystemMessage, UserMessage
from .config import get_credentials

class OpenAIProviderConfig(BaseModel):
    """Configuration for OpenAI provider"""
    api_key: str
    organization: Optional[str] = None
    default_model: str = "gpt-4o"
    default_tts_model: str = "tts-1"
    default_tts_voice: str = "alloy"

class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation"""
    
    def __init__(
        self, 
        config: Optional[OpenAIProviderConfig] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        default_model: str = "gpt-4o",
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            config: Provider configuration
            api_key: OpenAI API key (overrides config)
            organization: OpenAI organization ID (overrides config)
            default_model: Default model to use (overrides config)
        """
        # Get credentials if not provided
        if not config and not api_key:
            credentials = get_credentials("openai")
            api_key = credentials.get("api_key")
            organization = credentials.get("organization")
        
        # Use config if provided
        if config:
            api_key = api_key or config.api_key
            organization = organization or config.organization
            default_model = default_model or config.default_model
            
        # Create client
        self.client = AsyncOpenAI(
            api_key=api_key,
            organization=organization
        )
        
        self.default_model = default_model
        self.default_tts_model = "tts-1"
        self.default_tts_voice = "alloy"
        self.files = {}  # Store reference file IDs
        
    async def generate_text(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        reference_files: Optional[List[ReferenceFile]] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text response from OpenAI.
        
        Args:
            prompt: User prompt
            system_message: System instructions
            reference_files: List of reference files
            model: Model to use (default: gpt-4o)
            **kwargs: Additional parameters for API
            
        Returns:
            LLMResponse with generated text
        """
        model = model or self.default_model
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        # Add file references if any
        file_ids = []
        if reference_files:
            for ref_file in reference_files:
                if not ref_file.file_id:
                    ref_file = await self.add_reference_file(ref_file.path)
                file_ids.append(ref_file.file_id)
        
        # Make API call
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            file_ids=file_ids if file_ids else None,
            **kwargs
        )
        
        # Extract and return response
        content = response.choices[0].message.content
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
        Generate code response from OpenAI.
        
        Args:
            prompt: User prompt for code generation
            system_message: System instructions
            reference_files: List of reference files
            model: Model to use (default: gpt-4o)
            **kwargs: Additional parameters for API
            
        Returns:
            LLMResponse with generated code
        """
        # Add code-specific instructions if no system message provided
        if not system_message:
            system_message = "You are a helpful AI assistant focused on generating clean, efficient, and well-documented code."
        
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
        model: Optional[str] = None,
        voice: Optional[str] = None,
        **kwargs
    ) -> AudioResponse:
        """
        Convert text to speech using OpenAI TTS.
        
        Args:
            text: Text to convert to speech
            model: TTS model to use (default: tts-1)
            voice: Voice to use (default: alloy)
            **kwargs: Additional parameters for API
            
        Returns:
            AudioResponse with audio data
        """
        model = model or self.default_tts_model
        voice = voice or self.default_tts_voice
        
        response = await self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            **kwargs
        )
        
        # Get audio data
        audio_data = await response.read()
        
        return AudioResponse(
            content=f"Audio generated using model {model} with voice {voice}",
            model=model,
            audio_data=audio_data,
            raw_response=None
        )
        
    async def add_reference_file(
        self, 
        file_path: Union[str, Path],
        purpose: str = "assistants",
        **kwargs
    ) -> ReferenceFile:
        """
        Add reference file to OpenAI.
        
        Args:
            file_path: Path to file
            purpose: File purpose (default: assistants)
            **kwargs: Additional parameters for API
            
        Returns:
            ReferenceFile with file_id
        """
        file_path = Path(file_path)
        
        # Upload file
        with open(file_path, "rb") as file:
            response = await self.client.files.create(
                file=file,
                purpose=purpose,
                **kwargs
            )
        
        # Create and return reference file
        ref_file = ReferenceFile(
            path=file_path,
            file_id=response.id,
            content_type=response.content_type
        )
        
        # Store for future reference
        self.files[str(file_path)] = ref_file
        
        return ref_file