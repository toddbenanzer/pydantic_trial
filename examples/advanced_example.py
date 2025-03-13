"""
Advanced usage examples for the LLM interface package.
Includes a smart agent implementation that combines multiple capabilities.
"""
import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from pydantic import BaseModel, Field

from llm_interface import (
    create_provider,
    LLMProvider,
    LLMResponse,
    ReferenceFile,
    AudioResponse
)

class SmartAgent(BaseModel):
    """
    Smart agent that combines multiple LLM capabilities.
    Can use different providers for different tasks and maintain context.
    """
    name: str
    description: str
    providers: Dict[str, LLMProvider]
    default_provider: str
    context: List[Dict[str, str]] = Field(default_factory=list)
    reference_files: List[ReferenceFile] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
    
    async def add_reference_file(self, file_path: Union[str, Path], provider: Optional[str] = None) -> ReferenceFile:
        """Add reference file to agent context"""
        provider_name = provider or self.default_provider
        provider = self.providers[provider_name]
        
        ref_file = await provider.add_reference_file(file_path)
        self.reference_files.append(ref_file)
        return ref_file
    
    async def ask(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        provider: Optional[str] = None,
        include_context: bool = True,
        **kwargs
    ) -> LLMResponse:
        """Ask the agent a question"""
        provider_name = provider or self.default_provider
        provider = self.providers[provider_name]
        
        # Combine with conversation context if requested
        if include_context and self.context:
            context_str = "\n\nPrevious conversation:\n"
            for message in self.context:
                role = message["role"]
                content = message["content"]
                context_str += f"{role}: {content}\n"
            
            prompt = context_str + f"\n\nCurrent question: {prompt}"
            
        # Generate response
        response = await provider.generate_text(
            prompt=prompt,
            system_message=system_message,
            reference_files=self.reference_files if include_context else None,
            **kwargs
        )
        
        # Update context
        self.context.append({"role": "user", "content": prompt})
        self.context.append({"role": "assistant", "content": response.content})
        
        return response
    
    async def generate_code(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate code using the agent"""
        provider_name = provider or self.default_provider
        provider = self.providers[provider_name]
        
        response = await provider.generate_code(
            prompt=prompt,
            system_message=system_message,
            reference_files=self.reference_files,
            **kwargs
        )
        
        # Update context
        self.context.append({"role": "user", "content": f"Generate code: {prompt}"})
        self.context.append({"role": "assistant", "content": response.content})
        
        return response
    
    async def text_to_speech(
        self, 
        text: str, 
        provider: Optional[str] = None,
        **kwargs
    ) -> AudioResponse:
        """Convert text to speech using the agent"""
        provider_name = provider or self.default_provider
        provider = self.providers[provider_name]
        
        try:
            return await provider.text_to_speech(text, **kwargs)
        except NotImplementedError:
            # Try to find a provider that supports TTS
            for name, prov in self.providers.items():
                try:
                    return await prov.text_to_speech(text, **kwargs)