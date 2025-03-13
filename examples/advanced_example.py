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
                except NotImplementedError:
                    continue
            
            # If no provider supports TTS, raise error
            raise NotImplementedError("No configured provider supports text-to-speech")

# Example usage of the SmartAgent
async def example_smart_agent():
    # Create providers
    openai_provider = create_provider("openai")
    anthropic_provider = create_provider("anthropic")
    
    # Create agent with multiple providers
    agent = SmartAgent(
        name="Research Assistant",
        description="A smart assistant that can answer questions and generate code",
        providers={
            "openai": openai_provider,
            "anthropic": anthropic_provider
        },
        default_provider="openai"
    )
    
    # Sample reference document
    ref_file_path = Path("research_data.txt")
    with open(ref_file_path, "w") as f:
        f.write("The population of France in 2023 was approximately 68 million people.\n"
                "The capital city is Paris with a population of about 2.1 million.\n"
                "France is known for its cuisine, art, and the Eiffel Tower.")
    
    # Add reference file
    await agent.add_reference_file(ref_file_path)
    
    # Ask a question using the default provider (OpenAI)
    response1 = await agent.ask(
        prompt="What is the population of France?",
        system_message="You are a helpful research assistant."
    )
    print(f"Response using OpenAI:\n{response1.content}\n")
    
    # Ask a follow-up question (should include context from previous interaction)
    response2 = await agent.ask(
        prompt="And what is its capital city?",
        system_message="You are a helpful research assistant."
    )
    print(f"Follow-up response with context:\n{response2.content}\n")
    
    # Generate code using Anthropic
    code_response = await agent.generate_code(
        prompt="Write a Python function to calculate the factorial of a number.",
        provider="anthropic"
    )
    print(f"Code generation using Anthropic:\n{code_response.content}\n")
    
    # Clean up
    os.remove(ref_file_path)

if __name__ == "__main__":
    asyncio.run(example_smart_agent())