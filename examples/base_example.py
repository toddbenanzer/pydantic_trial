"""
Usage examples for the LLM interface package.
"""
import asyncio
import os
from pathlib import Path

# With the refactored structure, imports remain the same as the package 
# exports all providers from its top-level namespace
from llm_interface import (
    create_provider,
    OpenAIProvider,
    AnthropicProvider,
    AWSProvider,
    HuggingFaceProvider,
    ReferenceFile
)

# Example 1: Simple text generation with OpenAI
async def example_openai_text():
    provider = OpenAIProvider()
    response = await provider.generate_text(
        prompt="Explain quantum computing in simple terms.",
        system_message="You are a helpful assistant that explains complex topics in simple language."
    )
    print("OpenAI Text Generation:")
    print(response.content)
    print("\n")

# Example 2: Code generation with Anthropic
async def example_anthropic_code():
    provider = AnthropicProvider()
    response = await provider.generate_code(
        prompt="Write a Python function that calculates the Fibonacci sequence up to n terms.",
        system_message="You are a helpful coding assistant that writes clean and efficient code."
    )
    print("Anthropic Code Generation:")
    print(response.content)
    print("\n")

# Example 3: Text to speech with AWS
async def example_aws_tts():
    provider = AWSProvider()
    response = await provider.text_to_speech(
        text="Hello world! This is a text to speech example using AWS Polly.",
        voice_id="Matthew"
    )
    # Save audio to file
    output_path = Path("aws_tts_example.mp3")
    with open(output_path, "wb") as f:
        f.write(response.audio_data)
    print(f"AWS Text to Speech: Audio saved to {output_path}")
    print("\n")

# Example 4: Using reference files with OpenAI
async def example_openai_with_references():
    provider = OpenAIProvider()
    
    # Create a sample reference file
    ref_file_path = Path("sample_data.txt")
    with open(ref_file_path, "w") as f:
        f.write("The capital of France is Paris.\nThe capital of Japan is Tokyo.\nThe capital of Egypt is Cairo.")
    
    # Add reference file
    ref_file = await provider.add_reference_file(ref_file_path)
    
    # Generate text with reference
    response = await provider.generate_text(
        prompt="What is the capital of Japan?",
        reference_files=[ref_file]
    )
    
    print("OpenAI with Reference Files:")
    print(response.content)
    print("\n")
    
    # Clean up
    os.remove(ref_file_path)

# Example 5: HuggingFace text generation
async def example_huggingface_text():
    provider = HuggingFaceProvider()
    response = await provider.generate_text(
        prompt="Write a short poem about artificial intelligence.",
        model="mistralai/Mistral-7B-Instruct-v0.2"
    )
    print("HuggingFace Text Generation:")
    print(response.content)
    print("\n")

# Example 6: Using factory function
async def example_factory_function():
    # Create OpenAI provider
    openai_provider = create_provider("openai")
    
    # Create Anthropic provider with custom configuration
    anthropic_provider = create_provider(
        "anthropic", 
        default_model="claude-3-sonnet-20240229"
    )
    
    # Use the providers
    response1 = await openai_provider.generate_text(
        prompt="What is the meaning of life?",
        max_tokens=100
    )
    
    response2 = await anthropic_provider.generate_text(
        prompt="How does machine learning work?",
        max_tokens=150
    )
    
    print("Factory Function Example:")
    print("OpenAI response:", response1.content[:100] + "...")
    print("Anthropic response:", response2.content[:100] + "...")
    print("\n")

# Run examples
async def run_examples():
    print("Running examples...\n")
    
    try:
        await example_openai_text()
    except Exception as e:
        print(f"Error in OpenAI text example: {e}")
    
    try:
        await example_anthropic_code()
    except Exception as e:
        print(f"Error in Anthropic code example: {e}")
    
    try:
        await example_aws_tts()
    except Exception as e:
        print(f"Error in AWS TTS example: {e}")
    
    try:
        await example_openai_with_references()
    except Exception as e:
        print(f"Error in OpenAI with references example: {e}")
    
    try:
        await example_huggingface_text()
    except Exception as e:
        print(f"Error in HuggingFace text example: {e}")
    
    try:
        await example_factory_function()
    except Exception as e:
        print(f"Error in factory function example: {e}")
    
    print("Examples completed!")

if __name__ == "__main__":
    asyncio.run(run_examples())