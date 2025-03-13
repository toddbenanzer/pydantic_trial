"""
HuggingFace provider implementation.
Supports interaction with HuggingFace's Inference API for various models.
"""
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
import httpx
import asyncio
import base64

from pydantic import BaseModel, Field

from ..base import LLMProvider, LLMResponse, AudioResponse, ReferenceFile
from ..config import get_credentials

class HuggingFaceProviderConfig(BaseModel):
    """Configuration for HuggingFace provider"""
    api_token: str
    default_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    api_base: str = "https://api-inference.huggingface.co/models"
    default_tts_model: str = "espnet/kan-bayashi_ljspeech_vits"

class HuggingFaceProvider(LLMProvider):
    """HuggingFace API provider implementation"""
    
    def __init__(
        self, 
        config: Optional[HuggingFaceProviderConfig] = None,
        api_token: Optional[str] = None,
        default_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        api_base: str = "https://api-inference.huggingface.co/models"
    ):
        """
        Initialize HuggingFace provider.
        
        Args:
            config: Provider configuration
            api_token: HuggingFace API token (overrides config)
            default_model: Default model to use (overrides config)
            api_base: API base URL (overrides config)
        """
        # Get credentials if not provided
        if not config and not api_token:
            credentials = get_credentials("huggingface")
            api_token = credentials.get("api_token")
        
        # Use config if provided
        if config:
            api_token = api_token or config.api_token
            default_model = default_model or config.default_model
            api_base = api_base or config.api_base
        
        if not api_token:
            raise ValueError("HuggingFace API token is required")
            
        self.api_token = api_token
        self.default_model = default_model
        self.api_base = api_base
        self.default_tts_model = "espnet/kan-bayashi_ljspeech_vits"
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.api_token}"},
            timeout=60.0  # Some models can take time to generate responses
        )
        
    async def _query_inference_api(
        self, 
        model: str, 
        payload: Dict[str, Any]
    ) -> Any:
        """
        Query HuggingFace Inference API.
        
        Args:
            model: Model ID to use
            payload: Request payload
            
        Returns:
            Response from API
        """
        url = f"{self.api_base}/{model}"
        
        try:
            response = await self.client.post(url, json=payload)
            
            # Handle rate limiting or other errors
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "5"))
                print(f"Rate limited, retrying after {retry_after} seconds...")
                await asyncio.sleep(retry_after)
                return await self._query_inference_api(model, payload)
                
            # Handle other errors
            response.raise_for_status()
            
            try:
                return response.json()
            except json.JSONDecodeError:
                # Some endpoints return binary data (like TTS)
                return response.content
                
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error from HuggingFace API: {e.response.status_code} {e.response.text}"
            print(error_msg)
            raise ValueError(error_msg)
            
        except httpx.RequestError as e:
            error_msg = f"Request error with HuggingFace API: {str(e)}"
            print(error_msg)
            raise ValueError(error_msg)
    
    def _format_chat_prompt(
        self, 
        prompt: str,
        system_message: Optional[str] = None,
        reference_files: Optional[List[ReferenceFile]] = None
    ) -> str:
        """
        Format prompt for chat models.
        
        Args:
            prompt: User prompt
            system_message: System instructions
            reference_files: List of reference files
            
        Returns:
            Formatted prompt
        """
        formatted_prompt = ""
        
        # Add system message if provided
        if system_message:
            formatted_prompt = f"<s>[INST] {system_message} [/INST]</s>\n\n"
        
        # Add reference file content if provided
        if reference_files:
            reference_content = ""
            for ref_file in reference_files:
                try:
                    file_content = ref_file.read_text()
                    reference_content += f"\nReference file ({ref_file.path.name}):\n{file_content}\n"
                except Exception as e:
                    print(f"Warning: Could not read file {ref_file.path}: {e}")
                    
            # Add references to prompt
            if reference_content:
                prompt = f"Reference materials:\n{reference_content}\n\nBased on the above references, please respond to this: {prompt}"
        
        # Format for Mistral or Llama style chat
        # This format works with many models like Mistral, Llama, etc.
        if not formatted_prompt:
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        else:
            formatted_prompt += f"<s>[INST] {prompt} [/INST]"
            
        return formatted_prompt
        
    async def generate_text(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        reference_files: Optional[List[ReferenceFile]] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text response from HuggingFace Inference API.
        
        Args:
            prompt: User prompt
            system_message: System instructions
            reference_files: List of reference files
            model: Model to use (default: mistralai/Mistral-7B-Instruct-v0.2)
            **kwargs: Additional parameters for API
            
        Returns:
            LLMResponse with generated text
        """
        model = model or self.default_model
        
        # Check if this is a chat model (Mistral, Llama, etc.)
        is_chat_model = any(name in model.lower() for name in ["mistral", "llama", "zephyr", "falcon"])
        
        if is_chat_model:
            # Format for chat-style models
            formatted_prompt = self._format_chat_prompt(
                prompt=prompt,
                system_message=system_message,
                reference_files=reference_files
            )
        else:
            # Format for completion-style models
            formatted_prompt = prompt
            
            # Add system message if provided
            if system_message:
                formatted_prompt = f"{system_message}\n\n{formatted_prompt}"
                
            # Add reference file content if provided
            if reference_files:
                for ref_file in reference_files:
                    try:
                        file_content = ref_file.read_text()
                        formatted_prompt += f"\n\nReference file ({ref_file.path.name}):\n{file_content}"
                    except Exception as e:
                        print(f"Warning: Could not read file {ref_file.path}: {e}")
        
        # Prepare payload
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "do_sample": kwargs.get("do_sample", True)
            }
        }
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in ["max_tokens", "temperature", "top_p", "do_sample"]:
                payload["parameters"][key] = value
        
        # Query API
        response = await self._query_inference_api(model, payload)
        
        # Parse response based on model type
        if isinstance(response, list) and len(response) > 0:
            # Most completion models return a list of generated text options
            if "generated_text" in response[0]:
                content = response[0]["generated_text"]
            else:
                content = response[0]
        elif isinstance(response, dict):
            # Some models return a single dict with generated_text
            content = response.get("generated_text", str(response))
        else:
            # Fallback to stringifying the response
            content = str(response)
        
        # For chat models, try to extract just the assistant's part
        if is_chat_model and "[/INST]" in formatted_prompt and "[/INST]" in content:
            try:
                content = content.split("[/INST]")[1].strip()
            except (IndexError, AttributeError):
                # Keep the original content if extraction fails
                pass
                
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
        Generate code response from HuggingFace.
        
        Args:
            prompt: User prompt for code generation
            system_message: System instructions
            reference_files: List of reference files
            model: Model to use
            **kwargs: Additional parameters for API
            
        Returns:
            LLMResponse with generated code
        """
        # Use CodeLlama or similar model if none specified
        code_model = model or "codellama/CodeLlama-7b-hf"
        
        # Add code-specific instructions if no system message provided
        if not system_message:
            system_message = "You are a helpful assistant focused on writing clean, efficient, and well-documented code. Provide code with explanations when needed."
        
        # Ensure prompt clearly asks for code
        if "code" not in prompt.lower() and "function" not in prompt.lower():
            prompt = f"Generate code for this task: {prompt}"
        
        # Use the text generation with code-specific model
        return await self.generate_text(
            prompt=prompt,
            system_message=system_message,
            reference_files=reference_files,
            model=code_model,
            max_tokens=kwargs.get("max_tokens", 1024),  # Longer context for code
            temperature=kwargs.get("temperature", 0.2),  # Lower temperature for more deterministic code
            **kwargs
        )
    
    async def text_to_speech(
        self, 
        text: str, 
        model: Optional[str] = None,
        **kwargs
    ) -> AudioResponse:
        """
        Convert text to speech using HuggingFace TTS models.
        
        Args:
            text: Text to convert to speech
            model: TTS model to use
            **kwargs: Additional parameters
            
        Returns:
            AudioResponse with audio data
        """
        model = model or self.default_tts_model
        
        # Prepare payload
        payload = {"inputs": text}
        
        # Add speaker_id if provided (for multi-speaker models)
        if "speaker_id" in kwargs:
            payload["speaker_id"] = kwargs["speaker_id"]
        
        try:
            # Query API - TTS endpoints usually return binary audio data
            audio_data = await self._query_inference_api(model, payload)
            
            return AudioResponse(
                content=f"Audio generated using HuggingFace model {model}",
                model=model,
                audio_data=audio_data,
                raw_response=None
            )
        except Exception as e:
            error_msg = f"Text-to-speech failed with HuggingFace model {model}: {str(e)}"
            print(error_msg)
            raise ValueError(error_msg)
        
    async def add_reference_file(
        self, 
        file_path: Union[str, Path],
        **kwargs
    ) -> ReferenceFile:
        """
        Add reference file.
        HuggingFace doesn't support direct file uploads in the same way as OpenAI,
        so we'll create a ReferenceFile object for local processing.
        
        Args:
            file_path: Path to file
            **kwargs: Additional parameters
            
        Returns:
            ReferenceFile without file_id
        """
        file_path = Path(file_path)
        
        # Determine content type based on file extension
        content_type = None
        if file_path.suffix.lower() == '.txt':
            content_type = "text/plain"
        elif file_path.suffix.lower() == '.md':
            content_type = "text/markdown"
        elif file_path.suffix.lower() in ['.json', '.jsonl']:
            content_type = "application/json"
        elif file_path.suffix.lower() in ['.csv', '.tsv']:
            content_type = "text/csv"
        elif file_path.suffix.lower() in ['.py', '.js', '.java', '.c', '.cpp']:
            content_type = "text/x-code"
        
        # Create reference file
        ref_file = ReferenceFile(
            path=file_path,
            content_type=content_type
        )
        
        return ref_file
    
    def __del__(self):
        """Clean up resources when object is deleted"""
        # Close the HTTP client
        try:
            if hasattr(self, 'client') and self.client:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.client.aclose())
                else:
                    loop.run_until_complete(self.client.aclose())
        except Exception:
            pass  # Ignore errors during cleanup