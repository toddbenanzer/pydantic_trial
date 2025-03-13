"""
AWS Bedrock provider implementation.
Supports various models through AWS Bedrock service including:
- Anthropic Claude models
- Amazon Titan models
- Meta Llama models
- AI21 Jurassic models
"""
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
import base64

import boto3
from pydantic import BaseModel, Field

from ..base import LLMProvider, LLMResponse, AudioResponse, ReferenceFile
from ..config import get_credentials

class AWSProviderConfig(BaseModel):
    """Configuration for AWS Bedrock provider"""
    aws_access_key_id: str
    aws_secret_access_key: str
    region_name: str = "us-east-1"
    default_model_id: str = "anthropic.claude-3-opus-20240229"
    default_tts_model_id: str = "amazon.polly"

class AWSProvider(LLMProvider):
    """AWS Bedrock provider implementation"""
    
    def __init__(
        self,
        config: Optional[AWSProviderConfig] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
        default_model_id: str = "anthropic.claude-3-opus-20240229"
    ):
        """
        Initialize AWS Bedrock provider.
        
        Args:
            config: Provider configuration
            aws_access_key_id: AWS access key (overrides config)
            aws_secret_access_key: AWS secret key (overrides config)
            region_name: AWS region (overrides config)
            default_model_id: Default model ID to use (overrides config)
        """
        # Get credentials if not provided
        if not config and not (aws_access_key_id and aws_secret_access_key):
            credentials = get_credentials("aws")
            aws_access_key_id = credentials.get("aws_access_key_id")
            aws_secret_access_key = credentials.get("aws_secret_access_key")
            region_name = credentials.get("region_name", "us-east-1")
        
        # Use config if provided
        if config:
            aws_access_key_id = aws_access_key_id or config.aws_access_key_id
            aws_secret_access_key = aws_secret_access_key or config.aws_secret_access_key
            region_name = region_name or config.region_name
            default_model_id = default_model_id or config.default_model_id
        
        # Create client
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name or "us-east-1",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        self.default_model_id = default_model_id
        self.default_tts_model_id = "amazon.polly"
        
        # Map of model vendors to request formatters
        self.model_formatters = {
            "anthropic": self._format_anthropic_request,
            "amazon": self._format_titan_request,
            "meta": self._format_meta_request,
            "ai21": self._format_ai21_request
        }
        
        # Map of model vendors to response parsers
        self.response_parsers = {
            "anthropic": self._parse_anthropic_response,
            "amazon": self._parse_titan_response,
            "meta": self._parse_meta_response,
            "ai21": self._parse_ai21_response
        }
    
    def _get_vendor(self, model_id: str) -> str:
        """Get vendor name from model ID"""
        return model_id.split(".")[0]
    
    def _format_anthropic_request(
        self, 
        prompt: str,
        system_message: Optional[str] = None,
        reference_files: Optional[List[ReferenceFile]] = None,
        **kwargs
    ) -> Dict:
        """Format request for Anthropic models"""
        messages = [{"role": "user", "content": prompt}]
        
        # Add reference file content if provided
        if reference_files:
            combined_prompt = prompt
            for ref_file in reference_files:
                try:
                    file_content = ref_file.read_text()
                    combined_prompt += f"\n\nReference file ({ref_file.path.name}):\n{file_content}"
                except Exception as e:
                    print(f"Warning: Could not read file {ref_file.path}: {e}")
            
            messages = [{"role": "user", "content": combined_prompt}]
        
        request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": kwargs.get("max_tokens", 1024),
            "messages": messages
        }
        
        # Add system message if provided
        if system_message:
            request["system"] = system_message
        
        # Add temperature if provided
        if "temperature" in kwargs:
            request["temperature"] = kwargs["temperature"]
            
        return request
        
    def _format_titan_request(
        self, 
        prompt: str,
        system_message: Optional[str] = None,
        reference_files: Optional[List[ReferenceFile]] = None,
        **kwargs
    ) -> Dict:
        """Format request for Amazon Titan models"""
        formatted_prompt = prompt
        
        # Add system message if provided
        if system_message:
            formatted_prompt = f"{system_message}\n\n{prompt}"
        
        # Add reference file content if provided
        if reference_files:
            for ref_file in reference_files:
                try:
                    file_content = ref_file.read_text()
                    formatted_prompt += f"\n\nReference file ({ref_file.path.name}):\n{file_content}"
                except Exception as e:
                    print(f"Warning: Could not read file {ref_file.path}: {e}")
        
        request = {
            "inputText": formatted_prompt,
            "textGenerationConfig": {
                "maxTokenCount": kwargs.get("max_tokens", 1024),
                "temperature": kwargs.get("temperature", 0.7),
                "topP": kwargs.get("top_p", 0.9)
            }
        }
        
        return request
        
    def _format_meta_request(
        self, 
        prompt: str,
        system_message: Optional[str] = None,
        reference_files: Optional[List[ReferenceFile]] = None,
        **kwargs
    ) -> Dict:
        """Format request for Meta models"""
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        # Add reference file content if provided
        if reference_files:
            combined_prompt = prompt
            for ref_file in reference_files:
                try:
                    file_content = ref_file.read_text()
                    combined_prompt += f"\n\nReference file ({ref_file.path.name}):\n{file_content}"
                except Exception as e:
                    print(f"Warning: Could not read file {ref_file.path}: {e}")
            
            messages = [{"role": "system", "content": system_message}] if system_message else []
            messages.append({"role": "user", "content": combined_prompt})
        
        request = {
            "messages": messages,
            "max_gen_len": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9)
        }
        
        return request
        
    def _format_ai21_request(
        self, 
        prompt: str,
        system_message: Optional[str] = None,
        reference_files: Optional[List[ReferenceFile]] = None,
        **kwargs
    ) -> Dict:
        """Format request for AI21 models"""
        formatted_prompt = prompt
        
        # Add system message if provided
        if system_message:
            formatted_prompt = f"{system_message}\n\n{prompt}"
        
        # Add reference file content if provided
        if reference_files:
            for ref_file in reference_files:
                try:
                    file_content = ref_file.read_text()
                    formatted_prompt += f"\n\nReference file ({ref_file.path.name}):\n{file_content}"
                except Exception as e:
                    print(f"Warning: Could not read file {ref_file.path}: {e}")
        
        request = {
            "prompt": formatted_prompt,
            "maxTokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
            "topP": kwargs.get("top_p", 0.9)
        }
        
        return request
    
    def _parse_anthropic_response(self, response: Dict) -> str:
        """Parse response from Anthropic models"""
        body = json.loads(response.get("body").read().decode("utf-8"))
        return body.get("content", [{"text": ""}])[0].get("text", "")
    
    def _parse_titan_response(self, response: Dict) -> str:
        """Parse response from Amazon Titan models"""
        body = json.loads(response.get("body").read().decode("utf-8"))
        return body.get("results", [{"outputText": ""}])[0].get("outputText", "")
    
    def _parse_meta_response(self, response: Dict) -> str:
        """Parse response from Meta models"""
        body = json.loads(response.get("body").read().decode("utf-8"))
        return body.get("generation", "")
    
    def _parse_ai21_response(self, response: Dict) -> str:
        """Parse response from AI21 models"""
        body = json.loads(response.get("body").read().decode("utf-8"))
        return body.get("completions", [{"data": {"text": ""}}])[0].get("data", {}).get("text", "")
        
    async def generate_text(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        reference_files: Optional[List[ReferenceFile]] = None,
        model_id: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text response from AWS Bedrock.
        
        Args:
            prompt: User prompt
            system_message: System instructions
            reference_files: List of reference files
            model_id: Model ID to use (default: anthropic.claude-3-opus-20240229)
            **kwargs: Additional parameters for API
            
        Returns:
            LLMResponse with generated text
        """
        model_id = model_id or self.default_model_id
        vendor = self._get_vendor(model_id)
        
        # Get request formatter for vendor
        formatter = self.model_formatters.get(vendor)
        if not formatter:
            raise ValueError(f"Unsupported model vendor: {vendor}")
        
        # Format request
        request_body = formatter(
            prompt=prompt,
            system_message=system_message,
            reference_files=reference_files,
            **kwargs
        )
        
        # Make API call
        response = self.client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
        
        # Parse response
        parser = self.response_parsers.get(vendor)
        if not parser:
            raise ValueError(f"Unsupported model vendor for response parsing: {vendor}")
            
        content = parser(response)
        
        return LLMResponse(
            content=content,
            model=model_id,
            raw_response=response
        )
        
    async def text_to_speech(
        self, 
        text: str, 
        voice_id: Optional[str] = "Matthew",
        engine: Optional[str] = "neural",
        **kwargs
    ) -> AudioResponse:
        """
        Convert text to speech using AWS Polly.
        
        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use
            engine: TTS engine to use (standard, neural)
            **kwargs: Additional parameters for API
            
        Returns:
            AudioResponse with audio data
        """
        # Create Polly client
        polly_client = boto3.client(
            'polly',
            region_name=kwargs.get('region_name', 'us-east-1'),
            aws_access_key_id=self.client._credentials.access_key,
            aws_secret_access_key=self.client._credentials.secret_key
        )
        
        # Request speech synthesis
        response = polly_client.synthesize_speech(
            Text=text,
            OutputFormat='mp3',
            VoiceId=voice_id,
            Engine=engine
        )
        
        # Get audio stream
        audio_data = response['AudioStream'].read()
        
        return AudioResponse(
            content=f"Audio generated using AWS Polly with voice {voice_id}",
            model=f"aws-polly-{engine}",
            audio_data=audio_data,
            raw_response=response
        )
    
    async def add_reference_file(
        self, 
        file_path: Union[str, Path],
        **kwargs
    ) -> ReferenceFile:
        """
        Add reference file.
        AWS Bedrock doesn't support direct file uploads in the same way,
        so we'll create a ReferenceFile object for local processing.
        
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