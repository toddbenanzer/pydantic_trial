"""
Base module defining interfaces for LLM providers.
Contains common models and abstract base classes.
"""
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
import base64

from pydantic import BaseModel, Field, ConfigDict

class ContentType(str, Enum):
    """Enum for content types"""
    TEXT = "text"
    IMAGE = "image"
    CODE = "code"
    AUDIO = "audio"

class Message(BaseModel):
    """Base message model"""
    model_config = ConfigDict(extra="allow")
    
    role: str
    content: str
    
class SystemMessage(Message):
    """System message"""
    role: str = "system"
    
class UserMessage(Message):
    """User message"""
    role: str = "user"
    
class AssistantMessage(Message):
    """Assistant message"""
    role: str = "assistant"

class ReferenceFile(BaseModel):
    """Reference file model"""
    path: Path
    file_id: Optional[str] = None
    content_type: Optional[str] = None
    
    def read_binary(self) -> bytes:
        """Read file as binary"""
        with open(self.path, 'rb') as f:
            return f.read()
            
    def read_text(self) -> str:
        """Read file as text"""
        with open(self.path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def base64_encode(self) -> str:
        """Return base64 encoded content"""
        return base64.b64encode(self.read_binary()).decode('utf-8')

class LLMResponse(BaseModel):
    """Response from LLM"""
    model_config = ConfigDict(extra="allow")
    
    content: str
    model: str
    raw_response: Optional[Any] = None
    
class AudioResponse(LLMResponse):
    """Audio response from TTS"""
    content: str  # Description of audio
    audio_data: bytes  # Raw audio data
    
class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate_text(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        reference_files: Optional[List[ReferenceFile]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text response"""
        pass
    
    @abstractmethod
    async def generate_code(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        reference_files: Optional[List[ReferenceFile]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate code response"""
        pass
    
    @abstractmethod
    async def text_to_speech(
        self, 
        text: str, 
        **kwargs
    ) -> AudioResponse:
        """Convert text to speech"""
        pass
        
    @abstractmethod
    async def add_reference_file(
        self, 
        file_path: Union[str, Path],
        **kwargs
    ) -> ReferenceFile:
        """Add reference file to context"""
        pass
        
    def not_implemented(self, feature: str) -> None:
        """Raise NotImplementedError for unsupported features"""
        raise NotImplementedError(f"{feature} is not supported by this provider")