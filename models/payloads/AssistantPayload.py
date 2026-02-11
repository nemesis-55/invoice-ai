from pydantic import BaseModel
from typing import Optional, List

class LlmAttachment(BaseModel):
    file_name: str
    content_type: str
    data: str

class AssistantPayload(BaseModel):
    prompt: str
    attachments: Optional[List[LlmAttachment]] = None
    enable_thinking: bool = False  # Enable thinking mode for MiniCPM-V-4.5
