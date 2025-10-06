from dataclasses import dataclass
from typing import Optional, List

@dataclass
class LlmAttachment:
    file_name: str
    content_type: str
    data: str

@dataclass
class AssistantPayload:
    prompt: str
    attachments: Optional[List[LlmAttachment]] = None
