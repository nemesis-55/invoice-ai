from dataclasses import dataclass
from typing import Optional

@dataclass
class PromptPayload:
    prompt: str
    image: Optional[str] = None # Image is being used for classification