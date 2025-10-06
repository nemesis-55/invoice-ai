from pydantic import BaseModel
from typing import Optional

class PromptPayload(BaseModel):
    prompt: str
    image: Optional[str] = None # Image is being used for classification