from pydantic import BaseModel
from typing import Optional

class InvoiceExtractionPayload(BaseModel):
    pdf_data: str
    page_number: str = "0"
    deep_thinking: Optional[bool] = None