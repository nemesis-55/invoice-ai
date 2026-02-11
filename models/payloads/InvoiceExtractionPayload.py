from pydantic import BaseModel

class InvoiceExtractionPayload(BaseModel):
    pdf_data: str
    page_number: str = "0"
    enable_thinking: bool = False  # Enable thinking mode for MiniCPM-V-4.5