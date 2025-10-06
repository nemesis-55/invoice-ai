from pydantic import BaseModel

class InvoiceExtractionPayload(BaseModel):
    pdf_data: str
    page_number: str = "0"