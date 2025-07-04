from dataclasses import dataclass

@dataclass
class InvoiceExtractionPayload:
    pdf_data: str
    page_number: str = "0"