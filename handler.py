import base64
import torch
from PIL import Image
import fitz  # PyMuPDF for handling PDFs
import pytesseract
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import login
import io

# Constants
MODEL_DPI = 600
MODEL_TYPE = "openbmb/MiniCPM-V-2_6"
ADAPTOR_TYPE = "Zorro123444/invoice_extracter_2"
CACHE_DIR_MODEL = "./cache_dir/model"
CACHE_DIR_ADAPTOR = "./cache_dir/adaptor"

# Initialize FastAPI app
app = FastAPI()

# Define request and response schemas
class RequestData(BaseModel):
    pdf_data: str
    ocr_data: str = None

class ResponseData(BaseModel):
    response: dict

# Load Model and Tokenizer
def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer."""
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(
        MODEL_TYPE,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="sdpa",
        cache_dir=CACHE_DIR_MODEL,
    )
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTOR_TYPE,
        trust_remote_code=True,
        cache_dir=CACHE_DIR_ADAPTOR,
    ).eval()
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

# Convert PDF Page to Image
def pdf_to_image(pdf_bytes, dpi=MODEL_DPI):
    """Convert a PDF page to an image."""
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    if len(pdf_document) < 1:
        raise ValueError("The PDF does not contain any pages.")
    page = pdf_document.load_page(0)
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

# Extract Text using OCR
def extract_text_from_image(pdf_bytes, dpi=MODEL_DPI):
    """Extract text from an image derived from a PDF."""
    image = pdf_to_image(pdf_bytes, dpi)
    return pytesseract.image_to_string(image)

# Generate Detailed Prompt
def generate_prompt(pdf_bytes, ocr_data):
    """Create the detailed prompt for the model."""
    try:
        print("Generating prompt...")
        image = pdf_to_image(pdf_bytes)
        question = (
            "You are an AI model specialized in data extraction from invoices. "
            "Below, you are provided with OCR-extracted text from an invoice. "
            "Your task is to analyze the OCR data and extract key details to structure them as a JSON object.\n\n"
            f"### OCR Data:\n{ocr_data}\n\n"
            "### Instructions:\n"
            "1. Extract all the required fields as specified in the JSON structure below.\n"
            "2. Ensure the output is a syntactically valid JSON string.\n"
            "3. If a field is missing or unavailable in the OCR text, set its value to an empty string \"\".\n"
            "4. Maintain the exact formatting of numeric values and dates as found in the input.\n"
            "5. Do not include additional explanations or comments in your output.\n\n"
            "### JSON Structure:\n"
            "The JSON structure must match the following format exactly:\n"
            "{\n"
            "    \"OrderNumber\": \"<string>\",\n"
            "    \"InvoiceNumber\": \"<string>\",\n"
            "    \"BuyerName\": \"<string>\",\n"
            "    \"BuyerAddress1\": \"<string>\",\n"
            "    \"BuyerZipCode\": \"<string>\",\n"
            "    \"BuyerCity\": \"<string>\",\n"
            "    \"BuyerCountry\": \"<string>\",\n"
            "    \"ReceiverName\": \"<string>\",\n"
            "    \"ReceiverAddress1\": \"<string>\",\n"
            "    \"ReceiverZipCode\": \"<string>\",\n"
            "    \"ReceiverCity\": \"<string>\",\n"
            "    \"ReceiverCountry\": \"<string>\",\n"
            "    \"SellerName\": \"<string>\",\n"
            "    \"NetAmount\": \"<string>\",\n"
            "    \"OrderDate\": \"<YYYY-MM-DD>\",\n"
            "    \"Currency\": \"<string>\",\n"
            "    \"TermsOfDelCode\": \"<string>\",\n"
            "    \"OrderItems\": [\n"
            "        {\n"
            "            \"ArticleNumber\": \"<string>\",\n"
            "            \"Description\": \"<string>\",\n"
            "            \"HsCode\": \"<string>\",\n"
            "            \"CountryOfOrigin\": \"<string>\",\n"
            "            \"Quantity\": \"<string>\",\n"
            "            \"NetWeight\": \"<string>\",\n"
            "            \"NetAmount\": \"<string>\",\n"
            "            \"PricePerPiece\": \"<string>\",\n"
            "            \"EclEuNO\": \"<string>\"\n"
            "        }\n"
            "    ],\n"
            "    \"NetWeight\": \"<string>\",\n"
            "    \"NumberOfUnits\": \"<string>\"\n"
            "}\n\n"
            "### Note:\n"
            "Ensure the JSON structure is returned exactly as shown above, with appropriate values extracted from the OCR data."
        )
        
        return [{"role": "user", "content": [image, question]}]
    except Exception as e:
        raise RuntimeError(f"Error generating prompt: {e}")


# Perform Inference
def perform_inference(messages, model, tokenizer):
    """Run inference using the model."""
    print("Performing inference...")
    with torch.no_grad():
        response = model.chat(image=None, msgs=messages, tokenizer=tokenizer, max_new_tokens=8192)
    return response

# FastAPI POST Endpoint to handle the PDF processing
@app.post("/generate", response_model=ResponseData)
async def generate_invoice_data(request: RequestData):
    """Process incoming requests and generate structured invoice data."""
    try:
        pdf_data = request.pdf_data
        ocr_data = request.ocr_data

        if not pdf_data:
            raise HTTPException(status_code=400, detail="Missing PDF data.")

        pdf_bytes = base64.b64decode(pdf_data)

        # Extract OCR data if not provided
        if not ocr_data:
            print("No OCR data provided. Extracting from PDF...")
            ocr_data = extract_text_from_image(pdf_bytes)

        # Generate prompt and perform inference
        prompt = generate_prompt(pdf_bytes, ocr_data)
        response = perform_inference(prompt, model, tokenizer)

        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during processing: {str(e)}")

# Model Initialization (on container startup)
def initialize_model():
    """Load the model and tokenizer when the container starts."""
    login("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")  # Replace with your Hugging Face token
    global model, tokenizer
    model, tokenizer = load_model_and_tokenizer()

# Initialize model on container start
initialize_model()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
