import base64
import torch
from PIL import Image
import fitz  # PyMuPDF for handling PDFs
import pytesseract
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login

# Constants
MODEL_DPI = 600
MODEL_TYPE = "./model/invoice-ai-2_6"

# Load Model and Tokenizer
def load_model_and_tokenizer():
    """Load the main model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE, trust_remote_code=True)
        
        # Log the loading process of the base model
        model =  AutoModel.from_pretrained(
                MODEL_TYPE,
                trust_remote_code=True, torch_dtype=torch.bfloat16
                ).eval().cuda()
        
        return model, tokenizer
    except Exception as e:
        print(f"exception: {e}")


# Convert PDF Page to Image
def pdf_to_image(pdf_bytes, dpi=MODEL_DPI):
    """Convert a single-page PDF to an image."""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(pdf_document) < 1:
            raise ValueError("The PDF does not contain any pages.")
        page = pdf_document.load_page(0)
        pix = page.get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except Exception as e:
        raise ValueError(f"Error converting PDF to image: {e}")

# Extract Text using OCR
def extract_text_from_image(pdf_bytes, dpi=MODEL_DPI):
    """Extract text from an image derived from the PDF."""
    print("Extracting text from PDF...")
    try:
        image = pdf_to_image(pdf_bytes, dpi)
        text = pytesseract.image_to_string(image)
        print(f"Extracted text length: {len(text)} characters.")
        return text
    except Exception as e:
        raise RuntimeError(f"Error during text extraction: {e}")

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

# Handle Inference
def perform_inference(messages, model, tokenizer):
    """Perform model inference."""
    print("Performing inference...")
    try:
        with torch.no_grad():
            response = model.chat(image=None, msgs=messages, tokenizer=tokenizer, max_new_tokens=8192)
        return response
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

# Main Request Handler
def run(request):
    """Process incoming requests."""
    print("Processing request...")
    try:
        input_data = request.get("input", {})
        pdf_data = input_data.get("pdf_data")
        ocr_data = input_data.get("ocr_data")

        if not pdf_data:
            return {"error": "Missing PDF data."}

        pdf_bytes = base64.b64decode(pdf_data)

        if not ocr_data:
            print("No OCR data provided. Extracting...")
            ocr_data = extract_text_from_image(pdf_bytes)

        prompt = generate_prompt(pdf_bytes, ocr_data)
        response = perform_inference(prompt, model, tokenizer)
        return {"response": response}
    except Exception as e:
        return {"error": f"Exception during processing: {e}"}
    
# Log in with your Hugging Face token
login("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")    
model, tokenizer = load_model_and_tokenizer()

import base64
import fitz  # PyMuPDF
import os


def page_to_pdf_bytes(page):
    """
    Convert a single PDF page to PDF bytes.

    Args:
        page (fitz.Page): The page object from a PDF.

    Returns:
        bytes: PDF bytes for the single page.
    """
    single_page_doc = fitz.open()  # Create an empty PDF
    single_page_doc.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
    pdf_bytes = single_page_doc.tobytes()
    return pdf_bytes

def process_pdf(file_path):
    """
    Loads a PDF file, converts each page into PDF bytes, and sends it page by page to the API.

    Args:
        file_path (str): Path to the PDF file to be processed.
    """
    # Open the PDF document
    pdf_document = fitz.open(file_path)


    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)  # Load each page
        pdf_bytes = page_to_pdf_bytes(page)  # Convert the page to PDF bytes
        post_data = {
            "input": {
                "pdf_data": base64.b64encode(pdf_bytes).decode('utf-8')  # Convert PDF bytes to base64 string
            }
        }

        response = run(post_data)
        print(response)

# Path to your PDF file
pdf_file_path = "/Users/saurav.kumar3/Downloads/XYLEM_1_30_2.pdf"  # Replace with the actual file path

# Process the PDF and get the results
process_pdf(pdf_file_path)

print(f"Processing completed for {os.path.basename(pdf_file_path)}")

