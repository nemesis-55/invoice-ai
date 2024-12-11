import base64
import torch
from PIL import Image
import fitz  # PyMuPDF for handling PDFs
import pytesseract
from transformers import AutoTokenizer, AutoModel
import runpod
from huggingface_hub import login
import base64
import fitz  # PyMuPDF
from peft import PeftModel
import numpy

# Constants
MODEL_DPI = 600
MODEL_TYPE = "openbmb/MiniCPM-V-2_6"
ADAPTOR_TYPE = "Zorro123444/invoice_extracter_2"
adaptor_dir = "./cache/adaptor"
print("login to hugging face")
login("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")

# Load Model and Tokenizer
def load_model_and_tokenizer():
    """Load the main model and tokenizer."""
    try:
        print("loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE, trust_remote_code=True)
        print("loading model")
        model = AutoModel.from_pretrained(MODEL_TYPE, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation='sdpa', trust_remote_code=True, cache_dir=adaptor_dir)
        print("loading complete")
        model = PeftModel.from_pretrained(model, ADAPTOR_TYPE,torch_dtype=torch.bfloat16, device_map="cuda" , attn_implementation='sdpa', trust_remote_code=True, cache_dir=adaptor_dir).cuda().eval()
        return model, tokenizer
    except Exception as e:
        print(f"exception: {e}")
        return None, None


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
        print(f"Error converting PDF to image: {e}")
        raise ValueError(f"Error converting PDF to image: {e}")

# Extract Text using OCR
def extract_text_from_image(pdf_bytes, dpi=MODEL_DPI):
    """Extract text from an image derived from the PDF."""
    try:
        image = pdf_to_image(pdf_bytes, dpi)
        text = pytesseract.image_to_string(image)
        print(f"Extracted text length: {len(text)} characters.")
        return text
    except Exception as e:
        print(f"Error during text extraction: {e}")
        raise RuntimeError(f"Error during text extraction: {e}")

# Generate Detailed Prompt
def generate_prompt(pdf_bytes, ocr_data):
    """Create the detailed prompt for the model."""
    try:
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
        print(f"Error generating prompt: {e}")
        raise RuntimeError(f"Error generating prompt: {e}")

# Handle Inference
def perform_inference(messages, model, tokenizer):
    """Perform model inference."""
    try:
        with torch.no_grad():
            response = model.chat(image=None, msgs=messages, tokenizer=tokenizer, max_new_tokens=4096)
        return response
    except Exception as e:
        print(f"Inference failed: {e}")
        raise RuntimeError(f"Inference failed: {e}")

# Main Request Handler
def run(request):
    """Process incoming requests."""
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
        return response
    except Exception as e:
        print(f"Exception during processing: {e}")
        return {"error": f"Exception during processing: {e}"}

model, tokenizer = load_model_and_tokenizer()

# Initialize and Start RunPod Handler
if __name__ == "__main__":
    print("Initializing RunPod serverless handler.")
    runpod.serverless.start({"handler": run})
