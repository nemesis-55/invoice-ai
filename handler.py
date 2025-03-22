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
MODEL_DPI = 300
MODEL_TYPE = "openbmb/MiniCPM-V-2_6"
ADAPTOR_TYPE = "Zorro123444/invoice_extracter_5"
adaptor_dir = "/runpod-volume/cache/adaptor"
print("login to hugging face")
login("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")

# Load Model and Tokenizer
def load_model_and_tokenizer():
    """Load the main model and tokenizer."""
    try:
        print("loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE, trust_remote_code=True)
        print("loading model")
        model = AutoModel.from_pretrained(MODEL_TYPE, device_map="auto", attn_implementation='sdpa', trust_remote_code=True, cache_dir=adaptor_dir, torch_dtype=torch.bfloat16)
        print("loading complete")
        model = PeftModel.from_pretrained(model, ADAPTOR_TYPE, device_map="auto" , attn_implementation='sdpa', trust_remote_code=True, cache_dir=adaptor_dir, torch_dtype=torch.bfloat16).cuda().eval()
        return model, tokenizer
    except Exception as e:
        print(f"exception: {e}")
        return None, None


# Convert PDF Page to Image
def pdf_to_image(pdf_bytes, dpi=MODEL_DPI):
    """Convert a single-page PDF to an image."""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        zoom = dpi / 72  # 72 dpi is the default resolution
        matrix = fitz.Matrix(zoom, zoom)
        if len(pdf_document) < 1:
            raise ValueError("The PDF does not contain any pages.")
        page = pdf_document.load_page(0)
        pix = page.get_pixmap(matrix=matrix)
        mode = "RGBA" if pix.alpha else "RGB"
        return Image.frombytes(mode, [pix.width, pix.height], pix.samples)
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
            "Extract key details from the given image to return a valid JSON object.\n\n"
            "### Instructions:\n"
            "1. Extract the required fields as per the JSON structure.\n"
            "3. If a field is missing, set its value to \"\".\n"
            "### JSON Output:\n"
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
            "    \"ActualFreight\": \"<string>\",\n"
            "    \"OrderItems\": [\n"
            "        {\n"
            "            \"Description\": \"<string>\",\n"
            "            \"HsCode\": \"<string>\",\n"
            "            \"HsCodeExport\": \"<string>\",\n"
            "            \"Quantity\": \"<string>\",\n"
            "            \"ArticleNumber\": \"<string>\",\n"
            "            \"GrossWeight\": \"<string>\",\n"
            "            \"NetWeight\": \"<string>\",\n"
            "            \"CountryOfOrigin\": \"<string>\",\n"
            "            \"NumberOfUnits\": \"<string>\",\n"
            "            \"TypeOfUnit\": \"<string>\",\n"
            "            \"PricePerPiece\": \"<string>\",\n"
            "            \"NetAmount\": \"<string>\"\n"
            "        }\n"
            "    ],\n"
            "    \"NetWeight\": \"<string>\",\n"
            "    \"NumberOfUnits\": \"<string>\"\n"
            "}\n\n"
            "### Note: Output value must not contain any double quote (\")  \n"
            "Ensure the JSON structure is returned exactly as shown above, with appropriate values extracted using OCR data and image."
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
            # ocr_data = extract_text_from_image(pdf_bytes)

        prompt = generate_prompt(pdf_bytes, ocr_data)
        response = perform_inference(prompt, model, tokenizer)
        return {"response": response}
    except Exception as e:
        print(f"Exception during processing: {e}")
        return {"error": f"Exception during processing: {e}"}

model, tokenizer = load_model_and_tokenizer()

# Initialize and Start RunPod Handler
if __name__ == "__main__":
    print("Initializing RunPod serverless handler.")
    runpod.serverless.start({"handler": run})
