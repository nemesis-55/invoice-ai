import io
import os
import base64
import torch
import fitz  # PyMuPDF for handling PDFs
from PIL import Image
import json
from transformers import AutoTokenizer, AutoModel
import re
from peft import PeftModel
import pytesseract
import runpod

# Constants
DEFAULT_OCR_TARGET_SIZE = (1024, 1024)
DEFAULT_INPUT_TARGET_SIZE = (724, 724)
MODEL_DPI = 600
CACHE_DIR_MODEL = "./cache_dir/model"
CACHE_DIR_ADAPTOR = "./cache_dir/adaptor"

# Load model and tokenizer
model_type = "openbmb/MiniCPM-Llama3-V-2_5"
path_to_adapter = os.getenv("ADAPTER_DIR", "Zorro123444/xylem_invoice_extracter")

print("Loading model and tokenizer...")
model = AutoModel.from_pretrained(model_type, trust_remote_code=True, device_map="cuda", torch_dtype=torch.bfloat16, 
                                  cache_dir=CACHE_DIR_MODEL)
model = PeftModel.from_pretrained(model, path_to_adapter, device_map="cuda", trust_remote_code=True, torch_dtype=torch.bfloat16, 
                                  cache_dir=CACHE_DIR_ADAPTOR).eval()
tokenizer = AutoTokenizer.from_pretrained(path_to_adapter, trust_remote_code=True)
print("Model and tokenizer loaded successfully.")

def compact_ocr_data(ocr_data):
    """Compact the OCR extracted data by removing excessive spaces and line breaks."""
    print("Compacting OCR data...")
    cleaned_text = re.sub(r'\n+', '\n', ocr_data)
    cleaned_text = re.sub(r'[ ]{2,}', ' ', cleaned_text).strip()
    cleaned_text = re.sub(r'\n\s+', '\n', cleaned_text)
    cleaned_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', cleaned_text)
    print("OCR data compacted.")
    return cleaned_text

def pdf_page_to_image(pdf_page_bytes, dpi=MODEL_DPI, target_size=DEFAULT_INPUT_TARGET_SIZE):
    """Converts base64-encoded PDF bytes into an image and resizes it."""
    try:
        print("Converting PDF page to image...")
        pdf_document = fitz.open(stream=pdf_page_bytes, filetype="pdf")
        if len(pdf_document) < 1:
            raise ValueError("The PDF does not contain any pages.")
        page = pdf_document.load_page(0)
        pix = page.get_pixmap(dpi=dpi)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
        print(f"PDF converted to image with size {resized_image.size}.")
        return resized_image
    except Exception as e:
        print(f"Error rendering PDF page: {e}")
        return None

def load_image(image_data):
    """Loads a base64-encoded image, resizes it, and returns the image."""
    try:
        print("Loading image...")
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print(f"Image loaded with size: {image.size}")
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def replace_none_with_empty(value):
    """Replaces None or 'null' values with an empty string."""
    return value if value not in [None, "null"] else ""

def convert_to_order_structure(input_json):
    """Converts the input JSON to a specified order structure."""
    print("Converting JSON to order structure...")
    try:
        order_structure = {
            "OrderNumber": replace_none_with_empty(input_json.get("OrderNumber", "")),
            "InvoiceNumber": replace_none_with_empty(input_json.get("InvoiceNumber", "")),
            "BuyerName": replace_none_with_empty(input_json.get("BuyerName", "")),
            "BuyerAddress1": replace_none_with_empty(input_json.get("BuyerAddress1", "")),
            "BuyerZipCode": replace_none_with_empty(input_json.get("BuyerZipCode", "")),
            "BuyerCity": replace_none_with_empty(input_json.get("BuyerCity", "")),
            "BuyerCountry": replace_none_with_empty(input_json.get("BuyerCountry", "")),
            "ReceiverName": replace_none_with_empty(input_json.get("ReceiverName", "")),
            "ReceiverAddress1": replace_none_with_empty(input_json.get("ReceiverAddress1", "")),
            "ReceiverZipCode": replace_none_with_empty(input_json.get("ReceiverZipCode", "")),
            "ReceiverCity": replace_none_with_empty(input_json.get("ReceiverCity", "")),
            "ReceiverCountry": replace_none_with_empty(input_json.get("ReceiverCountry", "")),
            "SellerName": replace_none_with_empty(input_json.get("SellerName", "")),
            "NetAmount": replace_none_with_empty(input_json.get("NetAmount", "")),
            "OrderDate": replace_none_with_empty(input_json.get("OrderDate", "")),
            "Currency": replace_none_with_empty(input_json.get("Currency", "")),
            "TermsOfDelCode": replace_none_with_empty(input_json.get("TermsOfDelCode", "")),
            "OrderItems": [
                {
                    "ArticleNumber": replace_none_with_empty(item.get("ArticleNumber", "")),
                    "Description": replace_none_with_empty(item.get("Description", "")),
                    "HsCode": replace_none_with_empty(item.get("HsCode", "")),
                    "CountryOfOrigin": replace_none_with_empty(item.get("CountryOfOrigin", "")),
                    "Quantity": replace_none_with_empty(item.get("Quantity", "")),
                    "NetWeight": replace_none_with_empty(item.get("NetWeight", "")),
                    "NetAmount": replace_none_with_empty(item.get("NetAmount", "")),
                    "PricePerPiece": replace_none_with_empty(item.get("PricePerPiece", "")),
                    "EclEuNO": replace_none_with_empty(item.get("EclEuNO", "")),
                }
                for item in input_json.get("OrderItems", [])
            ],
            "NetWeight": replace_none_with_empty(input_json.get("NetWeight", "")),
            "NumberOfUnits": replace_none_with_empty(input_json.get("NumberOfUnits", ""))
        }
        print("JSON converted to order structure successfully.")
        return json.dumps(order_structure, indent=4)
    except Exception as e:
        print(f"Error converting JSON to order structure: {e}")
        return {}

def generate_detailed_prompt(ocr_data):
    # Create a compressed description of the task
    question = (
        "You are given an invoice image and extracted OCR text.\n\n"
        f"OCR Data:\n{ocr_data}\n\n"
        "Extract the following fields from the invoice and return them as JSON with the correct format:\n"
        
        "1. **OrderNumber**: Numeric Integer, 6-15 digits (e.g., '529448')\n"
        "2. **InvoiceNumber**: Numeric Integer, 6-15 digits (e.g., '602582')\n"
        "3. **BuyerName**: String, 5-100 chars (e.g., 'XYLEM WATER SOLUTIONS AS')\n"
        "4. **BuyerAddress1**: String, 5-150 chars (e.g., 'FETVEIEN 23')\n"
        "5. **BuyerZipCode**: String, pattern '[A-Z]{2}-\\d{4}' (e.g., 'NO-2007')\n"
        "6. **BuyerCity**: String, max 50 chars (e.g., 'KJELLER')\n"
        "7. **BuyerCountry**: String (e.g., 'NORWAY')\n"
        "8. **BuyerOrgNumber**: Numeric, 9 digits (e.g., '918088067')\n"
        "9. **ReceiverName**: String (e.g., 'XYLEM WATER SOLUTIONS NORGE AS')\n"
        "10. **ReceiverAddress1**: String (e.g., 'JANAFLATEN 37')\n"
        "11. **ReceiverZipCode**: String, pattern '[A-Z]{2}-\\d{4}' (e.g., 'NO-5179')\n"
        "12. **ReceiverCity**: String (e.g., 'GODVIK')\n"
        "13. **ReceiverCountry**: String (e.g., 'NORWAY')\n"
        "14. **SellerName**: String (e.g., 'xylem')\n"
        "15. **OrderDate**: Date, format 'YYYY-MM-DD' (e.g., '2023-09-18')\n"
        "16. **Currency**: String, 3 chars (e.g., 'NOK')\n"
        "17. **TermsOfDelCode**: String (e.g., 'DDP')\n"
        "18. **OrderItems** (list):\n"
        "    - **ArticleNumber**: Numeric Integer 5-15 digits (e.g., '841180')\n"
        "    - **Description**: String (e.g., 'KONDENSATOR 14 MFD 450V')\n"
        "    - **HsCode**: Numeric Integer, 8 digits (e.g., '85322900')\n"
        "    - **CountryOfOrigin**: String (e.g., 'BG')\n"
        "    - **Quantity**: Numeric Decimal (e.g., '1.0')\n"
        "    - **NetWeight**: Numeric Decimal (e.g., '0.070')\n"
        "    - **NetAmount**: Numeric Decimal (e.g., '160.28')\n"
        "    - **PricePerPiece**: Numeric Decimal (e.g., '160.28')\n"
        "19. **NetWeight**: Numeric Decimal(e.g., '55.000')\n"
        "20. **GrossWeight**: Numeric Decimal(if available)\n"
        "21. **NumberOfUnits**: Numeric Decimal (e.g., '1')\n"
        "22. **NumberOfPallets**: Numeric Decimal (if available)\n\n"
        "Match all values to the invoice exactly. Missing fields should be returned as null. Return the result as valid JSON."
    )

    print(f"Generated prompt length: {len(question)}")
    return question

def handle_inference(image, prompt):
    """Handles model inference with a given prompt and returns the result."""
    print("Performing inference...")
    msgs = [{"role": "user", "content": prompt}]

    try:
        with torch.no_grad():
            outputs = model.chat(image=image, msgs=msgs, tokenizer=tokenizer, max_new_tokens=8192)
        order_json = convert_to_order_structure(convert_string_to_json(outputs))
        print("Inference completed.")
        return order_json
    except Exception as e:
        print(f"Error during inference: {e}")
        return {}

def convert_string_to_json(data_str):
    """Convert the model output string to a JSON object."""
    print("Converting string to JSON...")
    json_str = data_str.replace("'", '"').replace("None", "null")
    try:
        json_data = json.loads(json_str)
        print("String successfully converted to JSON.")
        return json_data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return {}

def extract_text_from_image(image):
    """Extract text from an image using OCR."""
    print("Extracting text from image...")
    text = pytesseract.image_to_string(image)
    print(f"Extracted text length: {len(text)}...")  # Print first 100 characters of extracted text
    return text

def run(request):
    """Main run function for processing requests."""
    print("Starting request processing...")
    try:
        input_data = request["input"]
        pdf_data = input_data.get("pdf_data")
        ocr_data = input_data.get("ocr_data")

        if not pdf_data:
            return {"error": "Missing pdf data!"}

        pdf_bytes = base64.b64decode(pdf_data)

        if not ocr_data:
            print("No OCR data provided, extracting from image...")
            ocr_data = extract_text_from_image(pdf_page_to_image(pdf_bytes, MODEL_DPI, DEFAULT_OCR_TARGET_SIZE))
            ocr_data = compact_ocr_data(ocr_data)

        image = pdf_page_to_image(pdf_bytes, 300, DEFAULT_INPUT_TARGET_SIZE)
        prompt = generate_detailed_prompt(ocr_data)
        response = handle_inference(image, prompt)

        print("Request processed successfully.")
        return {"response": response}

    except Exception as e:
        print(f"Error during request processing: {e}")
        return {"error": f"Exception during processing: {str(e)}"}

# Start RunPod handler
print("Starting RunPod serverless handler...")
runpod.serverless.start({"handler": run})

