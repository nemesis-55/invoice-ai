import os
import base64
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import pytesseract
import runpod
from pdf2image import convert_from_bytes
from huggingface_hub import login

# Constants
CACHE_DIR_MODEL = os.getenv("ADAPTER_DIR", "./cache_dir/model")
CACHE_DIR_ADAPTOR = os.getenv("CACHE_DIR_ADAPTOR", "./cache_dir/adaptor")

# Load model and tokenizer
model_type = "openbmb/MiniCPM-V-2_6"
path_to_adapter = os.getenv("ADAPTER_DIR", "Zorro123444/invoice_extracter_2")

# Log in with your Hugging Face token
login("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")

print("Loading model and tokenizer...")
try:
    model = AutoModel.from_pretrained(
        model_type, trust_remote_code=True, device_map="cuda", cache_dir=CACHE_DIR_MODEL
    )
    model = PeftModel.from_pretrained(
        model, path_to_adapter, device_map="cuda", trust_remote_code=True, cache_dir=CACHE_DIR_ADAPTOR
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    raise

def pdf_bytes_to_images(pdf_bytes, dpi=300):
    try:
        print("Converting PDF bytes to images...")
        images = convert_from_bytes(pdf_bytes, dpi=dpi)
        print(f"Converted {len(images)} pages to images.")
        return images
    except Exception as e:
        print(f"Error converting PDF bytes to images: {e}")
        raise

def extract_text_from_pdf_bytes(pdf_bytes):
    try:
        print("Extracting text from PDF bytes...")
        images = pdf_bytes_to_images(pdf_bytes, dpi=600)
        text = ""
        for idx, image in enumerate(images):
            print(f"Extracting text from page {idx + 1}...")
            text += pytesseract.image_to_string(image)
        print("Text extraction completed.")
        return text
    except Exception as e:
        print(f"Error during text extraction: {e}")
        raise

def generate_detailed_prompt(pdf_bytes, ocr_data):
    try:
        print("Generating detailed prompt...")
        images = pdf_bytes_to_images(pdf_bytes)
        if not images:
            raise ValueError("No images generated from the PDF bytes.")
        
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
        prompt = [{'role': 'user', 'content': [images[0], question]}]
        print("Detailed prompt generated.")
        return prompt
    except Exception as e:
        print(f"Error generating detailed prompt: {e}")
        raise

def handle_inference(prompt):
    print("Performing inference...")
    try:
        with torch.no_grad():
            outputs = model.chat(image=None, msgs=prompt, tokenizer=tokenizer, max_new_tokens=8192)
        print("Inference completed.")
        return outputs
    except Exception as e:
        print(f"Error during inference: {e}")
        return {"error": f"Inference failed: {e}"}

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
            ocr_data = extract_text_from_pdf_bytes(pdf_bytes)

        prompt = generate_detailed_prompt(pdf_bytes, ocr_data)
        response = handle_inference(prompt)

        print("Request processed successfully.")
        return {"response": response}

    except Exception as e:
        print(f"Error during request processing: {e}")
        return {"error": f"Exception during processing: {str(e)}"}

# Start RunPod handler
if __name__ == "__main__":
    print("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": run})
