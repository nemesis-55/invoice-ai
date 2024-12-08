import base64
import torch
from PIL import Image
import fitz  # PyMuPDF for handling PDFs
import pytesseract
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import runpod
from huggingface_hub import login
import os
import subprocess
import logging
import traceback


# Custom logging handler to print and log to a file
class PrintAndLogHandler(logging.Handler):
    def emit(self, record):
        log_message = self.format(record)
        print(log_message)  # Print to console
        with open('handler.log', 'a') as log_file:  # Log to a file
            log_file.write(log_message + '\n')

# Set up logging to both console and file
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
handler = PrintAndLogHandler()
logger.addHandler(handler)

# Install flash-attn
logger.info("Installing flash-attn...")
subprocess.check_call([os.sys.executable, "-m", "pip", "install", "flash-attn"])
logger.info("flash-attn installed successfully.")


# Constants
MODEL_DPI = 600
MODEL_TYPE = "openbmb/MiniCPM-V-2_6"
ADAPTOR_TYPE = "Zorro123444/invoice_extracter_2"
CACHE_DIR_MODEL = "./cache_dir/model"
CACHE_DIR_ADAPTOR = "./cache_dir/adaptor"

# Load Model and Tokenizer
def load_model_and_tokenizer():
    """Load the main model and tokenizer."""
    logging.info("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE, trust_remote_code=True)
        logging.info("Tokenizer loaded.")
        
        # Log the loading process of the base model
        logging.info(f"Loading model from {MODEL_TYPE}...")
        model = AutoModel.from_pretrained(
            ADAPTOR_TYPE,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR_MODEL
        ).eval().cuda()
        logging.info("Model and adapter loaded successfully.")
        
        return model, tokenizer
    except Exception as e:
        logging.error(f"Model or adapter loading failed with error: {str(e)}")
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())

# Convert PDF Page to Image
def pdf_to_image(pdf_bytes, dpi=MODEL_DPI):
    """Convert a single-page PDF to an image."""
    try:
        logging.info("Converting PDF to image...")
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(pdf_document) < 1:
            raise ValueError("The PDF does not contain any pages.")
        page = pdf_document.load_page(0)
        pix = page.get_pixmap(dpi=dpi)
        logging.info("PDF converted to image.")
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except Exception as e:
        logging.error(f"Error converting PDF to image: {e}")
        raise ValueError(f"Error converting PDF to image: {e}")

# Extract Text using OCR
def extract_text_from_image(pdf_bytes, dpi=MODEL_DPI):
    """Extract text from an image derived from the PDF."""
    logging.info("Extracting text from image...")
    try:
        image = pdf_to_image(pdf_bytes, dpi)
        text = pytesseract.image_to_string(image)
        logging.info(f"Extracted text length: {len(text)} characters.")
        return text
    except Exception as e:
        logging.error(f"Error during text extraction: {e}")
        raise RuntimeError(f"Error during text extraction: {e}")

# Generate Detailed Prompt
def generate_prompt(pdf_bytes, ocr_data):
    """Create the detailed prompt for the model."""
    logging.info("Generating prompt...")
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
        logging.error(f"Error generating prompt: {e}")
        raise RuntimeError(f"Error generating prompt: {e}")

# Handle Inference
def perform_inference(messages, model, tokenizer):
    """Perform model inference."""
    logging.info("Performing inference...")
    try:
        with torch.no_grad():
            response = model.chat(image=None, msgs=messages, tokenizer=tokenizer, max_new_tokens=8192)
        logging.info("Inference completed successfully.")
        return response
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        raise RuntimeError(f"Inference failed: {e}")

# Main Request Handler
def run(request):
    """Process incoming requests."""
    logging.info("Processing request...")
    try:
        input_data = request.get("input", {})
        pdf_data = input_data.get("pdf_data")
        ocr_data = input_data.get("ocr_data")

        if not pdf_data:
            logging.error("Missing PDF data.")
            return {"error": "Missing PDF data."}

        pdf_bytes = base64.b64decode(pdf_data)

        if not ocr_data:
            logging.info("No OCR data provided. Extracting...")
            ocr_data = extract_text_from_image(pdf_bytes)

        prompt = generate_prompt(pdf_bytes, ocr_data)
        response = perform_inference(prompt, model, tokenizer)
        return {"response": response}
    except Exception as e:
        logging.error(f"Exception during processing: {e}")
        return {"error": f"Exception during processing: {e}"}
    
# Log in with your Hugging Face token
login("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")    
model, tokenizer = load_model_and_tokenizer()

# Initialize and Start RunPod Handler
if __name__ == "__main__":
    logging.info("Initializing RunPod serverless handler.")
    runpod.serverless.start({"handler": run})
