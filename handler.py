import sys
import base64
import torch
from PIL import Image
import fitz  # PyMuPDF for handling PDFs
from transformers import AutoTokenizer, AutoModel
import runpod
from huggingface_hub import login
import base64
import fitz  # PyMuPDF
from peft import PeftModel
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper.order_csv_utils import convert_csv_to_order_json_string

# Constants
MODEL_DPI = 300
MODEL_TYPE = "openbmb/MiniCPM-V-2_6"
ADAPTOR_TYPE = "GothiaDigitalSolutions/invoice-extractor"
cache = "/runpod-volume/cache"
login(os.getenv("HF_TOKEN"))

# Load Model and Tokenizer
def load_model_and_tokenizer():
    """Load the main model and tokenizer."""
    try:
        print("loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE, trust_remote_code=True)
        print("Loading base model...")
        base_model = AutoModel.from_pretrained(
            MODEL_TYPE,
            device_map="cuda",
            attn_implementation="sdpa",
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, 
            cache_dir=cache
        )

        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(
            base_model,
            ADAPTOR_TYPE,
            device_map="cuda",
            attn_implementation="sdpa",
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, 
            cache_dir=cache
        ).eval()

        print("Model Loading Complete")
        return model, tokenizer
    except Exception as e:
        print(f"exception: {e}")
        return None, None


# Convert PDF Page to Image
def pdf_to_image(pdf_bytes, dpi=MODEL_DPI):
    """Convert a single-page PDF to an image."""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = pdf_document.load_page(0)
        zoom = dpi / 72  # 72 is the default resolution
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        mode = "RGBA" if pix.alpha else "RGB"
        
        image =  Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        return image
    except Exception as e:
        print(f"Error converting PDF to image: {e}")
        raise ValueError(f"Error converting PDF to image: {e}")

# Generate Detailed Prompt
def generate_prompt(pdf_bytes):
    """Create the detailed prompt for the model."""
    try:
        image = pdf_to_image(pdf_bytes)
        question = (
            "You are given an image of an invoice.\n"
            "Extract and return the data as a CSV-formatted string where each row represents a single item from the invoice.\n\n"
            "Instructions:\n"
            "- Use commas as separators.\n"
            "- Include a header row with the field names listed below.\n"
            "- Repeat the order-level fields for each item row.\n"
            "- Use appropriate unicode values for special characters (e.g., Å, Ø, É) \n"
            "- If a value is missing or not visible, use an empty string \"\".\n"
            "- Do not add any commentary or formatting — return only the CSV content.\n\n"
            "CSV Columns:\n"
            "OrderNumber,InvoiceNumber,BuyerName,BuyerAddress1,BuyerZipCode,BuyerCity,BuyerCountry,"
            "ReceiverName,ReceiverAddress1,ReceiverZipCode,ReceiverCity,ReceiverCountry,"
            "SellerName,OrderDate,Currency,TermsOfDelCode,ActualFreight,NumberOfUnits,OtherAmount,"
            "Description,HsCode,HsCodeExport,Quantity,ArticleNumber,GrossWeight,NetWeight,"
            "CountryOfOrigin,TypeOfUnit,PricePerPiece,NetAmount,OrderLevelNetAmount,"
            "OrderLevelNetWeight,OrderLevelGrossWeight\n\n"
            "Return the CSV string only."
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

        if not pdf_data:
            return {"error": "Missing PDF data."}

        pdf_bytes = base64.b64decode(pdf_data)

        prompt = generate_prompt(pdf_bytes)
        response = perform_inference(prompt, model, tokenizer)
        json_response = convert_csv_to_order_json_string(response)
        return {"response": json_response}
    except Exception as e:
        print(f"Exception during processing: {e}")
        return {"error": f"Exception during processing: {e}"}

model, tokenizer = load_model_and_tokenizer()

# Initialize and Start RunPod Handler
if __name__ == "__main__":
    print("Initializing RunPod serverless handler.")
    runpod.serverless.start({"handler": run})
