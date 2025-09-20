import base64
from models.payloads.PromptPayload import PromptPayload
from models.payloads.InvoiceExtractionPayload import InvoiceExtractionPayload
import torch
from PIL import Image
import fitz  # PyMuPDF for handling PDFs
from transformers import AutoTokenizer, AutoModel
import runpod
from huggingface_hub import login, scan_cache_dir
import base64
import fitz  # PyMuPDF
from peft import PeftModel
import os
from helper.order_csv_utils import expand_order_items_list_to_json
import time
import json

# Cache config: Ensure Hugging Face cache uses mounted volume (not /root)
CACHE_DIR = "/runpod-volume/cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

# Constants
MODEL_DPI = 300
ADAPTOR_TYPE = "GothiaDigitalSolutions/invoice-extractor-3.0"
cache = os.environ["HF_HOME"]

# Configurable model loading parameters
MODEL_PRECISION = os.getenv("MODEL_PRECISION", "16bit")  # 16bit, 8bit, or 4bit
GPU_DEVICE = os.getenv("GPU_DEVICE", "single")  # single, auto, or cuda:0, cuda:1, etc.
TORCH_DTYPE_MAP = {
    "16bit": torch.bfloat16,
    "8bit": torch.bfloat16,  # Still use bfloat16 for computation, 8bit for storage
    "4bit": torch.bfloat16   # Still use bfloat16 for computation, 4bit for storage
}

# One-time cache cleanup (remove old unreferenced revisions to free space)
try:
    cache_info = scan_cache_dir(cache_dir=cache)
    delete_hashes = []
    for repo in cache_info.repos:
        # Sort revisions newest first (keep newest always)
        revisions = sorted(
            list(repo.revisions),
            key=lambda r: getattr(r, "last_modified", 0),
            reverse=True
        )
        for rev in revisions[1:]:  # skip most recent
            # Some versions expose refs on revision, some on repo; be defensive
            rev_refs = getattr(rev, "refs", None)
            if rev_refs in (None, set(), frozenset()):
                delete_hashes.append(rev.commit_hash)
    if delete_hashes:
        strategy = cache_info.delete_revisions(*delete_hashes)
        print(f"Cache cleanup: will free {strategy.expected_freed_size_str} removing {len(delete_hashes)} old revisions")
        strategy.execute()
    else:
        print("Cache cleanup: nothing to remove")
except Exception as e:
    print(f"Cache cleanup skipped: {e}")

# Hugging Face login
login(os.getenv("HF_TOKEN"))

# Load Model and Tokenizer
def load_model_and_tokenizer():
    """Load the main model and tokenizer with configurable precision and GPU settings."""
    try:
        print(f"Loading tokenizer for model: {ADAPTOR_TYPE}")
        tokenizer = AutoTokenizer.from_pretrained(ADAPTOR_TYPE, trust_remote_code=True)
        
        # Configure device mapping based on GPU_DEVICE setting
        if GPU_DEVICE == "single":
            device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif GPU_DEVICE == "auto":
            device_map = "auto"
        else:
            device_map = GPU_DEVICE  # Allow custom device specification like "cuda:1"
        
        # Configure quantization based on MODEL_PRECISION
        load_in_8bit = MODEL_PRECISION == "8bit"
        load_in_4bit = MODEL_PRECISION == "4bit"
        torch_dtype = TORCH_DTYPE_MAP.get(MODEL_PRECISION, torch.bfloat16)
        
        print(f"Loading model with precision: {MODEL_PRECISION}, device_map: {device_map}")
        print(f"Model configuration: torch_dtype={torch_dtype}, load_in_8bit={load_in_8bit}, load_in_4bit={load_in_4bit}")
        
        # Build model loading kwargs
        model_kwargs = {
            "device_map": device_map,
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
            "cache_dir": cache
        }
        
        # Add quantization settings if needed
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        
        model = AutoModel.from_pretrained(ADAPTOR_TYPE, **model_kwargs)
        
        # Only call .cuda() and .eval() if not using quantization and device_map isn't handling it
        if not load_in_8bit and not load_in_4bit and device_map not in ["auto", "cuda"]:
            model = model.cuda().eval()
        else:
            model = model.eval()
        
        # Test model with a simple message
        messages = [
            {"role": "user", "content": "hey"}
        ]
        print(f"Test messages: {messages}")
        response = model.chat(image=None, msgs=messages, tokenizer=tokenizer, max_new_tokens=8192)
        print(f"Test response: {response}")

        print(f"Model loading complete with {MODEL_PRECISION} precision on {device_map}")
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load model and tokenizer: {e}")
        return None, None


# Convert PDF Page to Image
def pdf_to_image(pdf_bytes, dpi=MODEL_DPI):
    """Convert a single-page PDF to an image."""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = pdf_document.load_page(0)
        pix = page.get_pixmap(dpi = dpi)
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
            "Extract the following fields from the invoice image and return a JSON object:\n"
            "- OrderNumber\n"
            "- InvoiceNumber\n"
            "- BuyerName\n"
            "- BuyerAddress1\n"
            "- BuyerZipCode\n"
            "- BuyerCity\n"
            "- BuyerCountry\n"
            "- ReceiverName\n"
            "- ReceiverAddress1\n"
            "- ReceiverZipCode\n"
            "- ReceiverCity\n"
            "- ReceiverCountry\n"
            "- SellerName\n"
            "- NetAmount\n"
            "- OrderDate (YYYY-MM-DD)\n"
            "- Currency\n"
            "- TermsOfDelCode\n"
            "- ActualFreight\n"
            "- OrderItemsList: a list of lists. Each inner list represents one item and follows the column order:\n"
            "  ['Description', 'HsCode', 'HsCodeExport', 'Quantity', 'ArticleNumber', 'GrossWeight', "
            "'NetWeight', 'CountryOfOrigin', 'NumberOfUnits', 'TypeOfUnit', 'PricePerPiece', 'NetAmount']\n"
            "- NetWeight\n"
            "- OtherAmount\n"
            "- NumberOfUnits\n"
            "Use exact text from the image. If a value is missing, set it to an empty string \"\".\n"
            "Respond with only the JSON object."
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
            print(f"Inference messages: {messages}")
            response = model.chat(image=None, msgs=messages, tokenizer=tokenizer, max_new_tokens=8192)
            print(f"Inference response: {response}")
        return response
    except Exception as e:
        print(f"Inference failed: {e}")
        raise RuntimeError(f"Inference failed: {e}")

# Main Request Handler
def run(request):
    """Process incoming requests."""
    try:        
        payload = request.get("input", {})
        action = payload.get("action")
        data = payload.get("data", {})

        if action == "INVOICE_EXTRACTION":
            return handle_extract_invoice(data)
        elif action == "PROMPT":
            return handle_prompt(data)
        elif action == "ASSISTANT":
            return handle_assistant_request(data)

    except Exception as e:
        print(f"Exception during processing: {e}")
        return {"error": f"Exception during processing: {e}"}

def handle_extract_invoice(data):
    try:
        payload = InvoiceExtractionPayload(**data)
    except TypeError as e:
        print(f"Invalid prompt payload: {e}")
        return {"error": f"Invalid prompt payload: {e}"}

    pdf_data = payload.pdf_data
    page_number = payload.page_number or "0"

    if not pdf_data:
        print("Missing PDF data.")
        return {"error": "Missing PDF data."}

    pdf_bytes = base64.b64decode(pdf_data)
    prompt = generate_prompt(pdf_bytes)
    response = perform_inference(prompt, model, tokenizer)

    # this assumes the response is a JSON string, so in the prompt it should be mentioned to return a JSON string
    response = json.loads(response)
    
    # add key value pair for page number in response for all order items
    for item in response.get("OrderItemsList", []):
        item.append(int(page_number))

    json_response = expand_order_items_list_to_json(response)
    return {"response": json_response}

def handle_prompt(data):
    try:
        payload = PromptPayload(**data)
    except TypeError as e:
        print(f"Invalid prompt payload: {e}")
        return {"error": f"Invalid prompt payload: {e}"}
    
    messages = [{"role": "user", "content": payload.prompt}]
    response = perform_inference(messages, model, tokenizer)

    # this assumes the response is a JSON string, so in the prompt it should be mentioned to return a JSON string
    response = json.loads(response)
    return {"response": response}

# # Create a new handler function to handle assistant requests
def handle_assistant_request(data):
    try:
        payload = PromptPayload(**data)
    except TypeError as e:
        print(f"Invalid prompt payload: {e}")
        return{"error": f"Invalid prompt payload: {e}"}
    messages = [{"role":"user", "content": payload.prompt}]
    response = perform_inference(messages, model, tokenizer)
    return {"response": response}



start_time = time.time()
model, tokenizer = load_model_and_tokenizer()
print(f"Model loaded in {time.time() - start_time:.2f} seconds")


# Initialize and Start RunPod Handler
if __name__ == "__main__":
    print("Initializing RunPod serverless handler")
    runpod.serverless.start({"handler": run})
