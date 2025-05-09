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
from helper.order_csv_utils import expand_order_items_list_to_json
import time
import json

# Constants
MODEL_DPI = 300
ADAPTOR_TYPE = "GothiaDigitalSolutions/invoice-extractor-3.0"
cache = "/runpod-volume/cache"
login(os.getenv("HF_TOKEN"))

# Load Model and Tokenizer
def load_model_and_tokenizer():
    """Load the main model and tokenizer."""
    try:
        print("loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(ADAPTOR_TYPE, trust_remote_code=True)
        print("Loading model...")
        model = AutoModel.from_pretrained(
            ADAPTOR_TYPE,
            device_map="cuda",
            attn_implementation="sdpa",
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, 
            cache_dir=cache
        ).cuda().eval()
        messages = [
            {"role": "user", "content": "hey"}
        ]
        print("messages: ", messages)
        response = model.chat(image=None, msgs=messages, tokenizer=tokenizer, max_new_tokens=8192)
        print("response: ", response)

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
            print("messages: ", messages)
            response = model.chat(image=None, msgs=messages, tokenizer=tokenizer, max_new_tokens=8192)
            print("response: ", response)
        return response
    except Exception as e:
        print(f"Inference failed: {e}")
        raise RuntimeError(f"Inference failed: {e}")

# Main Request Handler
def run(request):
    """Process incoming requests."""
    try:
        # Parse request if it's a JSON string
        if isinstance(request, str):
            request = json.loads(request)

        print(f"DEBUG LOG:\nRequest: {request}")
        
        input_data = request.get("input", {})
        pdf_data = input_data.get("pdf_data")
        page_number = input_data.get("page_number", 0)

        if not pdf_data:
            return {"error": "Missing PDF data."}

        pdf_bytes = base64.b64decode(pdf_data)
        prompt = generate_prompt(pdf_bytes)
        response = perform_inference(prompt, model, tokenizer)

        # add key value pair for page number in response for all order items
        for item in response.get("OrderItemsList", []):
            item.append(int(page_number))

        json_response = expand_order_items_list_to_json(response)
        return {"response": json_response}
    except Exception as e:
        print(f"Exception during processing: {e}")
        return {"error": f"Exception during processing: {e}"}

start_time = time.time()
model, tokenizer = load_model_and_tokenizer()
print(f"Model loaded in {time.time() - start_time:.2f} seconds")


# Initialize and Start RunPod Handler
if __name__ == "__main__":
    print("Initializing RunPod serverless handler.")
    runpod.serverless.start({"handler": run})
