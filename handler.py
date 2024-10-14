import os
import io
import base64
import torch
import fitz  # PyMuPDF
import runpod
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

# Constants
DEFAULT_TARGET_SIZE = (1600, 1600)
MODEL_DPI = 300

# Load model path from environment variables or default
MODEL_DIR = os.getenv("MODEL_DIR", "./model")

# Load the tokenizer and model with GPU support and optimized for BF16 precision
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cuda",
    cache_dir="./cache_dir"
)

print("Model loaded with BF16 precision on GPU.")

def pdf_page_to_image(pdf_bytes, dpi=MODEL_DPI, target_size=DEFAULT_TARGET_SIZE):
    """
    Converts a single-page PDF (from bytes) into a low-resolution image,
    resized to a target size.
    
    Args:
        pdf_bytes (bytes): The input PDF file in byte format (single-page PDF).
        dpi (int): DPI for rendering the PDF. Default is 300.
        target_size (tuple): Target size for the image. Default is (1600, 1600).
    
    Returns:
        PIL.Image.Image: The low-res image resized to the target size.
    """
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    if len(pdf_document) < 1:
        raise ValueError("The input PDF does not contain any pages.")

    # Render the first page to an image
    page = pdf_document.load_page(0)
    pix = page.get_pixmap(dpi=dpi)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Resize image to target size
    return resize_image(image, target_size)

def resize_image(image, target_size):
    """
    Resize an image to the given target size while maintaining aspect ratio.
    
    Args:
        image (PIL.Image.Image): Input image.
        target_size (tuple): Target size (width, height).
    
    Returns:
        PIL.Image.Image: Resized image.
    """
    return image.resize(target_size, Image.Resampling.LANCZOS)

def generate_prompt(image, ocr_data):
    """
    Generates a detailed prompt to extract structured data from the image and OCR text.
    
    Args:
        image (PIL.Image.Image): Image of the invoice or PDF.
        ocr_data (str): OCR extracted text from the image.
    
    Returns:
        list: A formatted prompt for the language model.
    """
    question = (
        "You are provided with an image and OCR extracted text of an invoice PDF page.\n\n"
        "OCR data:\n"
        f"{ocr_data}\n\n"
        "Task: Use the image and OCR text to extract specific information and output as a JSON object with these fields:\n"
        "1. **OrderNumber**, 2. **InvoiceNumber**, 3. **BuyerName**, 4. **BuyerAddress1**, 5. **BuyerZipCode**, etc.\n"
        "Ensure accuracy, follow the expected format, and return fields with empty strings or null if not available."
        "Ensure output is always pure JSON."
    )
    
    return [{"role": "user", "content": [image, question]}]

def handler(event):
    """
    Main handler for RunPod serverless function. Processes the input event to generate 
    a detailed prompt and model response.
    
    Args:
        event (dict): Input event data.
    
    Returns:
        dict: The response containing the model's output or an error message.
    """
    try:
        # Extract data from the event
        event_data = event.get("input", {})
        image_data = event_data.get("image")
        pdf_bytes = event_data.get("pdf_bytes")
        ocr_data = event_data.get("ocr_data", "")

        # Load the image or PDF and convert it to an image
        if image_data:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif pdf_bytes:
            image = pdf_page_to_image(pdf_bytes)
        else:
            raise ValueError("No image or PDF bytes provided in the input.")

        # Generate the prompt using the image and OCR data
        prompt = generate_prompt(image, ocr_data)

        # Generate a response from the model
        print("Generating response...")
        with torch.no_grad():
            output = model.chat(
                image=None,
                msgs=prompt,
                tokenizer=tokenizer,
                max_new_tokens=8192
            )

        return {"response": output}

    except Exception as e:
        print(f"Error in handler: {e}")
        return {"error": str(e)}

def health_check(event):
    """
    Health check endpoint for the serverless function.
    
    Args:
        event (dict): Event data.
    
    Returns:
        dict: Health check status.
    """
    print("Health check hit.")
    return {"status": "ok"}

# Start the RunPod serverless function
runpod.serverless.start({"handler": handler})
