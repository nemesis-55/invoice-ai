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


def generate_detailed_prompt(image, ocr_data):

    # Create a detailed description of the task
    question = (
        "You are provided with an invoice in PDF format as an image, along with the extracted OCR text.\n\n"
        "OCR Data:\n"
        f"{ocr_data}\n\n"
        "Your task is to extract specific fields from the invoice and return them in a valid JSON format. "
        "Make sure that the extracted values match the original values from the invoice exactly, with no formatting changes. "
        "If a field is missing, return an empty string.\n\n"
        
        "### Field Descriptions:\n"
        "1. **OrderNumber**: The order number as it appears on the invoice.\n"
        "2. **InvoiceNumber**: The invoice number from the document.\n"
        "3. **BuyerName**: The full name of the buyer as written.\n"
        "4. **BuyerAddress1**: The first line of the buyer’s address, unchanged.\n"
        "5. **BuyerZipCode**: The postal code of the buyer.\n"
        "6. **BuyerCity**: The city where the buyer is located.\n"
        "7. **BuyerCountry**: The country of the buyer.\n"
        "8. **BuyerOrgNumber**: Organization number of the buyer, if available.\n"
        "9. **ReceiverName**: The full name of the receiver exactly as it appears.\n"
        "10. **ReceiverAddress1**: The first line of the receiver’s address without changes.\n"
        "11. **ReceiverZipCode**: The postal code of the receiver.\n"
        "12. **ReceiverCity**: The city where the receiver is located.\n"
        "13. **ReceiverCountry**: The country where the receiver resides.\n"
        "14. **SellerName**: The name of the seller or company issuing the invoice.\n"
        "15. **OrderDate**: The exact date when the order was placed.\n"
        "16. **Currency**: The currency code or symbol used for the transaction (e.g., $, EUR).\n"
        "17. **TermsOfDelCode**: Code representing the terms of delivery (e.g., DDP).\n"
        
        "18. **OrderItems** (list): For each item in the order, extract the following fields:\n"
        "    - **ArticleNumber**: Extract the article number of the item.\n"
        "    - **Description**: Extract the description of the item.\n"
        "    - **HsCode**: Extract the H.S.CODE as a string for the item. (eg. H.S.CODE:84109910.)\n"
        "    - **CountryOfOrigin**: The country where the item was manufactured.\n"
        "    - **Quantity**: The number of units ordered.\n"
        "    - **NetWeight**: The net weight of the item.\n"
        "    - **NetAmount**: The total net amount for the item.\n"
        "    - **PricePerPiece**: Extract the price per piece of the item.\n"
        "    - **GrossWeight**: Extract the gross weight of the item, if applicable.\n"
        
        "19. **NetWeight**: Total net weight of the order.\n"
        "20. **GrossWeight**: Total gross weight of the order.\n"
        "21. **NumberOfUnits**: Total number of units in the order.\n"
        "22. **NumberOfPallets**: Total number of pallets in the order.\n"
        
        "### Important Notes:\n"
        "Ensure all extracted values match the exact values in the original PDF without any changes. "
        "Return the result as a valid JSON object. If any field is missing set it with an empty string. "
        "Avoid adding any comments or additional information beyond the JSON structure."
    )

    # Create the prompt in the desired format
    prompt = [{"role": "user", "content": [image, question]}]

    return prompt


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
        prompt = generate_detailed_prompt(image, ocr_data)

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
