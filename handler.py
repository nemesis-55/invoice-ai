import os
import io
import base64
import torch
import fitz  # PyMuPDF for handling PDFs
import runpod
from PIL import Image
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# Constants
DEFAULT_TARGET_SIZE = (1600, 1600)
MODEL_DPI = 100

# Load model path from environment variables or default
MODEL_DIR = os.getenv("MODEL_DIR", "./model")

# Load the tokenizer and model with GPU support and FP16 precision
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # Use FP16 precision
    device_map="cuda",  # Use GPU
    cache_dir="./cache_dir"
)

print("Model loaded with FP16 precision on GPU.")

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
        "Your task is to extract specific fields from the invoice and return them in a valid mentioned JSON format. "
        "Make sure that the extracted values is not modified, use ocr_data to pick correct data. Default value of the below field is empty string "" \n\n"
        
        "### Field Descriptions:\n"
        "1. **OrderNumber**: The order number as it appears on the invoice (numeric, 6+ digits, e.g., 531947).\n"
        "2. **InvoiceNumber**: The invoice number from the document (numeric, 6+ digits, e.g., 623946).\n"
        "3. **BuyerName**: The full name of the buyer as written (e.g., XYLEM WATER SOLUTIONS AS).\n"
        "4. **BuyerAddress1**: The first line of the buyer’s address, unchanged (e.g., FETVEIEN 23).\n"
        "5. **BuyerZipCode**: The postal code of the buyer (e.g., NO-2007).\n"
        "6. **BuyerCity**: The city where the buyer is located (e.g., KJELLER).\n"
        "7. **BuyerCountry**: The country of the buyer (e.g., NORWAY).\n"
        "8. **BuyerOrgNumber**: Organization number of the buyer, if available (numeric, 9 digits).\n"
        "9. **ReceiverName**: The full name of the receiver exactly as it appears (e.g., Rørleggermester Kjell Horn AS).\n"
        "10. **ReceiverAddress1**: The first line of the receiver’s address without changes (e.g., Elvegårdsveien 5).\n"
        "11. **ReceiverZipCode**: The postal code of the receiver (e.g., 8370).\n"
        "12. **ReceiverCity**: The city where the receiver is located (e.g., Leknes).\n"
        "13. **ReceiverCountry**: The country where the receiver resides (e.g., NORWAY).\n"
        "14. **SellerName**: The name of the seller or company issuing the invoice (e.g., xylem).\n"
        "15. **OrderDate**: The exact date when the order was placed (e.g., 2023-12-05).\n"
        "16. **Currency**: The currency code or symbol used for the transaction (e.g., NOK).\n"
        "17. **TermsOfDelCode**: Code representing the terms of delivery (e.g., DDP).\n"
        
        "18. **OrderItems** (list): For each item in the order, extract the following fields:\n"
        "    - **ArticleNumber**: Extract the article number of the item (e.g., 6697700).\n"
        "    - **Description**: Extract the description of the item (e.g., GEJDFÄSTE ENHET).\n"
        "    - **HsCode**: Extract the HS code for the item (8-digit tariff number, e.g., 73269098).\n"
        "    - **CountryOfOrigin**: The country where the item was manufactured (e.g., SE).\n"
        "    - **Quantity**: The number of units ordered (numeric, e.g., 2).\n"
        "    - **NetWeight**: The net weight of the item (numeric, e.g., 0.466).\n"
        "    - **NetAmount**: The total net amount for the item (numeric, e.g., 144.30).\n"
        "    - **PricePerPiece**: Extract the price per piece of the item (numeric, e.g., 72.15).\n"
        "    - **GrossWeight**: Extract the gross weight of the item, if applicable.\n"
        
        "19. **NetWeight**: Total net weight of the order (if available).\n"
        "20. **GrossWeight**: Total gross weight of the order (if applicable).\n"
        "21. **NumberOfUnits**: Total number of units in the order (numeric).\n"
        "22. **NumberOfPallets**: Total number of pallets in the order (if applicable).\n"
        
        "### Important Notes:\n"
        "Ensure all extracted values match the exact values in it. "
        "Field value is always string with default value as empty string."
        "output json must exactly match above mentioned structure."
        "output of the above fields must not be None, NaN etc it should be always empty string eg. ("") "
    )

    # Create the prompt in the desired format
    prompt = [{"role": "user", "content": [image, question]}]

    return prompt

def load_image(image_data):
    """
    Loads an image from base64-encoded data, resizes it to the target size, and sets its DPI.

    Args:
        image_data (str): Base64-encoded image data.
        target_size (tuple): Target size to resize the image (default is 1600x1600 pixels).
        dpi (tuple): The target DPI for the image (default is 300x300 DPI).
    
    Returns:
        PIL.Image.Image: The loaded, resized image with the specified DPI.
    """
    # Decode the base64-encoded image data
    image_bytes = base64.b64decode(image_data)
    
    # Open the image from bytes and convert it to RGB mode
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    return image



def handler(event):
    """
    Main handler for the serverless function. Processes the input event to generate 
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
            image = load_image(image_data)
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

        return {"response": json.dumps(output)}

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
