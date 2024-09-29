import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import io
import base64
import runpod
import torch
import json

# Load model path from environment variables
model_path = "/workspace/model/model_repo"

# Load tokenizer and model from the pre-downloaded directory
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print("Tokenizer loaded.")

print("Loading model in 16-bit precision onto GPU...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="balanced"  # Automatically balance model across available GPUs
).cuda().eval()

print("Model loaded in 16-bit precision.")

def generate_detailed_prompt(image, ocr_data):
    """
    Generates a detailed prompt for training a GPT model to extract JSON invoice data from an image.

    Args:
        image_path (str): Path to the image file.
        json_data (dict): The JSON data representing the expected output.
        ocr_data (str): OCR text data extracted from the image.

    Returns:
        dict: A dictionary containing the detailed prompt structure.
    """


    # Create a detailed description of the task
    question = (
        "<image> You are provided with an image and OCR extracted text of an invoice PDF page.\n\n"
        "OCR data starts here:\n"
        f"{ocr_data}\n\n"
        "OCR data ends here.\n\n"
        "Task: Use the provided image and OCR text to extract specific information from the invoice image and output it as a JSON object with the following fields:\n\n"

        "1. **OrderNumber**: Extract the order number from the invoice. It is usually a unique identifier for the order, often displayed prominently at the top or near the order details section.\n"
        "2. **InvoiceNumber**: Extract the invoice number, which is a unique number assigned to this invoice. Look for labels like 'Invoice No.', 'Invoice Number', or similar.\n"
        "3. **BuyerName**: Extract the full name of the buyer. This information is typically found in the 'Bill To' or 'Buyer' section of the invoice.\n"
        "4. **BuyerAddress1**: Extract the first line of the buyer's address. It usually includes the street address or P.O. Box.\n"
        "5. **BuyerZipCode**: Extract the postal code for the buyer’s address. This is usually found alongside the address details.\n"
        "6. **BuyerCity**: Extract the city of the buyer's address. It is often located near the postal code and address details.\n"
        "7. **BuyerCountry**: Extract the country of the buyer’s address. Look for labels such as 'Country' or 'Country of Residence'.\n"
        "8. **ReceiverName**: Extract the full name of the receiver. This is typically found in the 'Ship To' or 'Receiver' section of the invoice.\n"
        "9. **ReceiverAddress1**: Extract the first line of the receiver’s address, similar to how you would extract the buyer’s address.\n"
        "10. **ReceiverZipCode**: Extract the postal code for the receiver's address. This is usually close to the address information.\n"
        "11. **ReceiverCity**: Extract the city for the receiver’s address. It should be near the postal code and address details.\n"
        "12. **ReceiverCountry**: Extract the country of the receiver's address. Look for similar labels as for the buyer's country.\n"
        "13. **SellerName**: Extract the name of the seller or company issuing the invoice. This information is often found at the top of the invoice or in the 'Sold By' section.\n"
        "14. **OrderDate**: Extract the order date from the invoice. It is typically listed near the invoice number or in the order details section, formatted as 'YYYY-MM-DD'.\n"
        "15. **Currency**: Extract the currency used for the invoice. Look for symbols or abbreviations such as '$', 'EUR', or 'NOK'.\n"
        "16. **TermsOfDelCode**: Extract the delivery terms code or description. This might be found in the shipping or terms section of the invoice, describing how the delivery is handled.\n"
        "17. **OrderItems**: Extract each item listed on the invoice as a list of objects, each containing the following fields:\n"
        "    - **ArticleNumber**: Extract the article number for each item, often found next to the item description.\n"
        "    - **HsCode**: Extract the HS code or tariff number for the item, if available. This is typically found in the item description or details section.\n"
        "    - **Description**: Extract the description of the item as listed on the invoice.\n"
        "    - **CountryOfOrigin**: Extract the country of origin for each item, if specified.\n"
        "    - **Quantity**: Extract the quantity of each item, listed in numerical format.\n"
        "    - **GrossWeight**: Extract the GrossWeight of each item, listed in numerical format.\n"
        "    - **Unit**: Extract the Unit of each item, listed in numerical format.\n"
        "    - **NetAmount**: Extract the net amount for each item, formatted to include two decimal places.\n"
        "    - **PricePerPiece**: Extract the price per piece of each item, formatted to include two decimal places.\n"
        "    - **NetWeight**: Extract the net weight of each item if available.\n"
        "18. **ConsigneeCity**: Extract the city of the consignee.\n"
        "19. **ConsigneeCountry**: Extract the country of the consignee (if available).\n"
        "20. **ConsignorName**: Extract the consignor's name.\n"
        "21. **ConsignorZipcode**: Extract the consignor's postal code.\n"
        "22. **ConsignorCity**: Extract the consignor's city.\n"
        "23. **ConsignorCountry**: Extract the consignor's country.\n"
        "24. **SellerRef**: Extract the seller reference number.\n"
        "25. **OrderMark**: Extract the order mark.\n"
        "26. **SupplierOrderNo**: Extract the supplier's order number.\n"
        "27. **PickupDate**: Extract the date of pickup (if available).\n"
        "28. **DeliveryDate**: Extract the delivery date (if available).\n"
        "29. **HsCode2**: Extract the secondary HS code (if available).\n"
        "30. **ProcedureCode2**: Extract the procedure code (if available).\n"
        "31. **SummaryInfo**: Extract the summary information from the order.\n"
        "32. **AdditionalInfo**: Extract additional information from the order.\n"
        "33. **Measurement**: Extract the measurement for the order.\n"
        "34. **MeasurementUnit**: Extract the unit of measurement for the order.\n"
        "35. **InvoicedFreight**: Extract the invoiced freight amount.\n"
        "36. **ExportCustomsId**: Extract the export customs ID (if available).\n"
        "37. **TransactionType**: Extract the transaction type for the order.\n"
        "38. **TransportMeansId**: Extract the transport means ID.\n"
        "39. **TransportMeansNationality**: Extract the nationality of the transport means.\n"
        "40. **CustomsCreditOfficeNumber**: Extract the customs credit office number.\n"
        "41. **DeclarationDate**: Extract the declaration date.\n"

        "Ensure that the extracted values are accurate and follow the expected format. If a field is not found in the invoice, include it in the JSON output with an empty string or null value."
    )

    # Create the prompt in the desired format
    prompt = [{"role": "user", "content": [image, question]}]

    return prompt

def handler(event):
    try:
        # Extract the prompt and image data from the event
        event_data = event.get("input")
        image_data = event_data.get("image", None)
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        ocr_data = event_data.get("ocr_data", "")
        prompt = generate_detailed_prompt(image, ocr_data)

        # Tokenize and generate response
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
    print("Health check hit.")
    return {"status": "ok"}

# Start RunPod serverless function
runpod.serverless.start({"handler": handler})
