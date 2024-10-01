import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import io
import base64
import runpod
import torch
import os

# Load model path from environment variables
model_name = os.getenv("MODEL_DIR", "./model")

# Load the tokenizer with trust_remote_code enabled
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
# Load the model on GPUs (balanced across GPUs)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

print("Model loaded in 16-bit precision.")

def generate_detailed_prompt(image, ocr_data):
    """
    Generates a compact prompt for training a GPT model to extract JSON invoice data from an image.

    Args:
        image_path (str): Path to the image file.
        json_data (dict): The JSON data representing the expected output.
        ocr_data (str): OCR text data extracted from the image.

    Returns:
        dict: A dictionary containing the detailed prompt structure.
    """
    question = (
        "You are provided with an image and OCR extracted text of an invoice PDF page.\n\n"
        "OCR data:\n"
        f"{ocr_data}\n\n"
        "Task: Use the image and OCR text to extract specific information and output as a JSON object with these fields:\n\n"
        "1. **OrderNumber**: Unique identifier for the order, usually prominent near order details.\n"
        "2. **InvoiceNumber**: Unique number assigned to the invoice, labeled like 'Invoice No.'.\n"
        "3. **BuyerName**: Full name of the buyer, typically found in 'Bill To' or 'Buyer' section.\n"
        "4. **BuyerAddress1**: First line of the buyer’s address, often the street or P.O. Box.\n"
        "5. **BuyerZipCode**: Postal code of the buyer's address.\n"
        "6. **BuyerCity**: City of the buyer's address.\n"
        "7. **BuyerCountry**: Country of the buyer’s address.\n"
        "8. **ReceiverName**: Full name of the receiver, typically in 'Ship To' or 'Receiver' section.\n"
        "9. **ReceiverAddress1**: First line of the receiver’s address.\n"
        "10. **ReceiverZipCode**: Postal code of the receiver's address.\n"
        "11. **ReceiverCity**: City of the receiver's address.\n"
        "12. **ReceiverCountry**: Country of the receiver’s address.\n"
        "13. **SellerName**: Seller or company name issuing the invoice, usually at the top.\n"
        "14. **OrderDate**: Order date in 'YYYY-MM-DD' format, near invoice/order details.\n"
        "15. **Currency**: Currency used, such as '$', 'EUR', or 'NOK'.\n"
        "16. **TermsOfDelCode**: Delivery terms code, typically in shipping or terms section.\n"
        "17. **OrderItems**: List of items with fields:\n"
        "   - **ArticleNumber**: Item's article number.\n"
        "   - **HsCode**: HS code or tariff number.\n"
        "   - **Description**: Item description.\n"
        "   - **CountryOfOrigin**: Item's country of origin.\n"
        "   - **Quantity**: Numerical quantity of the item.\n"
        "   - **GrossWeight**: Gross weight in numerical format.\n"
        "   - **Unit**: Unit of the item.\n"
        "   - **NetAmount**: Net amount, formatted to two decimal places.\n"
        "   - **PricePerPiece**: Price per piece, formatted to two decimal places.\n"
        "   - **NetWeight**: Net weight of the item.\n"
        "18. **ConsigneeCity**: City of the consignee.\n"
        "19. **ConsigneeCountry**: Country of the consignee.\n"
        "20. **ConsignorName**: Consignor's name.\n"
        "21. **ConsignorZipcode**: Consignor's postal code.\n"
        "22. **ConsignorCity**: Consignor's city.\n"
        "23. **ConsignorCountry**: Consignor's country.\n"
        "24. **SellerRef**: Seller reference number.\n"
        "25. **OrderMark**: Order mark.\n"
        "26. **SupplierOrderNo**: Supplier's order number.\n"
        "27. **PickupDate**: Date of pickup.\n"
        "28. **DeliveryDate**: Date of delivery.\n"
        "29. **HsCode2**: Secondary HS code.\n"
        "30. **ProcedureCode2**: Secondary procedure code.\n"
        "31. **SummaryInfo**: Summary of the order.\n"
        "32. **AdditionalInfo**: Additional information about the order.\n"
        "33. **Measurement**: Measurement of the order.\n"
        "34. **MeasurementUnit**: Unit of measurement.\n"
        "35. **InvoicedFreight**: Invoiced freight amount.\n"
        "36. **ExportCustomsId**: Export customs ID.\n"
        "37. **TransactionType**: Type of transaction for the order.\n"
        "38. **TransportMeansId**: ID of the transport means.\n"
        "39. **TransportMeansNationality**: Nationality of the transport means.\n"
        "40. **CustomsCreditOfficeNumber**: Customs credit office number.\n"
        "41. **DeclarationDate**: Declaration date.\n\n"
        "Ensure accuracy, follow the expected format, and return fields with empty strings or null if not available."
        "Ensure output is always pure json. Don't add any comments"
    )

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
