from azure.storage.blob import BlobServiceClient
import os
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import fitz
from PIL import Image
import pytesseract

def clean_numeric(value):
    """
    Cleans and normalizes numeric strings based on the following rules:
    1. Removes commas and any non-numeric characters except periods and minus signs.
    2. Keeps trailing zeros (e.g., '0.010' should remain as '0.010').
    3. If a value contains multiple numbers, picks the first valid number before a newline.
    4. If there are multiple periods, the first period is kept as the decimal, and the rest are removed.
    5. Returns empty string if input is empty.
    6. Raises an error if the value cannot be cleaned.

    Args:
        value (str): The value to be cleaned.

    Returns:
        str: Cleaned numeric string.

    Raises:
        ValueError: If the value cannot be cleaned to a valid number.
    """
    # Return empty string if value is empty
    if value is None or len(value) == 0:
        return ""

    if not isinstance(value, str):
        raise ValueError(f"Expected a string input, got {type(value)} instead.")

    # Remove commas and strip spaces
    value = value.replace(",", "").strip()

    # Remove spaces within the number to form a valid numeric string
    value = value.replace(" ", "")  # Remove spaces entirely

    # Check if there's a valid number before any newline (\n)
    if "\n" in value:
        value = value.split("\n")[0].strip()  # Take the part before the newline

    # Regular expression to match a valid number pattern (integer or decimal)
    match = re.search(r'-?\d+(\.\d+)?', value)

    if match:
        # Extract the first valid number
        cleaned_value = match.group(0)

        # Handle multiple periods by keeping the first one and removing any others
        if cleaned_value.count('.') > 1:
            cleaned_value = cleaned_value.split('.', 1)
            cleaned_value = cleaned_value[0] + '.' + re.sub(r'\.', '', cleaned_value[1])

        return cleaned_value
    else:
        raise ValueError(f"Could not clean the value: {value}")

def clean_article_number(value):
    """
    Cleans the ArticleNumber field by removing spaces and brackets.
    """
    if not value or not isinstance(value, str):
        return ""
    # Remove spaces and brackets
    return re.sub(r"[ \[\]\(\)]", "", value)

def clean_trailing_asterisk(value):
    """
    Removes a trailing asterisk (*) from a string value.

    Args:
        value (str): The input string.

    Returns:
        str: The cleaned string with no trailing asterisk.
    """
    if isinstance(value, str):
        return value.rstrip('*').strip()  # Remove trailing asterisk and any extra spaces
    return value 

def convert_to_order_structure(input_json):
    """
    Converts the input JSON to a specified order structure, ensuring validation of required fields.

    Args:
        input_json (dict): The input JSON data.

    Returns:
        dict: The converted JSON in the specified order structure.
    """
    order_structure = {
        "OrderNumber": clean_numeric(input_json.get("OrderNumber", "")),
        "InvoiceNumber": clean_numeric(input_json.get("InvoiceNumber", "")),
        "BuyerName": input_json.get("BuyerName", ""),
        "BuyerAddress1": input_json.get("BuyerAddress1", ""),
        "BuyerZipCode": clean_numeric(input_json.get("BuyerZipCode", "")),
        "BuyerCity": input_json.get("BuyerCity", ""),
        "BuyerCountry": input_json.get("BuyerCountry", ""),
        "ReceiverName": input_json.get("ReceiverName", ""),
        "ReceiverAddress1": input_json.get("ReceiverAddress1", ""),
        "ReceiverZipCode": clean_numeric(input_json.get("ReceiverZipCode", "")),
        "ReceiverCity": input_json.get("ReceiverCity", ""),
        "ReceiverCountry": input_json.get("ReceiverCountry", ""),
        "SellerName": input_json.get("SellerName", ""),
        "NetAmount": clean_numeric(input_json.get("NetAmount", "")),
        "OrderDate": input_json.get("OrderDate", ""),
        "Currency": input_json.get("Currency", ""),
        "TermsOfDelCode": input_json.get("TermsOfDelCode", ""),
        "OrderItems": [
            {
                "ArticleNumber": clean_article_number(item.get("ArticleNumber", "")),
                "Description": item.get("Description", ""),
                "HsCode": clean_numeric(item.get("HsCode", "")),
                "CountryOfOrigin": clean_trailing_asterisk(item.get("CountryOfOrigin", "")),
                "Quantity": clean_numeric(item.get("Quantity", "")),
                "NetWeight": clean_numeric(item.get("NetWeight", "")),
                "NetAmount": clean_numeric(item.get("NetAmount", "")),
                "PricePerPiece": clean_numeric(item.get("PricePerPiece", "")),
                "EclEuNO": clean_numeric(item.get("EclEuNO", ""))
            }
            for item in input_json.get("OrderItems", [])
        ],
        "NetWeight": clean_numeric(input_json.get("NetWeight", "")),
        "NumberOfUnits": clean_numeric(input_json.get("NumberOfUnits", ""))
    }

    return order_structure

def clean_dataset(input_file, output_file):
    """
    Cleans the dataset based on defined rules, converts to a structured format,
    and saves the cleaned dataset to the output file.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Clean and convert data
    cleaned_data = {}
    for key, record in data.items():
        cleaned_data[key] = convert_to_order_structure(record)

    # Save cleaned dataset
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=4)
    print(f"Cleaned and converted dataset saved to {output_file}")

def download_blob_folder(sas_url, folder_path, output_directory):
    # Parse the SAS URL to extract the storage account, container, and folder path
    url_parts = urlparse(sas_url)
    account_url = f"https://{url_parts.netloc}"
    container_name = url_parts.path.split('/')[1]

    # Initialize the BlobServiceClient using the account URL and SAS token
    blob_service_client = BlobServiceClient(account_url=account_url, credential=url_parts.query)

    # Get the container client
    container_client = blob_service_client.get_container_client(container_name)

    # List all blobs in the specified folder
    blobs = container_client.list_blobs(name_starts_with=folder_path)

    # Create local directories as needed and download each blob
    for blob in blobs:
        blob_name = blob.name
        local_file_path = os.path.join(output_directory, blob_name)

        # Ensure the local directory exists
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download the blob
        with open(local_file_path, "wb") as download_file:
            download_stream = container_client.download_blob(blob_name)
            download_file.write(download_stream.readall())

        print(f"Downloaded: {blob_name} to {local_file_path}")

# Convert PDF Page to Image

def convert_pdf_to_images(pickup_id, pdf_path, image_output_dir, dpi=600):
    """
    Converts a PDF file into images for each page.

    Args:
        pdf_path (str): Path to the PDF file.
        image_output_dir (str): Directory to save the images.
        dpi (int): DPI resolution for converting PDF pages.
        target_size (tuple): Target size for output images (width, height).
    """
    os.makedirs(image_output_dir, exist_ok=True)
    pdf_document = fitz.open(pdf_path)

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        image_path = os.path.join(image_output_dir, f"{pickup_id}_{page_num + 1:03d}.png")
        image.save(image_path)
        print(f"Saved: {image_path}")


def create_raw_data(pickup_id, pdf_dir, json_dir, output_file, image_output_dir):
    """
    Creates raw data by mapping images to extracted JSON data page-wise for a specific pickup ID.
    Handles cases where PDFs and JSON files are located in nested folders.

    Args:
        pickup_id (str): The pickup ID to process.
        pdf_dir (str): Directory containing the PDF files.
        json_dir (str): Directory containing extracted JSON data.
        output_file (str): Path to save the raw data JSON file.
        image_output_dir (str): Directory to save images.
    """
    raw_data = {}
    pdf_folder = os.path.join(pdf_dir, pickup_id)

    if not os.path.exists(pdf_folder):
        print(f"PDF folder not found for Pickup ID {pickup_id}. Skipping...")
        return

    # Search for all PDFs in the folder and its subfolders
    pdf_files = []
    for root, _, files in os.walk(pdf_folder):
        pdf_files.extend([os.path.join(root, f) for f in files if f.endswith(".pdf")])

    # Throw an error if more than one PDF is found
    if len(pdf_files) > 1:
        raise ValueError(f"More than one PDF found for Pickup ID {pickup_id} in {pdf_folder}: {pdf_files}")
    elif not pdf_files:
        print(f"No PDF found for Pickup ID {pickup_id}. Skipping...")
        return

    pdf_path = pdf_files[0]  # Take the single found PDF
    json_root = os.path.join(json_dir, pickup_id)
    if not os.path.exists(json_root):
        print(f"Extracted JSON directory not found for Pickup ID {pickup_id}. Skipping...")
        return

    pdf_image_dir = os.path.join(image_output_dir, pickup_id)
    os.makedirs(pdf_image_dir, exist_ok=True)

    # Open the PDF using fitz
    doc = fitz.open(pdf_path)
    num_pages = doc.page_count  # Get the number of pages

    convert_pdf_to_images(pickup_id, pdf_path, pdf_image_dir)

    # Initialize raw_data for the current pickup_id
    raw_data[pickup_id] = {}

    # Process each page of the PDF
    for page_num in range(1, num_pages + 1):
        # Search for the JSON file for this specific page number
        json_files = []
        for root, _, files in os.walk(json_root):
            json_files.extend([os.path.join(root, f) for f in files if re.match(f"{page_num}__.*\\.json", f)])

        if not json_files:
            print(f"No JSON found for Pickup ID {pickup_id}, page {page_num}. Skipping...")
            continue

        # Use the first matching JSON file
        page_json_path = json_files[0]
        with open(page_json_path, "r", encoding="utf-8") as f:
            extracted_data = json.load(f)

        properties = extracted_data.get("Properties", {})
        if not properties:
            print(f"No properties in JSON for Pickup ID {pickup_id}, page {page_num}. Skipping...")
            continue

        

        # Add the image and properties data to the raw_data dictionary
        raw_data[pickup_id][page_num] = {
            "image_path": f"{image_output_dir}/{pickup_id}/{pickup_id}_{page_num + 1:03d}.png",
            "properties":convert_to_order_structure(properties)
        }

    # Save the raw data for this pickup_id to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=4)

    print(f"Raw data for Pickup ID {pickup_id} saved to {output_file}")

def process_pickup_id(sas_url, pickup_id, base_output_directory):
    download_blob_folder(sas_url, pickup_id, os.path.join(base_output_directory, pickup_id))


def create_training_data_entry(image_path, question, answer, id):
    return {
        "id": id,
        "image": image_path,
        "conversations": [
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
    }

def extract_text_from_image(image):
    """Extract text from an image derived from the PDF."""
    try:
        text = pytesseract.image_to_string(image)
        print(f"Extracted text length: {len(text)} characters.")
        return text
    except Exception as e:
        print(f"Error during text extraction: {e}")
        raise RuntimeError(f"Error during text extraction: {e}")

def generate_prompt(image_path):
    """Create the detailed prompt for the model."""
    try:
        image = Image.open(image_path)
        ocr_data = extract_text_from_image(image)
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
        
        return question
    except Exception as e:
        print(e)



def create_training_data(raw_data_path, output_file):
    """
    Creates training data from the raw data file.

    Args:
        raw_data_path (str): Path to the raw data JSON file.
        output_file (str): Path to save the generated training data.

    Returns:
        list: A list of training data entries.
    """
    training_data = []
    with open(raw_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    for pickup_id, page_data in raw_data.items():
        print(f"Processing Pickup ID: {pickup_id}")
        for page_num, data in page_data.items():
            try:
                print(f"Processing Page {page_num} for Pickup ID {pickup_id}")
                image_path = data.get("image_path")
                properties = data.get("properties", {})
                question = generate_prompt(image_path)
                answer = json.dumps(properties, indent=4)
                training_entry = create_training_data_entry(image_path, question, answer, f"{pickup_id}00000000{page_num}")
                training_data.append(training_entry)
            except Exception as e:
                print(f"Error processing page {page_num} for Pickup ID {pickup_id}: {e}")

    # Save the training data to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=4)
    print(f"Training data saved to {output_file}")
    return training_data


if __name__ == "__main__":
    pdf_connection_string = "https://saascustomsportalstorage.blob.core.windows.net/pickupfiles?sp=rli&st=2025-01-16T15:04:44Z&se=2026-01-16T23:04:44Z&sv=2022-11-02&sr=c&sig=GmbLCUpv%2F7TsLxvzWS0Y%2BEfYlcHxtxTzgz4hwHJN12c%3D"
    extracted_data_connection_string = "https://saascustomsportalstorage.blob.core.windows.net/processedpickupfiles?sp=rli&st=2025-01-16T15:04:11Z&se=2026-01-16T23:04:11Z&sv=2022-11-02&sr=c&sig=uupon7JS1M4d99zcToMjQvlzj9LiXqxqgdANOtqGKhs%3D"
    pickupIds = ["31929"]  # Add more pickup IDs as needed
    base_output_directory = "./data/pdf"
    executed_output_directory = "./data/raw_ouput"
    raw_data_output = "./data/raw_data.json"
    train_data_output = "./data/train_data.json"
    image_dir = "./data/image"

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=24) as executor:  # Adjust max_workers based on your system's resources
        futures = [
            executor.submit(process_pickup_id, pdf_connection_string, pickup_id, base_output_directory)
            for pickup_id in pickupIds
        ]
    
    with ThreadPoolExecutor(max_workers=24) as executor:  # Adjust max_workers based on your system's resources
        futures = [
            executor.submit(process_pickup_id, extracted_data_connection_string, pickup_id + "/ExtractedData", executed_output_directory)
            for pickup_id in pickupIds
        ]   

    create_raw_data("31929", base_output_directory, executed_output_directory, raw_data_output, image_dir)
    create_training_data(raw_data_output, train_data_output)
    print("All downloads are complete.")
