import os
import json
from azure.storage.blob import BlobServiceClient
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import pytesseract
from PIL import Image
import fitz
import re


def download_blob_folder(sas_url, folder_path, output_directory):
    """
    Downloads all blobs from a specified folder in an Azure Blob Storage container.

    Args:
        sas_url (str): The SAS URL for accessing Azure Blob Storage.
        folder_path (str): The folder path in the container to download.
        output_directory (str): The local directory to save the downloaded blobs.
    """
    url_parts = urlparse(sas_url)
    account_url = f"https://{url_parts.netloc}"
    container_name = url_parts.path.split('/')[1]

    blob_service_client = BlobServiceClient(account_url=account_url, credential=url_parts.query)
    container_client = blob_service_client.get_container_client(container_name)
    blobs = container_client.list_blobs(name_starts_with=folder_path)

    for blob in blobs:
        blob_name = blob.name
        local_file_path = os.path.join(output_directory, blob_name)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        with open(local_file_path, "wb") as download_file:
            download_stream = container_client.download_blob(blob_name)
            download_file.write(download_stream.readall())

        print(f"Downloaded: {blob_name} to {local_file_path}")


def create_raw_data(pickup_id, pdf_dir, json_dir, output_file, image_output_dir):
    """
    Creates raw data by mapping images to extracted JSON data page-wise for a specific pickup ID.

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
            "image_path": f"{pdf_image_dir}/{pickup_id}_{page_num:03d}.png",
            "properties": properties
        }

    # Save the raw data for this pickup_id to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=4)

    print(f"Raw data for Pickup ID {pickup_id} saved to {output_file}")


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
    training_data = []
    with open(raw_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    for pickup_id, page_data in raw_data.items():
        for page_num, data in page_data.items():
            try:
                image_path = data.get("image_path")
                properties = data.get("properties", {})
                question = generate_prompt(image_path)
                answer = json.dumps(properties, indent=4)
                training_entry = {
                    "id": f"{pickup_id}00000000{page_num}",
                    "image": image_path,
                    "conversations": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                }
                training_data.append(training_entry)
            except Exception as e:
                print(f"Error processing page {page_num} for Pickup ID {pickup_id}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=4)
    print(f"Training data saved to {output_file}")
    return training_data


if __name__ == "__main__":
    extracted_data_connection_string = "https://saascustomsportalstorage.blob.core.windows.net/processedpickupfiles?sp=rli&st=2025-01-16T15:04:11Z&se=2026-01-16T23:04:11Z&sv=2022-11-02&sr=c&sig=uupon7JS1M4d99zcToMjQvlzj9LiXqxqgdANOtqGKhs%3D"
    pickup_ids = ["31929"]
    executed_output_directory = "./data/raw_output"
    raw_data_output = "./data/raw_data.json"
    train_data_output = "./data/train_data.json"
    image_dir = "./data/image"
    base_output_directory = "./data/pdf"

    # Step 1: Download JSON Data
    with ThreadPoolExecutor(max_workers=24) as executor:
        for pickup_id in pickup_ids:
            executor.submit(download_blob_folder, extracted_data_connection_string, f"{pickup_id}/ExtractedData", executed_output_directory)

    # Step 2: Create Raw Data
    for pickup_id in pickup_ids:
        create_raw_data(pickup_id, base_output_directory, executed_output_directory, raw_data_output, image_dir)

    # Step 3: Create Training Data
    create_training_data(raw_data_output, train_data_output)

    print("JSON download, raw data creation, and training data generation complete.")
