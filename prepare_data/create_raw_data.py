import os
import json
import re
from urllib.parse import urlparse
from azure.storage.blob import BlobServiceClient
from concurrent.futures import ThreadPoolExecutor
import fitz  # PyMuPDF
from PIL import Image
import requests

# Constants
BLOB_SAS_URL = "https://saascustomsportalstorage.blob.core.windows.net/processedpickupfiles?sp=rli&st=2025-01-16T15:04:11Z&se=2026-01-16T23:04:11Z&sv=2022-11-02&sr=c&sig=uupon7JS1M4d99zcToMjQvlzj9LiXqxqgdANOtqGKhs%3D"
PDF_BLOB_URL = "https://saascustomsportalstorage.blob.core.windows.net/pickupfiles?sp=rli&st=2025-01-16T15:04:44Z&se=2026-01-16T23:04:44Z&sv=2022-11-02&sr=c&sig=GmbLCUpv%2F7TsLxvzWS0Y%2BEfYlcHxtxTzgz4hwHJN12c%3D"
FIELDS_TO_REMOVE = ["PageNumber", "ItemNumber"]
PICKUP_MAP = {"xylem": ["64189","64191","64461","66062"],
              "tarket": ["80482","80748","81006","79341"],
              "dentalspar": ["71823","72110","72951","78053","79886"]}
XYLEM_TOKEN = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjUiLCJjb21wYW55SWQiOiIxMjM5IiwibmFtZSI6Ilh5bGVtIiwicm9sZSI6IkN1c3RvbXNQb3J0YWwiLCJuYmYiOjE3Mzc2MTY2MDgsImV4cCI6MTc0NTM2NjQwMCwiaWF0IjoxNzM3NjE2NjA4LCJpc3MiOiJodHRwczovL3RyYW5zcG9ydGx5c3FsYXBpdjIuYXp1cmV3ZWJzaXRlcy5uZXQiLCJhdWQiOiJodHRwczovL3RyYW5zcG9ydGx5c3FsYXBpdjIuYXp1cmV3ZWJzaXRlcy5uZXQifQ.k7yB5Z2MDZcW4U-JrB4A61dbGG4rzdnB8tjusbLtcpY"
TARKETT_TOKEN = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjQiLCJjb21wYW55SWQiOiIxMTg4IiwibmFtZSI6IlRhcmtldHQiLCJyb2xlIjoiQ3VzdG9tc1BvcnRhbCIsIm5iZiI6MTczNzYxNjU3MywiZXhwIjoxNzQ1MzY2NDAwLCJpYXQiOjE3Mzc2MTY1NzMsImlzcyI6Imh0dHBzOi8vdHJhbnNwb3J0bHlzcWxhcGl2Mi5henVyZXdlYnNpdGVzLm5ldCIsImF1ZCI6Imh0dHBzOi8vdHJhbnNwb3J0bHlzcWxhcGl2Mi5henVyZXdlYnNpdGVzLm5ldCJ9.RDGKCmBxgyyqMoPaYiQt_5CulL8dz3Uc2_IiR08EeaA"
DENTALSPAR_TOKEN = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjYiLCJjb21wYW55SWQiOiIxMjIxIiwibmFtZSI6IkRFTlRBTFNQQVIgQVMiLCJyb2xlIjoiQ3VzdG9tc1BvcnRhbCIsIm5iZiI6MTczNzYxNjcwOSwiZXhwIjoxNzQ1MzY2NDAwLCJpYXQiOjE3Mzc2MTY3MDksImlzcyI6Imh0dHBzOi8vdHJhbnNwb3J0bHlzcWxhcGl2Mi5henVyZXdlYnNpdGVzLm5ldCIsImF1ZCI6Imh0dHBzOi8vdHJhbnNwb3J0bHlzcWxhcGl2Mi5henVyZXdlYnNpdGVzLm5ldCJ9.YI4DmA5iJonABF9jTIe2TqIvhLkVhho75ZP6hhlKoho"
TOKENS = {
    "xylem": f"{XYLEM_TOKEN}",
    "tarket": f"{TARKETT_TOKEN}",
    "dentalspar": f"{DENTALSPAR_TOKEN}"
}
PDF_OUTPUT_DIR = ".././data/pdf"
IMAGE_OUTPUT_DIR = ".././data/image"
EXTRACTED_OUTPUT_DIR = ".././data/raw_output"
RAW_DATA_OUTPUT = ".././data/raw_data.json"
MAX_WORKERS = 100
API_URL = "https://transportlysqlapiv2.azurewebsites.net/public/v1/OrderData"

# Function to fetch API data
def fetch_api_data(pickup_id, company):
    url = f"{API_URL}?pickupId={pickup_id}"
    headers = {
        "accept": "text/plain",
        "Authorization": TOKENS.get(company, "")
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch API data for Pickup ID {pickup_id}")
        return {}

# Function to load JSON from a file
def load_json(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return {}
    
    with open(file_path, "r", encoding="utf-8") as file:
        try:
            return json.load(file)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return {}

# Function to download blobs from a container
def download_blob_folder(sas_url, folder_path, output_directory):
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

# Function to convert PDF to images
def convert_pdf_to_images(pickup_id, pdf_path, image_output_dir, dpi=600):
    os.makedirs(image_output_dir, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    image_paths = {}
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        image_path = os.path.join(image_output_dir, f"{pickup_id}_{page_num + 1:03d}.png")
        image.save(image_path)
        image_paths[str(page_num + 1)] = image_path  # Store page-wise image path
        print(f"Saved: {image_path}")
    
    return image_paths

# Function to recursively remove fields from a JSON object
def remove_fields(obj, fields):
    if isinstance(obj, dict):
        return {k: remove_fields(v, fields) for k, v in obj.items() if k not in fields}
    elif isinstance(obj, list):
        return [remove_fields(i, fields) for i in obj]
    return obj

# Function to map extracted properties to order structure
def convert_to_order_structure(properties, db_data):
    return remove_fields({
        "OrderNumber": properties.get("OrderNumber", ""),
        "InvoiceNumber": properties.get("InvoiceNumber", ""),
        "BuyerName": db_data.get("BuyerName") if properties.get("BuyerName", "") == "" else properties.get("BuyerName", ""),
        "BuyerAddress1": db_data.get("BuyerAddress1") if properties.get("BuyerAddress1", "") == "" else properties.get("BuyerAddress1", ""),
        "BuyerZipCode": db_data.get("BuyerZipCode") if properties.get("BuyerZipCode", "") == "" else properties.get("BuyerZipCode", ""),
        "BuyerCity": db_data.get("BuyerCity") if properties.get("BuyerCity", "") == "" else properties.get("BuyerCity", ""),
        "BuyerCountry": db_data.get("BuyerCountry") if properties.get("BuyerCountry", "") == "" else properties.get("BuyerCountry", ""),
        "ReceiverName": db_data.get("ReceiverName") if properties.get("ReceiverName", "") == "" else properties.get("ReceiverName", ""),
        "ReceiverAddress1": db_data.get("ReceiverAddress1") if properties.get("ReceiverAddress1", "") == "" else properties.get("ReceiverAddress1", ""),
        "ReceiverZipCode": db_data.get("ReceiverZipCode") if properties.get("ReceiverZipCode", "") == "" else properties.get("ReceiverZipCode", ""),
        "ReceiverCity": db_data.get("ReceiverCity") if properties.get("ReceiverCity", "") == "" else properties.get("ReceiverCity", ""),
        "ReceiverCountry": db_data.get("ReceiverCountry") if properties.get("ReceiverCountry", "") == "" else properties.get("ReceiverCountry", ""),
        "SellerName": db_data.get("SellerName") if properties.get("SellerName", "") == "" else properties.get("SellerName", ""),
        "NetAmount": db_data.get("NetAmount", ""),
        "OrderDate": db_data.get("OrderDate") if properties.get("OrderDate", "") == "" else properties.get("OrderDate", ""),
        "Currency": db_data.get("Currency", ""),
        "TermsOfDelCode": db_data.get("TermsOfDelCode", ""),
        "OrderItems": db_data.get("Items", []),
        "NetWeight": db_data.get("NetWeight", ""),
        "NumberOfUnits": db_data.get("NumberOfUnits", "")
    }, FIELDS_TO_REMOVE)

# Function to create raw data from extracted JSON
raw_data = {}
def create_raw_data(pickup_id, company, json_dir, image_paths):
    api_data = fetch_api_data(pickup_id, company)
    json_root = os.path.join(json_dir, pickup_id, "ExtractedData")

    if not os.path.exists(json_root):
        print(f"Extracted JSON directory not found for Pickup ID {pickup_id}. Skipping...")
        return

    raw_data[pickup_id] = {}
    page_wise_data = {}
    
    for order in api_data:
        for item in order["Order"]["Items"]:
            page_number = str(item["PageNumber"])
            if page_number not in page_wise_data:
                order_copy = json.loads(json.dumps(order))
                order_copy["Order"]["Items"] = []
                page_wise_data[page_number] = order_copy
            page_wise_data[page_number]["Order"]["Items"].append(item)
    for root, _, files in os.walk(json_root):
        for file in files:
            if re.match(r"\d+_.*\.json", file):
                page_num = file.split("_")[0]
                page_json_path = os.path.join(root, file)
                extracted_data = load_json(page_json_path)
                properties = extracted_data.get("Properties", {})

                data = convert_to_order_structure(properties, page_wise_data.get(page_num, {}).get("Order", {}))
                if data == {} or data["OrderItems"] == []:
                    continue
                raw_data[pickup_id][page_num] = {
                    "image_path": image_paths.get(pickup_id).get(page_num, ""),
                    "data": data
                }
    

# Main Execution
if __name__ == "__main__":
    # pickup_ids = []
    # for company, pickup_id_list in PICKUP_MAP.items():
    #     pickup_ids.extend(pickup_id_list)

    
    # # Step 1: Download PDF files
    # with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    #     for pickup_id in pickup_ids:
    #         executor.submit(download_blob_folder, PDF_BLOB_URL, pickup_id, PDF_OUTPUT_DIR)
    
    # # Step 2: Convert PDFs to images
    # image_paths_map = {}
    # for pickup_id in pickup_ids:
    #     pdf_folder = os.path.join(PDF_OUTPUT_DIR, pickup_id)
    #     for pdf_file in os.listdir(pdf_folder):
    #         if pdf_file.endswith(".pdf"):
    #             pdf_path = os.path.join(pdf_folder, pdf_file)
    #             image_paths_map[pickup_id] = convert_pdf_to_images(pickup_id, pdf_path, IMAGE_OUTPUT_DIR)
    # with open(".././data/image_path_map.json", "w", encoding="utf-8") as f:
    #     json.dump(image_paths_map, f, indent=4)
    
    # # Step 3: Download extracted JSON files
    # with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    #     for pickup_id in pickup_ids:
    #         executor.submit(download_blob_folder, BLOB_SAS_URL, pickup_id + "/ExtractedData", EXTRACTED_OUTPUT_DIR)
    
    # Step 4: Process raw data
    with open(".././data/image_path_map.json", "r", encoding="utf-8") as file:
        image_paths_map = json.load(file)
    for company, pickup_ids in PICKUP_MAP.items():
        for pickup_id in pickup_ids:
            create_raw_data(pickup_id, company, EXTRACTED_OUTPUT_DIR, image_paths_map)

    with open(RAW_DATA_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=4)
    
    
    print("Raw data creation complete.")
