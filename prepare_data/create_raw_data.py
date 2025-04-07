import os
import json
import re
from urllib.parse import urlparse
from azure.storage.blob import BlobServiceClient
from concurrent.futures import ThreadPoolExecutor
import fitz  # PyMuPDF
from PIL import Image
import requests
from concurrent.futures import ProcessPoolExecutor, as_completed

# Constants
BLOB_SAS_URL = "https://saascustomsportalstorage.blob.core.windows.net/processedpickupfiles?sp=rli&st=2025-01-16T15:04:11Z&se=2026-01-16T23:04:11Z&sv=2022-11-02&sr=c&sig=uupon7JS1M4d99zcToMjQvlzj9LiXqxqgdANOtqGKhs%3D"
PDF_BLOB_URL = "https://saascustomsportalstorage.blob.core.windows.net/pickupfiles?sp=rli&st=2025-01-16T15:04:44Z&se=2026-01-16T23:04:44Z&sv=2022-11-02&sr=c&sig=GmbLCUpv%2F7TsLxvzWS0Y%2BEfYlcHxtxTzgz4hwHJN12c%3D"
FIELDS_TO_REMOVE = ["PageNumber", "ItemNumber"]
PICKUP_MAP = {
    "tarket": ["116083", "152743", "148871", "153876","153870","153832","153520","153517","153462","153456","153115","153108","153100","153098","153085","152767","152743","152739","152681","152678"," ","Xylem "," ","153702","153695","153319","152971","152608","152228","152227","152124","151907","151530","151529","151275","151168","150796","150403","150083","149740","149738","149665","149660", "148842", "148803", "148774", "148450", "147823", "144425", "143419", "146791", "146473", "116064", "116056", "115803", "115786", "115736"],
    "xylem": ["149740", "149738", "149377", "149032", "149000", "148374", "147658", "143949", "149377", "149032", "149000", "148782", "148374", "148224", "148010", "147658", "147656", "146377", "145776", "145370"]
    }
XYLEM_TOKEN = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjUiLCJjb21wYW55SWQiOiIxMjM5IiwibmFtZSI6Ilh5bGVtIiwicm9sZSI6IkN1c3RvbXNQb3J0YWwiLCJuYmYiOjE3Mzc2MTY2MDgsImV4cCI6MTc0NTM2NjQwMCwiaWF0IjoxNzM3NjE2NjA4LCJpc3MiOiJodHRwczovL3RyYW5zcG9ydGx5c3FsYXBpdjIuYXp1cmV3ZWJzaXRlcy5uZXQiLCJhdWQiOiJodHRwczovL3RyYW5zcG9ydGx5c3FsYXBpdjIuYXp1cmV3ZWJzaXRlcy5uZXQifQ.k7yB5Z2MDZcW4U-JrB4A61dbGG4rzdnB8tjusbLtcpY"
TARKETT_TOKEN = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjQiLCJjb21wYW55SWQiOiIxMTg4IiwibmFtZSI6IlRhcmtldHQiLCJyb2xlIjoiQ3VzdG9tc1BvcnRhbCIsIm5iZiI6MTczNzYxNjU3MywiZXhwIjoxNzQ1MzY2NDAwLCJpYXQiOjE3Mzc2MTY1NzMsImlzcyI6Imh0dHBzOi8vdHJhbnNwb3J0bHlzcWxhcGl2Mi5henVyZXdlYnNpdGVzLm5ldCIsImF1ZCI6Imh0dHBzOi8vdHJhbnNwb3J0bHlzcWxhcGl2Mi5henVyZXdlYnNpdGVzLm5ldCJ9.RDGKCmBxgyyqMoPaYiQt_5CulL8dz3Uc2_IiR08EeaA"
DENTALSPAR_TOKEN = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjYiLCJjb21wYW55SWQiOiIxMjIxIiwibmFtZSI6IkRFTlRBTFNQQVIgQVMiLCJyb2xlIjoiQ3VzdG9tc1BvcnRhbCIsIm5iZiI6MTczNzYxNjcwOSwiZXhwIjoxNzQ1MzY2NDAwLCJpYXQiOjE3Mzc2MTY3MDksImlzcyI6Imh0dHBzOi8vdHJhbnNwb3J0bHlzcWxhcGl2Mi5henVyZXdlYnNpdGVzLm5ldCIsImF1ZCI6Imh0dHBzOi8vdHJhbnNwb3J0bHlzcWxhcGl2Mi5henVyZXdlYnNpdGVzLm5ldCJ9.YI4DmA5iJonABF9jTIe2TqIvhLkVhho75ZP6hhlKoho"
TOKENS = {
    "xylem": f"{XYLEM_TOKEN}",
    "tarket": f"{TARKETT_TOKEN}",
    "dentalspar": f"{DENTALSPAR_TOKEN}"
}
PDF_OUTPUT_DIR = "./data/pdf"
IMAGE_OUTPUT_DIR = "./data/image"
EXTRACTED_OUTPUT_DIR = "./data/extractedData"
RAW_DATA_OUTPUT = "./data/raw_dental_data.json"
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

def download_blob_folder(sas_url, pickup_id, output_directory, max_workers=8):
    url_parts = urlparse(sas_url)
    account_url = f"https://{url_parts.netloc}"
    container_name = url_parts.path.split('/')[1]

    blob_service_client = BlobServiceClient(account_url=account_url, credential=url_parts.query)
    container_client = blob_service_client.get_container_client(container_name)
    blobs = list(container_client.list_blobs(name_starts_with=pickup_id))

    def download_blob(blob):
        blob_name = blob.name

        # Determine destination folder based on file type
        if blob_name.lower().endswith(".pdf"):
            destination_dir = os.path.join(output_directory, f"pdf/{pickup_id}")
        elif blob_name.lower().endswith(".json"):
            destination_dir = os.path.join(output_directory, f"extractedData/{pickup_id}")
        else:
            # Skip non-pdf and non-json files
            return None

        # Maintain relative path
        relative_path = os.path.relpath(blob_name, pickup_id)
        local_file_path = os.path.join(destination_dir, relative_path)

        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download blob
        try:
            download_stream = container_client.download_blob(blob_name)
            with open(local_file_path, "wb") as download_file:
                download_file.write(download_stream.readall())
            return f"Downloaded: {blob_name} to {local_file_path}"
        except Exception as e:
            return f"Failed to download {blob_name}: {e}"

    # Run downloads in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_blob, blob) for blob in blobs]
        for future in as_completed(futures):
            result = future.result()
            if result:
                print(result)


# Function to convert PDF to images
def convert_pdf_to_images(pickup_id, pdf_path, image_output_dir, dpi=200):
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

# Define the fields structure for individual order items
ORDER_ITEM_FIELDS = [
    "Description", "HsCode", "HsCodeExport", "Quantity", "ArticleNumber",
    "GrossWeight", "NetWeight", "CountryOfOrigin", "NumberOfUnits",
    "TypeOfUnit", "PricePerPiece", "NetAmount"
]

def convert_to_order_structure(properties):
    def clean_hscode(value: str) -> str:

        value = value.strip()  # Remove leading/trailing whitespace
        if value.startswith("H.S.CODE:"):
            return value.replace("H.S.CODE:", "", 1).strip()
        return value
    

    def clean_number(value: str) -> str:
        value = value.strip()
        value = value.replace("(", "")
        value = value.replace(")", "")
        return value

    def pick_value(key, default=None):
        val = properties.get(key, default)
        if val == None or val == "null":
            val = ""
        return val
    
    def clean_field(key, val):
        if key == "HsCode":
            val = clean_hscode(val)
        if key in ["ArticleNumber", "GrossWeight", "NetWeight", "NumberOfUnits", "NetAmount", "PricePerPiece"]:
            val = clean_number(val)
        return val

    def map_order_items(items):
        order_items = []
        for item in items:
            mapped_item = {field: clean_field(field, item.get(field, "")) for field in ORDER_ITEM_FIELDS}
            order_items.append(mapped_item)
        return order_items

    raw_items = properties.get("OrderItems", [])
    order_items = map_order_items(raw_items)

    return remove_fields({
        "OrderNumber": pick_value("OrderNumber"),
        "InvoiceNumber": pick_value("InvoiceNumber"),
        "BuyerName": pick_value("BuyerName"),
        "BuyerAddress1": pick_value("BuyerAddress1"),
        "BuyerZipCode": pick_value("BuyerZipCode"),
        "BuyerCity": pick_value("BuyerCity"),
        "BuyerCountry": pick_value("BuyerCountry"),
        "ReceiverName": pick_value("ReceiverName"),
        "ReceiverAddress1": pick_value("ReceiverAddress1"),
        "ReceiverZipCode": pick_value("ReceiverZipCode"),
        "ReceiverCity": pick_value("ReceiverCity"),
        "ReceiverCountry": pick_value("ReceiverCountry"),
        "SellerName": pick_value("SellerName"),
        "NetAmount": clean_number(pick_value("NetAmount")),
        "OrderDate": pick_value("OrderDate"),
        "Currency": pick_value("Currency"),
        "TermsOfDelCode": pick_value("TermsOfDelCode"),
        "OrderItems": order_items,
        "NetWeight": pick_value("NetWeight"),
        "ActualFreight": pick_value("ActualFreight"),
        "NumberOfUnits": pick_value("NumberOfUnits"),
        "OtherAmount": pick_value("OtherAmount")
    }, FIELDS_TO_REMOVE)




# Function to create raw data from extracted JSON
raw_data = {}

def create_raw_data(pickup_id, company, json_dir, image_paths):
    json_root = os.path.join(json_dir, pickup_id)
    if not os.path.exists(json_root):
        print(f"Extracted JSON directory not found for Pickup ID {pickup_id}. Skipping...")
        return
    

    raw_data[pickup_id] = {}
    
    page_wise_data = {}
    if company == "dentalspar":
        matching_json_file = None
        for root, _, files in os.walk(json_root):
            for file in files:
                if file.startswith(f"{1}_") and file.endswith(".json"):
                    matching_json_file = os.path.join(root, file)
                    break
            if matching_json_file:
                break
        extracted_data = load_json(matching_json_file)
        for item in extracted_data.get("Properties").get("OrderItems"):
            page_number = str(item["PageNumber"])
            if page_number not in page_wise_data:
                if page_number == "1":
                    order_copy = json.loads(json.dumps(extracted_data["Properties"]))
                    order_copy["OrderItems"] = []
                else:
                    order_copy = {
                                "OrderNumber": "",
                                "InvoiceNumber": "",
                                "BuyerName": "",
                                "BuyerAddress1": "",
                                "BuyerZipCode": "",
                                "BuyerCity": "",
                                "BuyerCountry": "",
                                "ReceiverName": "",
                                "ReceiverAddress1": "",
                                "ReceiverZipCode": "",
                                "ReceiverCity": "",
                                "ReceiverCountry": "",
                                "SellerName": "",
                                "NetAmount": "",
                                "OrderDate": "",
                                "Currency": "",
                                "TermsOfDelCode": "",
                                "OrderItems": [],
                                "NetWeight": "",
                                "NumberOfUnits": "",
                                "OtherAmount": ""
                            }
                page_wise_data[page_number] = order_copy
            page_wise_data[page_number].get("OrderItems").append(item)
    for page_num, image_path in image_paths.get(pickup_id, {}).items():
        properties = {}
        if company == "dentalspar":
            print(page_num)
            properties = page_wise_data.get(page_num, {})
        else:

        # Try to find the matching extracted JSON
            matching_json_file = None
            for root, _, files in os.walk(json_root):
                for file in files:
                    if file.startswith(f"{page_num}_") and file.endswith(".json"):
                        matching_json_file = os.path.join(root, file)
                        break
                if matching_json_file:
                    break

            # Load properties if JSON is found
            if matching_json_file:
                extracted_data = load_json(matching_json_file)
                properties = extracted_data.get("Properties", {})

       
        data = convert_to_order_structure(properties)

        raw_data[pickup_id][page_num] = {
            "image_path": image_path,
            "data": data
        }



def process_single_pdf(pickup_id, pdf_path, image_output_dir):
    return pickup_id, convert_pdf_to_images(pickup_id, pdf_path, image_output_dir)

def parallel_pdf_to_images(pickup_ids, pdf_output_dir, image_output_dir, output_json_path, max_workers=8):
    tasks = []

    for pickup_id in pickup_ids:
        try:
            pdf_folder = os.path.join(pdf_output_dir, pickup_id)
            for pdf_file in os.listdir(pdf_folder):
                if pdf_file.endswith(".pdf"):
                    pdf_path = os.path.join(pdf_folder, pdf_file)
                    tasks.append((pickup_id, pdf_path))
        except Exception as e:
            print(e)

    image_paths_map = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_pdf, pickup_id, pdf_path, image_output_dir) for pickup_id, pdf_path in tasks]
        for future in as_completed(futures):
            try:
                pickup_id, image_paths = future.result()
                image_paths_map[pickup_id] = image_paths
            except Exception as e:
                print(f"Failed to process a PDF: {e}")

    # Save the image paths mapping
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(image_paths_map, f, indent=4)    

# Main Execution
if __name__ == "__main__":
    pickup_ids = []
    for company, pickup_id_list in PICKUP_MAP.items():
        pickup_ids.extend(pickup_id_list)

    
    # Step 1: Download PDF files
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for pickup_id in pickup_ids:
            executor.submit(download_blob_folder, PDF_BLOB_URL, pickup_id, "./data")
    
    # Step 2: Convert PDFs to images
    parallel_pdf_to_images(
        pickup_ids=pickup_ids,
        pdf_output_dir=PDF_OUTPUT_DIR,
        image_output_dir=IMAGE_OUTPUT_DIR,
        output_json_path="./data/image_path_map.json",
        max_workers=8
    )
    
    # Step 3: Process raw data
    with open("./data/image_path_map.json", "r", encoding="utf-8") as file:
        image_paths_map = json.load(file)
    for company, pickup_ids in PICKUP_MAP.items():
        for pickup_id in pickup_ids:
            create_raw_data(pickup_id, company, EXTRACTED_OUTPUT_DIR, image_paths_map)

    with open(RAW_DATA_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=4)
    
    
    print("Raw data creation complete.")
