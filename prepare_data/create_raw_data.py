import os
import json
import sys
from urllib.parse import urlparse
from azure.storage.blob import BlobServiceClient
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import fitz  # PyMuPDF
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper.order_csv_utils import embed_order_items_csv_in_json

# Constants
BLOB_URL = os.environ["BLOB_URL"]
PICKUP_IDS = os.environ["PICKUP_IDS"].split(",")
FIELDS_TO_REMOVE = os.environ["FIELDS_TO_REMOVE"].split(",")
ORDER_ITEM_FIELDS = os.environ["ORDER_ITEM_FIELDS"].split(",")
PDF_OUTPUT_DIR = os.environ["PDF_OUTPUT_DIR"]
IMAGE_OUTPUT_DIR = os.environ["IMAGE_OUTPUT_DIR"]
EXTRACTED_OUTPUT_DIR = os.environ["EXTRACTED_OUTPUT_DIR"]
RAW_DATA_OUTPUT = os.environ["RAW_DATA_OUTPUT"]
MAX_WORKERS = int(os.environ["MAX_WORKERS"])


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

def download_blob_folder(sas_url, pickup_id, output_directory, max_workers=MAX_WORKERS):
    url_parts = urlparse(sas_url)
    account_url = f"https://{url_parts.netloc}"
    container_name = url_parts.path.split('/')[1]
    blob_service_client = BlobServiceClient(account_url=account_url, credential=url_parts.query)
    container_client = blob_service_client.get_container_client(container_name)
    blobs = list(container_client.list_blobs(name_starts_with=pickup_id))

    def download_blob(blob):
        blob_name = blob.name
        if blob_name.lower().endswith(".pdf"):
            destination_dir = os.path.join(output_directory, f"pdf/{pickup_id}")
        elif blob_name.lower().endswith(".json"):
            destination_dir = os.path.join(output_directory, f"extractedData/{pickup_id}")
        else:
            return None

        relative_path = os.path.relpath(blob_name, pickup_id)
        local_file_path = os.path.join(destination_dir, relative_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        try:
            download_stream = container_client.download_blob(blob_name)
            with open(local_file_path, "wb") as download_file:
                download_file.write(download_stream.readall())
            return f"Downloaded: {blob_name} to {local_file_path}"
        except Exception as e:
            return f"Failed to download {blob_name}: {e}"

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_blob, blob) for blob in blobs]
        for future in as_completed(futures):
            result = future.result()
            if result:
                print(result)

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
        image_paths[str(page_num + 1)] = image_path
        print(f"Saved: {image_path}")
    return image_paths

def remove_fields(obj, fields):
    if isinstance(obj, dict):
        return {k: remove_fields(v, fields) for k, v in obj.items() if k not in fields}
    elif isinstance(obj, list):
        return [remove_fields(i, fields) for i in obj]
    return obj

def convert_to_order_structure(properties):
    def clean_hscode(value):
        value = value.strip()
        return value.replace("H.S.CODE:", "", 1).strip() if value.startswith("H.S.CODE:") else value

    def clean_number(value):
        return value.strip().replace("(", "").replace(")", "")

    def pick_value(key, default=""):
        val = properties.get(key, default)
        return "" if val in [None, "null"] else val

    def clean_field(key, val):
        val="" if val in [None, "null"] else val
        if key == "HsCode":
            val = clean_hscode(val)
        if key in ["ArticleNumber", "GrossWeight", "NetWeight", "NumberOfUnits", "NetAmount", "PricePerPiece"]:
            val = clean_number(val)
        return val

    def map_order_items(items):
        return [{field: clean_field(field, item.get(field, "")) for field in ORDER_ITEM_FIELDS} for item in items]

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
        "OrderItems": map_order_items(properties.get("OrderItems", [])),
        "NetWeight": pick_value("NetWeight"),
        "GrossWeight": pick_value("GrossWeight"),
        "ActualFreight": pick_value("ActualFreight"),
        "NumberOfUnits": pick_value("NumberOfUnits"),
        "OtherAmount": pick_value("OtherAmount")
    }, FIELDS_TO_REMOVE)

raw_data = {}

def create_raw_data(pickup_id, json_dir, image_paths):
    json_root = os.path.join(json_dir, pickup_id)
    if not os.path.exists(json_root):
        print(f"Extracted JSON directory not found for Pickup ID {pickup_id}. Skipping...")
        return
    raw_data[pickup_id] = {}
    for page_num, image_path in image_paths.get(pickup_id, {}).items():
        properties = {}
        matching_json_file = next(
            (os.path.join(root, file)
             for root, _, files in os.walk(json_root)
             for file in files if (file.startswith(f"{page_num}_") or file.startswith(f"xyz {page_num}_")) and file.endswith(".json")),
            None
        )
        if matching_json_file:
            extracted_data = load_json(matching_json_file)
            properties = extracted_data.get("Properties", extracted_data.get("extracted_data", {}).get("Properties", {}))
        
        raw_data[pickup_id][page_num] = {
            "image_path": image_path,
            "data": embed_order_items_csv_in_json(convert_to_order_structure(properties)),
        }

def process_single_pdf(pickup_id, pdf_path, image_output_dir):
    return pickup_id, convert_pdf_to_images(pickup_id, pdf_path, image_output_dir)

def parallel_pdf_to_images(pickup_ids, pdf_output_dir, image_output_dir, output_json_path, max_workers=MAX_WORKERS):
    tasks = []
    skipped_pickup_ids = set()

    for pickup_id in pickup_ids:
        pdf_folder = os.path.join(pdf_output_dir, pickup_id)
        if not os.path.exists(pdf_folder):
            continue

        pdf_files = [file for file in os.listdir(pdf_folder) if file.endswith(".pdf")]

        if len(pdf_files) > 1:
            print(f"Warning: Multiple PDFs found for Pickup ID {pickup_id}. Skipping...")
            skipped_pickup_ids.add(pickup_id)
            continue

        if pdf_files:
            tasks.append((pickup_id, os.path.join(pdf_folder, pdf_files[0])))

    image_paths_map = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_pdf, pid, path, image_output_dir) for pid, path in tasks]
        for future in as_completed(futures):
            try:
                pickup_id, image_paths = future.result()
                image_paths_map[pickup_id] = image_paths
            except Exception as e:
                print(f"Failed to process a PDF: {e}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(image_paths_map, f)

    return image_paths_map, skipped_pickup_ids

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for pickup_id in PICKUP_IDS:
            executor.submit(download_blob_folder, BLOB_URL, pickup_id, "./data")

    (image_paths_map, skipped_pickup_ids) = parallel_pdf_to_images(
        pickup_ids=PICKUP_IDS,
        pdf_output_dir=PDF_OUTPUT_DIR,
        image_output_dir=IMAGE_OUTPUT_DIR,
        output_json_path="./data/image_path_map.json",
        max_workers=8
    )

    for pickup_id in PICKUP_IDS:
        if pickup_id in skipped_pickup_ids:
            print(f"Skipping raw data creation for Pickup ID {pickup_id} due to multiple PDFs.")
            continue
        create_raw_data(pickup_id, EXTRACTED_OUTPUT_DIR, image_paths_map)

    with open(RAW_DATA_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(raw_data, f)

    print("Raw data creation complete.")
