import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz  # PyMuPDF
from PIL import Image
from urllib.parse import urlparse, quote
from azure.storage.blob import BlobServiceClient
import sys
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re

# Constants
PICKUP_IDS = os.environ["PICKUP_IDS"].split(",")
PDF_OUTPUT_DIR = os.path.join("classification_data", "pdf")
IMAGE_OUTPUT_DIR = os.path.join("classification_data", "image")
MAX_WORKERS = int(os.environ["MAX_WORKERS"])

def ensure_directory_exists(directory):
    """Ensure the directory exists, create it if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def convert_pdf_to_images(pdf_path, output_dir, pickup_id):
    ensure_directory_exists(output_dir)
    pdf_filename = os.path.basename(pdf_path)
    pdf_identifier = f"pdf_{pdf_filename}"
    doc = fitz.open(pdf_path)
    image_paths = {}
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        image_filename = f"{pickup_id}_{pdf_identifier}_{page_num + 1:03d}.png"
        image_path = os.path.join(output_dir, image_filename)
        pix.save(image_path)
        image_paths[str(page_num + 1)] = image_path
    return pdf_identifier, image_paths

# Add blob storage functionality
def download_blob_folder(sas_url, pickup_id, output_directory, max_workers=MAX_WORKERS):
    url_parts = urlparse(sas_url)
    account_url = f"https://{url_parts.netloc}"
    container_name = url_parts.path.split('/')[1]
    blob_service_client = BlobServiceClient(account_url=account_url, credential=url_parts.query)
    container_client = blob_service_client.get_container_client(container_name)
    blobs = list(container_client.list_blobs(name_starts_with=pickup_id))

    def download_blob(blob):
        blob_name = blob.name
        if blob_name.lower().endswith(".pdf") or blob_name.lower().endswith(".xlsx"):
            destination_dir = os.path.join(output_directory, pickup_id)
            local_file_path = os.path.join(destination_dir, os.path.basename(blob_name))
            os.makedirs(destination_dir, exist_ok=True)
            with open(local_file_path, "wb") as file:
                file.write(blob_service_client.get_blob_client(container=container_name, blob=blob_name).download_blob().readall())

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_blob, blob) for blob in blobs if blob.name.lower().endswith(".pdf") or blob.name.lower().endswith(".xlsx")]
        for future in futures:
            future.result()

# Update process_pickup to include blob download
BLOB_URL = os.environ.get("BLOB_URL")

def process_pickup(pickup_id):
    download_blob_folder(BLOB_URL, pickup_id, PDF_OUTPUT_DIR)
    pdf_dir = os.path.join(PDF_OUTPUT_DIR, str(pickup_id).strip())
    ensure_directory_exists(pdf_dir)
    ensure_directory_exists(IMAGE_OUTPUT_DIR)
    if not os.path.exists(pdf_dir):
        print(f"PDF/Excel directory does not exist for pickup_id {pickup_id}: {pdf_dir}")
        return {}
    files = os.listdir(pdf_dir)
    pdf_files = [os.path.join(pdf_dir, file) for file in files if file.lower().endswith(".pdf")]
    excel_files = [os.path.join(pdf_dir, file) for file in files if file.lower().endswith(".xlsx")]
    if not pdf_files and not excel_files:
        print(f"No PDF or Excel files found for pickup_id {pickup_id} in {pdf_dir}")
        return {}
    image_paths_map = {}
    # PDF logic
    for pdf_path in pdf_files:
        pdf_identifier, image_paths = convert_pdf_to_images(pdf_path, IMAGE_OUTPUT_DIR, pickup_id)
        image_paths_map[pdf_identifier] = image_paths
    # Excel logic
    for excel_path in excel_files:
        excel_identifier, image_path = convert_excel_to_image(excel_path, IMAGE_OUTPUT_DIR, pickup_id)
        image_paths_map[excel_identifier] = {"1": image_path}  # page_num "1" for Excel
    return image_paths_map
# Add after imports
import openpyxl
import numpy as np

# Excel to image conversion (A1:J50)
def convert_excel_to_image(excel_path, output_dir, pickup_id):
    ensure_directory_exists(output_dir)
    excel_filename = os.path.basename(excel_path)
    excel_identifier = f"excel_{excel_filename}"
    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active
    # Extract A1:O30 (columns 1-15, rows 1-30)
    data = []
    for row in ws.iter_rows(min_row=1, max_row=30, min_col=1, max_col=10, values_only=True):
        data.append([str(cell) if cell is not None else "" for cell in row])
    # Create image using matplotlib
    fig, ax = plt.subplots(figsize=(20, 10))  # Wider and taller for clarity
    ax.axis('off')
    table = ax.table(cellText=data, loc='center', cellLoc='center', colLabels=[f"{chr(65+i)}" for i in range(15)])
    table.auto_set_font_size(False)
    table.set_fontsize(14)  # Larger font for visibility
    table.scale(2.5, 2.5)   # Increase cell size to prevent overlap
    # Adjust column widths and row heights for clarity
    for (row, col), cell in table.get_celld().items():
        # cell.set_height(0.08)
        cell.set_width(0.25)
        cell.set_text_props(va='center', ha='center')
    image_filename = f"{pickup_id}_{excel_identifier}_001.png"
    image_path = os.path.join(output_dir, image_filename)
    plt.savefig(image_path, bbox_inches='tight')
    plt.close(fig)
    return excel_identifier, f"classification_data\\image\\{pickup_id}_{excel_identifier}_001.png"

def save_classification_raw_data():
    raw_data = {}
    for pickup_id in PICKUP_IDS:
        image_dir = os.path.join(IMAGE_OUTPUT_DIR, pickup_id)
        if not os.path.exists(image_dir):
            continue
        image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
        for image_file in image_files:
            # Extract page number from filename: page_{page_num}.png
            page_num = image_file.split('_')[1].split('.')[0] if '_' in image_file else image_file.split('.')[0]
            image_path = f"classification_data\\image\\{pickup_id}_{int(page_num):03d}.png"
            if pickup_id not in raw_data:
                raw_data[pickup_id] = {}
            raw_data[pickup_id][str(int(page_num))] = {
                "image_path": image_path
            }
    raw_data_path = os.path.join("classification_data", "raw_data.json")
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    with open(raw_data_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)
    print(f"Saved classification raw data to {raw_data_path}")

def parallel_pdf_to_images_classification(pickup_ids, pdf_output_dir, image_output_dir, output_json_path, max_workers=MAX_WORKERS):
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
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_pickup_images_only, pid, path, image_output_dir) for pid, path in tasks]
        for future in as_completed(futures):
            try:
                pickup_id, image_paths = future.result()
                image_paths_map[pickup_id] = image_paths
            except Exception as e:
                print(f"Failed to process a PDF: {e}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(image_paths_map, f)

    return image_paths_map, skipped_pickup_ids

def process_pickup_images_only(pickup_id, pdf_path, image_output_dir):
    image_paths = {}
    ensure_directory_exists(image_output_dir)
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        image_filename = f"{pickup_id}_{page_num + 1:03d}.png"
        image_path = os.path.join(image_output_dir, image_filename)
        pix.save(image_path)
        image_paths[str(page_num + 1)] = f"classification_data\\image\\{pickup_id}_{page_num + 1:03d}.png"
    return pickup_id, image_paths

def get_pickup_file_type(pickup_id: int, file_name: str) -> tuple[str, int]:
    """
    Calls the TrainingDataController GetPickupFileType endpoint and returns (seller_name, file_type).
    
    Args:
        base_url (str): The base URL of the API (e.g., "http://localhost:5000")
        pickup_id (int): The pickup ID to query
        file_name (str): The file name to query

    Returns:
        tuple: (seller_name, file_type) if found, else (None, None)
    """
    base_url = "http://localhost:5000"  # Change to your actual base URL
    url = f"{base_url}/api/trainingdata/pickup/{pickup_id}/getFileType"
    params = {"fileName": file_name}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            seller_name = data.get("SellerName")
            file_type = data.get("FileType")
            return seller_name, file_type
        else:
            # Not found or error
            return None, None
    except Exception as ex:
        print(f"Error calling API: {ex}")
        return None, None

def main():
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_pickup, pickup_id) for pickup_id in PICKUP_IDS]
        for future in futures:
            future.result()

if __name__ == "__main__":
    # Collect image paths for all pickups
    all_image_paths_map = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {pickup_id: executor.submit(process_pickup, pickup_id) for pickup_id in PICKUP_IDS}
        for pickup_id, future in futures.items():
            result = future.result()
            if result:
                all_image_paths_map[pickup_id] = result

    # Save image_path_map.json in classification_data folder (multi-PDF aware)
    image_path_map_path = os.path.join("classification_data", "image_path_map.json")
    with open(image_path_map_path, "w", encoding="utf-8") as f:
        json.dump(all_image_paths_map, f, indent=2)
    print(f"Image path map saved: {image_path_map_path}")

    # Save raw_data.json in the required format (multiple PDFs/Excels per pickup)
    raw_data = {}
    for pickup_id in PICKUP_IDS:
        if pickup_id not in all_image_paths_map:
            continue
        raw_data[pickup_id] = {}
        for identifier, pages in all_image_paths_map[pickup_id].items():
            # Get the file type and seller name for each identifier
            file_name_regex = r'^(?:pdf_|excel_)(.+)$'

            # Extract actual file name from identifier
            match = re.match(file_name_regex, identifier)
            if match:
                file_name = match.group(1)
            else:
                file_name = identifier

            seller_name, file_type = get_pickup_file_type(pickup_id, file_name)
            raw_data[pickup_id][identifier] = {}
            for page_num, image_path in pages.items():
                raw_data[pickup_id][identifier][page_num] = {
                    "image_path": image_path,
                    "SellerName": seller_name,
                    "FileType": file_type
                }
    raw_data_path = os.path.join("classification_data", "raw_data.json")
    with open(raw_data_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)
    print(f"Raw data creation complete: {raw_data_path}")
