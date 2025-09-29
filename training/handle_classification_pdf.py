from azure.storage.blob import BlobServiceClient
import os
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import fitz
from PIL import Image
import openpyxl
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

BLOB_URL="https://saascustomsportalstorage.blob.core.windows.net/pickupfiles?sp=rli&st=2025-01-16T15:04:44Z&se=2026-01-16T23:04:44Z&sv=2022-11-02&sr=c&sig=GmbLCUpv%2F7TsLxvzWS0Y%2BEfYlcHxtxTzgz4hwHJN12c%3D"

PDF_OUTPUT_DIR = os.path.join("classification_data", "pdf")
IMAGE_OUTPUT_DIR = os.path.join("classification_data", "image")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 4))

def download_blob_folder(sas_url, pickup_id, output_directory, max_workers=MAX_WORKERS):
    """
    Downloads all PDF and Excel files for a given pickup_id from Azure Blob Storage
    and saves them to the specified output directory, preserving folder structure.
    Uses parallel threads for faster downloads.
    """
    if not sas_url:
        raise ValueError("SAS URL is empty. Please set the BLOB_URL environment variable.")
    if isinstance(sas_url, bytes):
        sas_url = sas_url.decode()
    url_parts = urlparse(sas_url)
    path_parts = url_parts.path.split('/')
    if len(path_parts) < 2 or not path_parts[1]:
        raise ValueError("Invalid SAS URL format. Could not extract container name.")
    account_url = f"https://{url_parts.netloc}"
    container_name = path_parts[1]
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

def convert_pdf_to_images(pdf_path, output_dir, pickup_id):
    """
    Converts a PDF file to images, one per page, and saves them in the output directory.
    Returns a dictionary mapping page numbers to image file paths for the given pickup_id.
    """
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

# Excel to image conversion (A1:J50)
def convert_excel_to_image(excel_path, output_dir, pickup_id):
    """
    Converts a portion of an Excel sheet (A1:J50) to an image using matplotlib
    and saves it in the output directory. Returns the identifier and image path.
    """
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


def ensure_directory_exists(directory):
    """
    Utility function to ensure a directory exists. Creates it if it doesn't exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
def process_pickup(pickup_id):
    """
    Orchestrates the download and conversion process for a single pickup_id:
    - Downloads all relevant blobs (PDF/XLSX)
    - Converts PDFs to images
    - Converts Excels to images
    Returns a mapping of identifiers to image paths.
    """
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

def parallel_pdf_to_images_classification(pickup_ids, pdf_output_dir, image_output_dir, output_json_path, max_workers=MAX_WORKERS):
    """
    Converts PDFs to images in parallel for a list of pickup_ids.
    Skips pickups with multiple PDFs. Saves image path map to output_json_path.
    Returns image path map and set of skipped pickup_ids.
    """
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
        for future in ThreadPoolExecutor.as_completed(futures):
            try:
                pickup_id, image_paths = future.result()
                image_paths_map[pickup_id] = image_paths
            except Exception as e:
                print(f"Failed to process a PDF: {e}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(image_paths_map, f)

    return image_paths_map, skipped_pickup_ids

def process_pickup_images_only(pickup_id, pdf_path, image_output_dir):
    """
    Converts a single PDF to images for classification, returns pickup_id and image path map.
    """
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

if __name__ == "__main__":
    # List of pickup IDs to process
    pickup_ids = [
        "188274","187757","185758","189255","189592"
    ]
    all_image_paths_map = {}
    for pickup_id in pickup_ids:
        result = process_pickup(pickup_id)
        if result:
            all_image_paths_map[pickup_id] = result

    # Save image path map
    image_path_map_path = os.path.join("classification_data", "image_path_map.json")
    with open(image_path_map_path, "w", encoding="utf-8") as f:
        json.dump(all_image_paths_map, f, indent=2)
    print(f"Image path map saved: {image_path_map_path}")
