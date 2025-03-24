from azure.storage.blob import BlobServiceClient
import os
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import fitz
from PIL import Image

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

def save_pdf_page_as_image(pickup_id, page_num, page, image_output_dir, dpi):
    """Process a single PDF page and save it as an image."""
    pix = page.get_pixmap(dpi=dpi)
    image_path = os.path.join(image_output_dir, f"{pickup_id}_{page_num + 1:03d}.png")
    Image.frombytes("RGB", [pix.width, pix.height], pix.samples).save(image_path)
    print(f"Saved: {image_path}")


def convert_pdf_to_images(pickup_id, pdf_path, image_output_dir, dpi=100):
    os.makedirs(image_output_dir, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    image_paths = {}

    zoom = dpi / 72  # 72 dpi is the default resolution
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(matrix=matrix)

        mode = "RGBA" if pix.alpha else "RGB"
        image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

        image_path = os.path.join(image_output_dir, f"{pickup_id}_{page_num + 1:03d}.png")
        image.save(image_path)
        image_paths[str(page_num + 1)] = image_path
        print(f"Saved: {image_path}")
    
    return image_paths

if __name__ == "__main__":
    pdf_connection_string = "https://saascustomsportalstorage.blob.core.windows.net/pickupfiles?sp=rli&st=2025-01-16T15:04:44Z&se=2026-01-16T23:04:44Z&sv=2022-11-02&sr=c&sig=GmbLCUpv%2F7TsLxvzWS0Y%2BEfYlcHxtxTzgz4hwHJN12c%3D"
    pickup_ids = ["116083"]
    base_output_directory = "./data/pdf"
    image_dir = "./data/image_test"

    # with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    #     for pickup_id in pickup_ids:
    #         executor.submit(download_blob_folder, pdf_connection_string, pickup_id, base_output_directory)

    for pickup_id in pickup_ids:
        pdf_folder = os.path.join(base_output_directory, pickup_id)
        pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
        with ThreadPoolExecutor(max_workers=500) as executor:
            for pdf_file in pdf_files:
                executor.submit(convert_pdf_to_images,pickup_id, pdf_file, image_dir)

    print("PDF download and conversion complete.")
