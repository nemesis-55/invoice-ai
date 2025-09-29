# Import Azure Blob Storage SDK for downloading files from cloud
from azure.storage.blob import BlobServiceClient
# Standard library imports for file and path handling
import os
# For parsing SAS URLs
from urllib.parse import urlparse
# For parallel execution of downloads and conversions
from concurrent.futures import ThreadPoolExecutor
# PyMuPDF for PDF to image conversion
import fitz
# PIL for image manipulation and saving
from PIL import Image

def download_blob_folder(sas_url, folder_path, output_directory):
    """
    Downloads all blobs (files) from a specified folder in an Azure Blob Storage container
    to a local output directory, preserving the folder structure.
    """

    # Parse the SAS URL to extract account and container info
    url_parts = urlparse(sas_url)
    account_url = f"https://{url_parts.netloc}"
    container_name = url_parts.path.split('/')[1]


    # Create a BlobServiceClient and get the container client
    blob_service_client = BlobServiceClient(account_url=account_url, credential=url_parts.query)
    container_client = blob_service_client.get_container_client(container_name)
    # List all blobs in the specified folder
    blobs = container_client.list_blobs(name_starts_with=folder_path)

    # Download each blob and save it locally
    for blob in blobs:
        blob_name = blob.name
        local_file_path = os.path.join(output_directory, blob_name)
        # Ensure the local directory exists
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download the blob and write its contents to a local file
        with open(local_file_path, "wb") as download_file:
            download_stream = container_client.download_blob(blob_name)
            download_file.write(download_stream.readall())

        # Print confirmation of download
        print(f"Downloaded: {blob_name} to {local_file_path}")

def convert_pdf_to_images(pickup_id, pdf_path, image_output_dir, dpi=600):
    """
    Converts a PDF file to images, one per page, and saves them in the specified output directory.
    Returns a dictionary mapping page numbers to image file paths.
    """
    # Ensure the output directory exists
    os.makedirs(image_output_dir, exist_ok=True)
    # Open the PDF document
    pdf_document = fitz.open(pdf_path)
    image_paths = {}
    pdf_filename = os.path.basename(pdf_path)
    pdf_identifier = f"pdf_{pdf_filename}"
    # Iterate through each page and convert to image
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        mode = "RGBA" if pix.alpha else "RGB"
        image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        image_filename = f"{pickup_id}_{pdf_identifier}_{page_num + 1:03d}.png"
        image_path = os.path.join(image_output_dir, image_filename)
        image.save(image_path)
        image_paths[str(page_num + 1)] = image_path
        print(f"Saved: {image_path}")
    # Return mapping of page numbers to image paths
    return image_paths

if __name__ == "__main__":
    # Example SAS URL for Azure Blob Storage container
    pdf_connection_string = "https://saascustomsportalstorage.blob.core.windows.net/pickupfiles?sp=rli&st=2025-01-16T15:04:44Z&se=2026-01-16T23:04:44Z&sv=2022-11-02&sr=c&sig=GmbLCUpv%2F7TsLxvzWS0Y%2BEfYlcHxtxTzgz4hwHJN12c%3D"
    # List of pickup IDs to process
    pickup_ids = [
        "160573"]
    # Output directories for PDFs and images
    base_output_directory = "./data/pdf"
    image_dir = "./data/image"

    # Download all PDFs for each pickup_id in parallel
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for pickup_id in pickup_ids:
            executor.submit(download_blob_folder, pdf_connection_string, pickup_id, base_output_directory)

    # Convert all downloaded PDFs to images in parallel
    for pickup_id in pickup_ids:
        pdf_folder = os.path.join(base_output_directory, pickup_id)
        pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
        with ThreadPoolExecutor(max_workers=500) as executor:
            for pdf_file in pdf_files:
                executor.submit(convert_pdf_to_images, pickup_id, pdf_file, image_dir)

    print("PDF download and conversion complete.")
