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

def convert_pdf_to_images(pickup_id, pdf_path, image_output_dir, dpi=600):
    os.makedirs(image_output_dir, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    image_paths = {}

    # --- Use the same identifier logic as in create_raw_data.py ---
    def generate_pdf_identifier(pdf_path, pickup_id):
        pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        if pdf_filename.startswith(pickup_id):
            identifier = pdf_filename[len(pickup_id):].lstrip('_-')
        else:
            identifier = pdf_filename
        identifier = identifier.replace(' ', '_').replace('-', '_')
        identifier = ''.join(c for c in identifier if c.isalnum() or c == '_')
        identifier = identifier.strip('_')
        if not identifier:
            identifier = pdf_filename.replace(' ', '_').replace('-', '_')
            identifier = ''.join(c for c in identifier if c.isalnum() or c == '_').strip('_')
        return f"pdf_{identifier}"

    pdf_identifier = generate_pdf_identifier(pdf_path, pickup_id)

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        mode = "RGBA" if pix.alpha else "RGB"
        image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        # Use the same image naming pattern as in create_raw_data.py
        image_path = os.path.join(image_output_dir, f"{pickup_id}_{pdf_identifier}_{page_num + 1:03d}.png")
        image.save(image_path)
        image_paths[str(page_num + 1)] = image_path
        print(f"Saved: {image_path}")
    return image_paths

if __name__ == "__main__":
    pdf_connection_string = "https://saascustomsportalstorage.blob.core.windows.net/pickupfiles?sp=rli&st=2025-01-16T15:04:44Z&se=2026-01-16T23:04:44Z&sv=2022-11-02&sr=c&sig=GmbLCUpv%2F7TsLxvzWS0Y%2BEfYlcHxtxTzgz4hwHJN12c%3D"
    pickup_ids = ["160573","160228","159777","159776","159775","159749","159308","159288","159287","159283","158291","158269","161099","160729","160198","159759","159391","158975","158361","158029","157699","157365","157269","157189","160633","159675","159674","159565","159230","158943","158942","158156","157697","157696","157695","156047","160175","159676","159611","159179","158060","157480","157284","157279","157260","156836","156803","156232","155727","155694","160875","159950","158192","157779","156489","156294","154920","161103","160728","160226","159430","159024","158386","158040","155935","155609","155017","155016","155013","155009","154916","160876","159951","156298","155231","154990","153092","152381","151357","150634","150549","160043","158246","157223","157061","156450"
]
    base_output_directory = "./data/pdf"
    image_dir = "./data/image"

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for pickup_id in pickup_ids:
            executor.submit(download_blob_folder, pdf_connection_string, pickup_id, base_output_directory)

    for pickup_id in pickup_ids:
        pdf_folder = os.path.join(base_output_directory, pickup_id)
        pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
        with ThreadPoolExecutor(max_workers=500) as executor:
            for pdf_file in pdf_files:
                executor.submit(convert_pdf_to_images,pickup_id, pdf_file, image_dir)

    print("PDF download and conversion complete.")
