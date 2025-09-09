import os
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz  # PyMuPDF

# Constants
RAW_DATA_PATH = os.path.join("classification_data", "raw_data.json")
TRAIN_DATA_PATH = os.path.join("classification_data", "train_data.json")
TEST_DATA_PATH = os.path.join("classification_data", "test_data.json")
SPLIT_RATIO = float(os.environ["SPLIT_RATIO"])

CLASSIFICATION_PROMPT_TEMPLATE = (
    "<image>\n"
    "You are given an invoice document image.\n"
    "FileType: {file_type}\n"
    "SellerName: {seller_name}\n"
    "Your job is to learn from the provided image, file type, and seller name.\n"
    "After training, you should be able to classify any given document with the correct FileType and SellerName.\n"
    "For each document, output the classified FileType and SellerName.\n"
    "Respond with a JSON object containing only 'FileType' and 'SellerName'.\n"
    "Note: The expected values for FileType are '6' (OrderItems) and '5' (OrderHead). FileType should be either '6' or '7'."
)

def load_raw_data():
    if not os.path.exists(RAW_DATA_PATH):
        return []
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

def split_data(data):
    random.shuffle(data)
    split_index = int(len(data) * SPLIT_RATIO)
    return data[:split_index], data[split_index:]

def save_data(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

# Add processing logic for raw data
def process_page(pickup_id, page_num, data):
    """Process a single page of raw data for classification."""
    try:
        # Use the actual image_path from raw data for correctness
        image_path = data.get("image_path")
        file_type = data.get("FileType", "Invoice")
        seller_name = data.get("SellerName", "Unknown Seller")
        # Prompt with empty JSON schema for output mapping
        prompt = (
            "<image>\n"
            "You are given a single-page image from an invoice-like document.\n"
            "Extract SellerName and classify FileType.\n"
            "\n"
            "FileType definitions:\n"
            "- 6: Page primarily contains item-level line items (e.g., items table with HS/SKU, country of origin, item descriptions, quantities, unit prices, amounts) and does not include a full order-level header/summary.\n"
            "- 5: Page contains only order-level header/summary (seller/buyer details, invoice number/date, addresses, totals/summary, signatures/stamps) and absolutely no items table.\n"
            "- 0: Page contains both an order-level header/summary and an items table on the same page.\n"
            "\n"
            "Rules:\n"
            "- If the page is not an invoice or is unreadable, return empty strings for both fields.\n"
            "- Consider only visible content; do not use external knowledge.\n"
            "- Do not hallucinate values; leave a field empty if uncertain.\n"
            "\n"
            "SellerName extraction hints:\n"
            "- Prefer the legal seller name near labels such as Seller, Supplier, Shipper, From, Vendor.\n"
            "\n"
            "Output:\n"
            "- Strict JSON with keys \"FileType\" and \"SellerName\" only; double quotes; no extra text.\n"
            "- FileType must be one of \"0\", \"6\", or \"5\" when determinable; otherwise \"\".\n"
            "{ \"FileType\": \"\", \"SellerName\": \"\" }"
        )
        return {
            "id": f"{pickup_id}_{page_num}",
            "image": image_path,
            "FileType": file_type,
            "SellerName": seller_name,
            "conversations": [
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": json.dumps({"FileType": file_type, "SellerName": seller_name}, indent=1)
                }
            ]
        }
    except Exception as e:
        print(f"Failed to process {pickup_id}_{page_num}: {e}")
        return None

def create_training_data(raw_data_path):
    with open(raw_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    training_data = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for pickup_id, files in raw_data.items():
            for pdf_identifier, pages in files.items():
                for page_num, data in pages.items():
                    # Unique id: pickupid_pdfidentifier_pagenum
                    page_key = f"{pdf_identifier}_{page_num}"
                    futures.append(executor.submit(process_page, pickup_id, page_key, data))
        for future in as_completed(futures):
            result = future.result()
            if result:
                training_data.append(result)

    return training_data

def ensure_directory_exists(directory):
    """Ensure that the given directory exists, create if it does not."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def convert_pdf_to_images(pdf_path, output_dir):
    """Convert a PDF to images and save them in the output directory."""
    ensure_directory_exists(output_dir)  # Ensure directory exists
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        image_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
        pix.save(image_path)

def main():
    raw_data = create_training_data(RAW_DATA_PATH)
    train_data, test_data = split_data(raw_data)
    save_data(train_data, TRAIN_DATA_PATH)
    save_data(test_data, TEST_DATA_PATH)

if __name__ == "__main__":
    main()
