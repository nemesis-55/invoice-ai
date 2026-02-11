import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Constants
RAW_DATA_PATH = os.environ["RAW_DATA_OUTPUT"]
TRAIN_DATA_PATH = os.environ["TRAIN_DATA_PATH"]
# Note: TEST_DATA_PATH and SPLIT_RATIO are no longer used
# LLamaFactory handles train/validation splitting internally via val_size parameter

PROMPT_TEMPLATE = (
            "<image>\n"
            "Extract the following fields from the invoice image and return a JSON object:\n"
            "- OrderNumber\n"
            "- InvoiceNumber\n"
            "- BuyerName\n"
            "- BuyerAddress1\n"
            "- BuyerZipCode\n"
            "- BuyerCity\n"
            "- BuyerCountry\n"
            "- ReceiverName\n"
            "- ReceiverAddress1\n"
            "- ReceiverZipCode\n"
            "- ReceiverCity\n"
            "- ReceiverCountry\n"
            "- SellerName\n"
            "- NetAmount\n"
            "- OrderDate (YYYY-MM-DD)\n"
            "- Currency\n"
            "- TermsOfDelCode\n"
            "- ActualFreight\n"
            "- OrderItemsList: a list of lists. Each inner list represents one item and follows the column order:\n"
            "  ['Description', 'HsCode', 'HsCodeExport', 'Quantity', 'ArticleNumber', 'GrossWeight', "
            "'NetWeight', 'CountryOfOrigin', 'NumberOfUnits', 'TypeOfUnit', 'PricePerPiece','NetAmount','Discount','DiscountPercentage']\n"            "- NetWeight\n"
            "- OtherAmount\n"
            "- NumberOfUnits\n"
            "Use exact text from the image. If a value is missing, set it to an empty string \"\".\n"
            "Respond with only the JSON object."
        )
        
def process_page(pickup_id, page_key, data):
    """Process a single page of raw data."""
    try:
        image_path = data.get("image_path")
        properties = data.get("data", {})

        if image_path:
            image_path = image_path.replace("\\", "/")
        
        # Handle both legacy and new format for unique IDs
        # Legacy: pickup_id_page_num (e.g., "185486_1")
        # New multi-PDF: pickup_id_pdf_identifier_page_num (e.g., "185486_invoice_doc_1")
        unique_id = f"{pickup_id}_{page_key}"
        
        return {
            "id": unique_id,
            "images": [image_path] if image_path else [],
            "conversations": [
                {"from": "human", "value": PROMPT_TEMPLATE},
                {"from": "gpt", "value": json.dumps(properties, indent=1)}
            ]
        }
    except Exception as e:
        print(f"Failed to process {pickup_id}_{page_key}: {e}")
        return None

def create_training_data(raw_data_path):
    """Read raw data and convert to training format in parallel."""
    with open(raw_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    training_data = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_page, pickup_id, page_key, data)
            for pickup_id, pages in raw_data.items()
            for page_key, data in pages.items()
        ]

        for future in as_completed(futures):
            result = future.result()
            if result:
                training_data.append(result)

    print(f"Created {len(training_data)} training samples from raw data")
    return training_data

def save_training_data(data, train_path):
    """Save all data to training file. LLamaFactory will handle train/validation split."""
    # Shuffle data for randomness
    random.shuffle(data)
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    
    print(f"Saved {len(data)} training samples to {train_path}")
    print(f"Note: LLamaFactory will automatically split this into train/validation based on val_size parameter")

def main():
    training_data = create_training_data(RAW_DATA_PATH)
    save_training_data(training_data, TRAIN_DATA_PATH)
    print("Training data preparation complete.")

if __name__ == "__main__":
    main()