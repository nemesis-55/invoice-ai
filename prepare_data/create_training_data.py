import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Constants
RAW_DATA_PATH = os.environ["RAW_DATA_OUTPUT"]
TRAIN_DATA_PATH = os.environ["TRAIN_DATA_PATH"]
TEST_DATA_PATH = os.environ["TEST_DATA_PATH"]
SPLIT_RATIO = float(os.environ["SPLIT_RATIO"])

PROMPT_TEMPLATE = (
    "<image>\n"
    "Extract key fields from the invoice image and return a JSON object in the following format.\n"
    "If a value is not present, use an empty string \"\".\n"
    "Do not change or format any values — extract them exactly as shown in the image.\n"
    "Output only the JSON object, without any additional text.\n\n"
    "{\n"
    "  \"OrderNumber\": \"<string>\",\n"
    "  \"InvoiceNumber\": \"<string>\",\n"
    "  \"BuyerName\": \"<string>\",\n"
    "  \"BuyerAddress1\": \"<string>\",\n"
    "  \"BuyerZipCode\": \"<string>\",\n"
    "  \"BuyerCity\": \"<string>\",\n"
    "  \"BuyerCountry\": \"<string>\",\n"
    "  \"ReceiverName\": \"<string>\",\n"
    "  \"ReceiverAddress1\": \"<string>\",\n"
    "  \"ReceiverZipCode\": \"<string>\",\n"
    "  \"ReceiverCity\": \"<string>\",\n"
    "  \"ReceiverCountry\": \"<string>\",\n"
    "  \"SellerName\": \"<string>\",\n"
    "  \"NetAmount\": \"<string>\",\n"
    "  \"OrderDate\": \"<YYYY-MM-DD>\",\n"
    "  \"Currency\": \"<string>\",\n"
    "  \"TermsOfDelCode\": \"<string>\",\n"
    "  \"ActualFreight\": \"<string>\",\n"
    "  \"OrderItems\": [\n"
    "    {\n"
    "      \"Description\": \"<string>\",\n"
    "      \"HsCode\": \"<string>\",\n"
    "      \"HsCodeExport\": \"<string>\",\n"
    "      \"Quantity\": \"<string>\",\n"
    "      \"ArticleNumber\": \"<string>\",\n"
    "      \"GrossWeight\": \"<string>\",\n"
    "      \"NetWeight\": \"<string>\",\n"
    "      \"CountryOfOrigin\": \"<string>\",\n"
    "      \"NumberOfUnits\": \"<string>\",\n"
    "      \"TypeOfUnit\": \"<string>\",\n"
    "      \"PricePerPiece\": \"<string>\",\n"
    "      \"NetAmount\": \"<string>\"\n"
    "    }\n"
    "  ],\n"
    "  \"NetWeight\": \"<string>\",\n"
    "  \"OtherAmount\": \"<string>\",\n"
    "  \"NumberOfUnits\": \"<string>\"\n"
    "}\n"
)

def process_page(pickup_id, page_num, data):
    """Process a single page of raw data."""
    try:
        image_path = data.get("image_path")
        properties = data.get("data", {})
        return {
            "id": f"{pickup_id}_{page_num}",
            "image": image_path,
            "conversations": [
                {"role": "user", "content": PROMPT_TEMPLATE},
                {"role": "assistant", "content": json.dumps(properties, indent=4)}
            ]
        }
    except Exception as e:
        print(f"Failed to process {pickup_id}_{page_num}: {e}")
        return None

def create_training_data(raw_data_path):
    """Read raw data and convert to training format in parallel."""
    with open(raw_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    training_data = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_page, pickup_id, page_num, data)
            for pickup_id, pages in raw_data.items()
            for page_num, data in pages.items()
        ]

        for future in as_completed(futures):
            result = future.result()
            if result:
                training_data.append(result)

    return training_data

def split_data(data, train_path, test_path, split_ratio=SPLIT_RATIO):
    """Split data into training and testing sets and save to files."""
    random.shuffle(data)
    split_idx = int(len(data) * split_ratio)
    train_data, test_data = data[:split_idx], data[split_idx:]

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=4)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4)

    print(f"Saved {len(train_data)} training samples to {train_path}")
    print(f"Saved {len(test_data)} testing samples to {test_path}")

def main():
    training_data = create_training_data(RAW_DATA_PATH)
    split_data(training_data, TRAIN_DATA_PATH, TEST_DATA_PATH)
    print("Training and test data preparation complete.")

if __name__ == "__main__":
    main()