import json
import random
from PIL import Image
import pytesseract
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_text_from_image(image):
    """Extract text from an image derived from the PDF."""
    try:
        text = pytesseract.image_to_string(image)
        print(f"Extracted text length: {len(text)} characters.")
        return text
    except Exception as e:
        print(f"Error during text extraction: {e}")
        raise RuntimeError(f"Error during text extraction: {e}")

def generate_prompt(image_path):
    """Create the detailed prompt for the model."""
    try:
        image = Image.open(image_path)
        ocr_data = extract_text_from_image(image)
        question = (
            "<image> \n Extract key details from the given OCR-extracted invoice text and image to return a valid JSON object.\n\n"
            f"### OCR Data:\n{ocr_data}\n\n"
            "### Instructions:\n"
            "1. Extract the required fields as per the JSON structure.\n"
            "3. If a field is missing, set its value to \"\".\n"
            "### JSON Output:\n"
            "{\n"
            "    \"OrderNumber\": \"<string>\",\n"
            "    \"InvoiceNumber\": \"<string>\",\n"
            "    \"BuyerName\": \"<string>\",\n"
            "    \"BuyerAddress1\": \"<string>\",\n"
            "    \"BuyerZipCode\": \"<string>\",\n"
            "    \"BuyerCity\": \"<string>\",\n"
            "    \"BuyerCountry\": \"<string>\",\n"
            "    \"ReceiverName\": \"<string>\",\n"
            "    \"ReceiverAddress1\": \"<string>\",\n"
            "    \"ReceiverZipCode\": \"<string>\",\n"
            "    \"ReceiverCity\": \"<string>\",\n"
            "    \"ReceiverCountry\": \"<string>\",\n"
            "    \"SellerName\": \"<string>\",\n"
            "    \"NetAmount\": \"<string>\",\n"
            "    \"OrderDate\": \"<YYYY-MM-DD>\",\n"
            "    \"Currency\": \"<string>\",\n"
            "    \"TermsOfDelCode\": \"<string>\",\n"
            "    \"ActualFreight\": \"<string>\",\n"
            "    \"OrderItems\": [\n"
            "        {\n"
            "            \"Description\": \"<string>\",\n"
            "            \"HsCode\": \"<string>\",\n"
            "            \"HsCodeExport\": \"<string>\",\n"
            "            \"Quantity\": \"<string>\",\n"
            "            \"ArticleNumber\": \"<string>\",\n"
            "            \"GrossWeight\": \"<string>\",\n"
            "            \"NetWeight\": \"<string>\",\n"
            "            \"CountryOfOrigin\": \"<string>\",\n"
            "            \"NumberOfUnits\": \"<string>\",\n"
            "            \"TypeOfUnit\": \"<string>\",\n"
            "            \"PricePerPiece\": \"<string>\",\n"
            "            \"NetAmount\": \"<string>\"\n"
            "        }\n"
            "    ],\n"
            "    \"NetWeight\": \"<string>\",\n"
            "    \"NumberOfUnits\": \"<string>\"\n"
            "}\n\n"
            "### Note: Output value must not contain any double quote (\")  \n"
            "Ensure the JSON structure is returned exactly as shown above, with appropriate values extracted using OCR data and image."
        )
        return question
    except Exception as e:
        print(f"Error generating prompt: {e}")
        raise

def process_page(pickup_id, page_num, data):
    """Process each page to generate training data."""
    try:
        print(f"Processing Pickup ID {pickup_id}, Page {page_num}...")
        image_path = data.get("image_path")
        properties = data.get("data", {})
        question = generate_prompt(image_path)
        answer = json.dumps(properties, indent=4)
        training_entry = {
            "id": f"{pickup_id}_{page_num}",
            "image": image_path,
            "conversations": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        }
        print(f"Processed page {page_num} for Pickup ID {pickup_id}.")
        return training_entry
    except Exception as e:
        print(f"Error processing page {page_num} for Pickup ID {pickup_id}: {e}")
        return None

def create_training_data(raw_data_path):
    training_data = []
    print(f"Loading raw data from {raw_data_path}...")
    with open(raw_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Create a ThreadPoolExecutor for parallel processing of pages
    print("Starting parallel processing of pages...")
    with ThreadPoolExecutor() as executor:
        futures = []
        
        # Submit tasks for each page
        for pickup_id, page_data in raw_data.items():
            for page_num, data in page_data.items():
                futures.append(executor.submit(process_page, pickup_id, page_num, data))
        
        # Collect results from completed tasks
        for future in as_completed(futures):
            result = future.result()
            if result:
                training_data.append(result)

    print("Training data generation complete.")
    return training_data

def split_data(training_data, train_output, test_output, split_ratio=0.25):
    """Shuffle and split data into train and test sets."""
    print("Shuffling and splitting the data...")
    random.shuffle(training_data)
    
    split_index = int(len(training_data) * split_ratio)
    train_data = training_data[:split_index]
    test_data = training_data[split_index:]

    print(f"Saving {len(train_data)} training samples to {train_output}...")
    with open(train_output, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=4)

    print(f"Saving {len(test_data)} testing samples to {test_output}...")
    with open(test_output, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4)

    print(f"Train-test split complete. Train: {len(train_data)}, Test: {len(test_data)}")

if __name__ == "__main__":
    raw_data_output = "./data/raw_data.json"
    train_data_output = "./data/train_data.json"
    test_data_output = "./data/test_data.json"

    # Step 1: Create Training Data
    training_data = create_training_data(raw_data_output)

    # Step 2: Split into Train & Test
    split_data(training_data, train_data_output, test_data_output)

    print("Training and test data preparation complete.")
