import json
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
            "You are an AI model specialized in data extraction from invoices. "
            "Below, you are provided with OCR-extracted text from an invoice. "
            "Your task is to analyze the OCR data and extract key details to structure them as a JSON object.\n\n"
            f"### OCR Data:\n{ocr_data}\n\n"
            "### Instructions:\n"
            "1. Extract all the required fields as specified in the JSON structure below.\n"
            "2. Ensure the output is a syntactically valid JSON string.\n"
            "3. If a field is missing or unavailable in the OCR text, set its value to an empty string \"\".\n"
            "4. Maintain the exact formatting of numeric values and dates as found in the input.\n"
            "5. Do not include additional explanations or comments in your output.\n\n"
            "### JSON Structure:\n"
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
            "    \"OrderItems\": [\n"
            "        {\n"
            "            \"ArticleNumber\": \"<string>\",\n"
            "            \"Description\": \"<string>\",\n"
            "            \"HsCode\": \"<string>\",\n"
            "            \"CountryOfOrigin\": \"<string>\",\n"
            "            \"Quantity\": \"<string>\",\n"
            "            \"NetWeight\": \"<string>\",\n"
            "            \"NetAmount\": \"<string>\",\n"
            "            \"PricePerPiece\": \"<string>\",\n"
            "            \"EclEuNO\": \"<string>\"\n"
            "        }\n"
            "    ],\n"
            "    \"NetWeight\": \"<string>\",\n"
            "    \"NumberOfUnits\": \"<string>\"\n"
            "}\n\n"
            "### Note:\n"
            "Ensure the JSON structure is returned exactly as shown above, with appropriate values extracted from the OCR data."
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
            "id": f"{pickup_id}{page_num}",
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


def create_training_data(raw_data_path, output_file):
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

    # Save the training data to the output file
    print(f"Saving training data to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=4)
    print(f"Training data saved to {output_file}")
    return training_data


if __name__ == "__main__":
    raw_data_output = ".././data/raw_data.json"
    train_data_output = ".././data/train_data.json"

    # Step 3: Create Training Data
    create_training_data(raw_data_output, train_data_output)

    print("Raw data creation and training data generation complete.")
