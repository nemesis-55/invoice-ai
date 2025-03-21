import json

# File paths
new_data_path = "/Users/saurav.kumar3/invoice-ai/data/raw_data.json"
image_path_map_path = "/Users/saurav.kumar3/invoice-ai/data/image_path_map.json"
output_path = "/Users/saurav.kumar3/invoice-ai/data/mapped_output.json"

# Load JSON files
with open(new_data_path, "r", encoding="utf-8") as f:
    new_data = json.load(f)

with open(image_path_map_path, "r", encoding="utf-8") as f:
    image_path_map = json.load(f)

# Default empty data structure
empty_data = {
    "OrderNumber": "",
    "InvoiceNumber": "",
    "BuyerName": "",
    "BuyerAddress1": "",
    "BuyerZipCode": "",
    "BuyerCity": "",
    "BuyerCountry": "",
    "ReceiverName": "",
    "ReceiverAddress1": "",
    "ReceiverZipCode": "",
    "ReceiverCity": "",
    "ReceiverCountry": "",
    "SellerName": "",
    "NetAmount": "",
    "OrderDate": "",
    "Currency": "",
    "TermsOfDelCode": "",
    "OrderItems": [],
    "NetWeight": "",
    "NumberOfUnits": ""
}

# Map images to invoice data
mapped_data = {}

pickupId = {"79886", "71823", "72110", "78053", "72951", "64191", "64189", "66062", "64461"}  # Use a set for faster lookup

for doc_id, pages in image_path_map.items():
    if doc_id not in pickupId:
        continue
    
    # Ensure doc_id exists in mapped_data
    if doc_id not in mapped_data:
        mapped_data[doc_id] = {}

    for page_num, image_path in pages.items():
        page_data = new_data.get(doc_id, {}).get(page_num, {})
        if page_data == {}:
            continue
        mapped_data[doc_id][page_num] = {
            "image_path": image_path,
            "data": page_data
        }

# Save mapped output
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(mapped_data, f, indent=4)

print(f"Mapped output saved to: {output_path}")
