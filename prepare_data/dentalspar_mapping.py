import json

# Load extracted invoice JSON
json_file_path = "/Users/saurav.kumar3/invoice-ai/data/raw_output/72110/ExtractedData/1_285931-VK1963103_20240308072854.pdf/1_285931-VK1963103_20240308072854.pdf.json"  # Update path if needed
with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

properties = data.get("Properties", {})
order_items = properties.get("OrderItems", [])

# Initialize structured output
structured_data = {}

# Group order items by page
for item in order_items:
    page = str(item.get("PageNumber", 1))  # Default to page 1 if missing
    if page not in structured_data:
        if page == "1":
            print("page 1")
            structured_data[page] = {
                "OrderNumber": properties.get("OrderNumber", ""),
                "InvoiceNumber": properties.get("InvoiceNumber", ""),
                "BuyerName": properties.get("BuyerName", ""),
                "BuyerAddress1": properties.get("BuyerAddress1", ""),
                "BuyerZipCode": properties.get("BuyerZipCode", ""),
                "BuyerCity": properties.get("BuyerCity", ""),
                "BuyerCountry": properties.get("BuyerCountry", ""),
                "ReceiverName": properties.get("ReceiverName", ""),
                "ReceiverAddress1": properties.get("ReceiverAddress1", ""),
                "ReceiverZipCode": properties.get("ReceiverZipCode", ""),
                "ReceiverCity": properties.get("ReceiverCity", ""),
                "ReceiverCountry": properties.get("ReceiverCountry", ""),
                "SellerName": properties.get("SellerName", ""),
                "NetAmount": properties.get("NetAmount", ""),
                "OrderDate": properties.get("OrderDate", ""),
                "Currency": properties.get("Currency", ""),
                "TermsOfDelCode": properties.get("TermsOfDelCode", ""),
                "OrderItems": [],
                "NetWeight": "",
                "NumberOfUnits": ""
            }
        else:
            structured_data[page] = {
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

    # Add order items for the corresponding page
    structured_data[page]["OrderItems"].append({
        "Description": item.get("Description", ""),
        "HsCode": item.get("HsCode", ""),
        "HsCodeExport": item.get("HsCodeExport", ""),
        "Quantity": item.get("Quantity", ""),
        "ArticleNumber": item.get("ArticleNumber", ""),
        "GrossWeight": item.get("GrossWeight", ""),
        "NetWeight": item.get("NetWeight", ""),
        "CountryOfOrigin": item.get("CountryOfOrigin", ""),
        "NumberOfUnits": item.get("NumberOfUnits", ""),
        "TypeOfUnit": item.get("TypeOfUnit", ""),
        "PricePerPiece": item.get("PricePerPiece", ""),
        "NetAmount": item.get("NetAmount", "")
    })

# Save structured output to JSON file
output_path = "/Users/saurav.kumar3/invoice-ai/data/dentalspar_72110.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(structured_data, f, indent=4)

print(f"Structured invoice data saved to: {output_path}")
