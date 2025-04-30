import io
import csv
import json

# def convert_order_to_csv_string(order_json):
#     rows = []

#     # Separate order-level values
#     order_level_fields = ["NetAmount", "NetWeight", "GrossWeight"]
#     order_level_values = {f"OrderLevel{k}": order_json.get(k, "") for k in order_level_fields}

#     # Base info without order items or conflicting fields
#     base_info = {
#         k: v for k, v in order_json.items()
#         if k not in ("OrderItems", *order_level_fields)
#     }

#     for item in order_json.get("OrderItems", []):
#         row = base_info.copy()
#         row.update(item)
#         row.update(order_level_values)  # Add renamed order-level fields
#         rows.append(row)

#     if not rows:
#         return ""

#     output = io.StringIO(newline="")
#     writer = csv.DictWriter(output, fieldnames=rows[0].keys(), delimiter=',', quoting=csv.QUOTE_MINIMAL)
#     writer.writeheader()
#     writer.writerows(rows)

#     return output.getvalue().encode("utf-8").decode("utf-8")

# Convert CSV to JSON
# def convert_csv_to_order_json_string(csv_string):
#     order_item_properties = [
#         "Description", "HsCode", "HsCodeExport", "Quantity", "ArticleNumber",
#         "GrossWeight", "NetWeight", "CountryOfOrigin", "NumberOfUnits",
#         "TypeOfUnit", "PricePerPiece", "NetAmount"
#     ]
#     order_data = {}
#     order_items = []

#     csv_file = io.StringIO(csv_string, newline="")
#     reader = csv.DictReader(csv_file)

#     for row in reader:
#         if not order_data:
#             # Extract base info and order-level fields from renamed keys
#             order_data = {k: v for k, v in row.items() if k not in order_item_properties}
#             for key in ["NetAmount", "NetWeight", "GrossWeight"]:
#                 order_data[key] = order_data.pop(f"OrderLevel{key}", "")

#         item = {k: v for k, v in row.items() if k in order_item_properties}
#         order_items.append(item)

#     if order_items:
#         order_data["OrderItems"] = order_items

#     return json.dumps(order_data, ensure_ascii=False, indent=2)

def embed_order_items_csv_in_json(order_json):
    """
    Converts the 'OrderItems' list in the given order JSON object to a CSV string,
    embeds it under the 'OrderItemsCSV' key, and removes the original 'OrderItems' list.

    Parameters:
        order_json (dict): A dictionary representing the order, containing an 'OrderItems' key
                           with a list of item dictionaries.

    Returns:
        dict: A new dictionary where 'OrderItems' is replaced by 'OrderItemsCSV' containing
              the serialized CSV string.
    """

    order_items = order_json.get("OrderItems", [])

    if not order_items:
        order_json["OrderItemsCSV"] = ""
    else:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=order_items[0].keys(), delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(order_items)
        order_json["OrderItemsCSV"] = output.getvalue().encode("utf-8").decode("utf-8")

    # Remove the original list
    order_json.pop("OrderItems", None)
    
    return order_json

def expand_order_items_csv_to_list(order_json_with_csv):
    """
    Converts the 'OrderItemsCSV' string in the given JSON object back into a list of dictionaries,
    stores it under the 'OrderItems' key, and removes the 'OrderItemsCSV' key.

    Parameters:
        order_json_with_csv (dict): A dictionary containing an 'OrderItemsCSV' key with a CSV string.

    Returns:
        str: A JSON-formatted string of the full updated order, where 'OrderItemsCSV' is replaced
             by the parsed 'OrderItems' list.
    """
    
    order_items_csv = order_json_with_csv.get("OrderItemsCSV", "")
    
    if not order_items_csv.strip():
        # If the CSV string is empty, just remove it and return
        order_json_with_csv.pop("OrderItemsCSV", None)
        return json.dumps(order_json_with_csv, ensure_ascii=False, indent=2)
    
    csv_file = io.StringIO(order_items_csv, newline="")
    reader = csv.DictReader(csv_file)

    order_items = [row for row in reader]
    
    # Replace CSV string with structured order items list
    order_json_with_csv["OrderItems"] = order_items
    order_json_with_csv.pop("OrderItemsCSV", None)

    return json.dumps(order_json_with_csv, ensure_ascii=False, indent=2)