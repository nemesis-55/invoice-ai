import io
import csv
import json

def embed_order_items_csv_in_json(order_json):
    """
    Embeds the 'OrderItems' list from the JSON into a CSV format within the JSON.
    
    Parameters:
    - order_json (dict): The input JSON with 'OrderItems' to be embedded as CSV.
    
    Returns:
    - dict: The modified JSON with 'OrderItemsCSV' added and 'OrderItems' removed.
    """
    print("DEBUG: Entering embed_order_items_csv_in_json()")
    print(f"DEBUG: Initial order_json: {json.dumps(order_json, ensure_ascii=False, indent=2)}")
    
    order_items = order_json.get("OrderItems", [])
    print(f"DEBUG: Extracted OrderItems: {order_items}")
    
    # If no OrderItems, set 'OrderItemsCSV' to empty string
    if not order_items:
        order_json["OrderItemsCSV"] = ""
        print("DEBUG: No OrderItems found. Setting 'OrderItemsCSV' to an empty string.")
    else:
        # Convert OrderItems list to CSV format
        order_json["OrderItemsCSV"] = convert_list_to_csv(order_items)
        print("DEBUG: 'OrderItemsCSV' populated with CSV data.")
    
    # Remove 'OrderItems' field
    order_json.pop("OrderItems", None)
    print(f"DEBUG: Final order_json: {json.dumps(order_json, ensure_ascii=False, indent=2)}")
    
    return order_json

def convert_list_to_csv(order_items):
    """
    Converts a list of dictionaries (OrderItems) into a CSV string.
    
    Parameters:
    - order_items (list): List of dictionaries representing the order items.
    
    Returns:
    - str: CSV formatted string representing the order items.
    """
    output = io.StringIO(newline="")
    if order_items:
        writer = csv.DictWriter(output, fieldnames=order_items[0].keys(), delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(order_items)
    
    return output.getvalue().encode("utf-8").decode("utf-8")

def expand_order_items_csv_to_list(order_json_with_csv):
    """
    Expands the 'OrderItemsCSV' field in the input JSON into a list of dictionaries under 'OrderItems'.
    
    Parameters:
    - order_json_with_csv (dict): The input JSON containing 'OrderItemsCSV'.
    
    Returns:
    - dict: The modified JSON with 'OrderItems' populated and 'OrderItemsCSV' removed.
    """
    print("DEBUG: Entering expand_order_items_csv_to_list()")
    print(f"DEBUG: Initial order_json_with_csv: {json.dumps(order_json_with_csv, ensure_ascii=False, indent=2)}")
    order_json_with_csv = json.loads(order_json_with_csv)
    print(f"DEBUG: Parsed order_json_with_csv: {json.dumps(order_json_with_csv, ensure_ascii=False, indent=2)}")
    order_items_csv = order_json_with_csv.get("OrderItemsCSV", "").strip()
    print(f"DEBUG: Extracted OrderItemsCSV: {order_items_csv}")
    
    # If OrderItemsCSV is empty or whitespace, remove the field
    if not order_items_csv:
        order_json_with_csv.pop("OrderItemsCSV", None)
        order_json_with_csv["OrderItems"] = []
        print("DEBUG: 'OrderItemsCSV' is empty or whitespace. Removed from order_json_with_csv.")
        return json.dumps(order_json_with_csv, ensure_ascii=False, indent=2)
    
    try:
        # Convert CSV back into list of dictionaries
        order_json_with_csv["OrderItems"] = convert_csv_to_list(order_items_csv)
        order_json_with_csv.pop("OrderItemsCSV", None)
        print("DEBUG: 'OrderItemsCSV' converted to list and replaced in order_json_with_csv.")
    except Exception as e:
        print(f"ERROR: Exception during CSV processing: {e}")
        return f"Error processing CSV: {str(e)}"
    
    print(f"DEBUG: Final order_json_with_csv: {json.dumps(order_json_with_csv, ensure_ascii=False, indent=2)}")
    return json.dumps(order_json_with_csv, ensure_ascii=False, indent=2)

def convert_csv_to_list(order_items_csv):
    """
    Converts a CSV string back into a list of dictionaries.
    
    Parameters:
    - order_items_csv (str): The CSV string representing order items.
    
    Returns:
    - list: List of dictionaries representing the order items.
    """
    csv_file = io.StringIO(order_items_csv, newline="")
    reader = csv.DictReader(csv_file)
    order_items = []
    
    for row in reader:
        order_items.append({field: row.get(field, "") for field in reader.fieldnames})
    
    return order_items
