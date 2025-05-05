import io
import csv
import json

def embed_order_items_csv_in_json(order_json):
    print("DEBUG: Entering embed_order_items_csv_in_json()")
    print(f"DEBUG: Initial order_json: {json.dumps(order_json, ensure_ascii=False, indent=2)}")
    
    order_items = order_json.get("OrderItems", [])
    print(f"DEBUG: Extracted OrderItems: {order_items}")
    
    if not order_items:
        order_json["OrderItemsCSV"] = ""
        print("DEBUG: No OrderItems found. Setting 'OrderItemsCSV' to an empty string.")
    else:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=order_items[0].keys(), delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(order_items)
        order_json["OrderItemsCSV"] = output.getvalue().encode("utf-8").decode("utf-8")
        print("DEBUG: 'OrderItemsCSV' populated with CSV data.")
    
    order_json.pop("OrderItems", None)
    print(f"DEBUG: Final order_json: {json.dumps(order_json, ensure_ascii=False, indent=2)}")
    return order_json

def expand_order_items_csv_to_list(order_json_with_csv):
    print("DEBUG: Entering expand_order_items_csv_to_list()")
    print(f"DEBUG: Initial order_json_with_csv: {json.dumps(order_json_with_csv, ensure_ascii=False, indent=2)}")
    
    order_items_csv = order_json_with_csv.get("OrderItemsCSV", "")
    print(f"DEBUG: Extracted OrderItemsCSV: {order_items_csv}")
    
    if not order_items_csv.strip():
        order_json_with_csv.pop("OrderItemsCSV", None)
        print("DEBUG: 'OrderItemsCSV' is empty or whitespace. Removed from order_json_with_csv.")
        return json.dumps(order_json_with_csv, ensure_ascii=False, indent=2)
    
    try:
        csv_file = io.StringIO(order_items_csv, newline="")
        reader = csv.DictReader(csv_file)
        order_items = []
        for row in reader:
            order_items.append({field: row.get(field, "") for field in reader.fieldnames})
        order_json_with_csv["OrderItems"] = order_items
        order_json_with_csv.pop("OrderItemsCSV", None)
        print("DEBUG: 'OrderItemsCSV' converted to list and replaced in order_json_with_csv.")
    except Exception as e:
        print(f"ERROR: Exception during CSV processing: {e}")
        return f"Error processing CSV: {str(e)}"
    
    print(f"DEBUG: Final order_json_with_csv: {json.dumps(order_json_with_csv, ensure_ascii=False, indent=2)}")
    return json.dumps(order_json_with_csv, ensure_ascii=False, indent=2)
