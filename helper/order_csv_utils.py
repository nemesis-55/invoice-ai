import io
import csv
import json

def convert_order_to_csv_string(order_json):
    rows = []

    # Separate order-level values
    order_level_fields = ["NetAmount", "NetWeight", "GrossWeight"]
    order_level_values = {f"OrderLevel{k}": order_json.get(k, "") for k in order_level_fields}

    # Base info without order items or conflicting fields
    base_info = {
        k: v for k, v in order_json.items()
        if k not in ("OrderItems", *order_level_fields)
    }

    for item in order_json.get("OrderItems", []):
        row = base_info.copy()
        row.update(item)
        row.update(order_level_values)  # Add renamed order-level fields
        rows.append(row)

    if not rows:
        return ""

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=rows[0].keys(), delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()
    writer.writerows(rows)

    return output.getvalue().encode("utf-8").decode("utf-8")

# Convert CSV to JSON
def convert_csv_to_order_json_string(csv_string):
    order_item_properties = [
        "Description", "HsCode", "HsCodeExport", "Quantity", "ArticleNumber",
        "GrossWeight", "NetWeight", "CountryOfOrigin", "NumberOfUnits",
        "TypeOfUnit", "PricePerPiece", "NetAmount"
    ]
    order_data = {}
    order_items = []

    csv_file = io.StringIO(csv_string, newline="")
    reader = csv.DictReader(csv_file)

    for row in reader:
        if not order_data:
            # Extract base info and order-level fields from renamed keys
            order_data = {k: v for k, v in row.items() if k not in order_item_properties}
            for key in ["NetAmount", "NetWeight", "GrossWeight"]:
                order_data[key] = order_data.pop(f"OrderLevel{key}", "")

        item = {k: v for k, v in row.items() if k in order_item_properties}
        order_items.append(item)

    if order_items:
        order_data["OrderItems"] = order_items

    return json.dumps(order_data, ensure_ascii=False, indent=2)
