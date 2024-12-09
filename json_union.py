import json

def get_union_of_json_fields_from_file(file_path):
    try:
        # Read the JSON file
        with open(file_path, 'r') as file:
            json_list = json.load(file)

        # Initialize result dictionary to store the union of fields
        result = {'OrderItems': [{}]}

        # Iterate through the list of JSON objects
        for json_obj in json_list:
            for conversations in json_obj["conversations"]:
                if conversations["role"] == "assistant":
                    for key, value in conversations["content"].items():
                        if value:  # Check if the value is not empty (None, empty string, etc.)
                            if key == 'OrderItems':
                                for item in value:
                                    for item_key, item_value in item.items():
                                        if item_value:
                                            result['OrderItems'][0][item_key] = item_value
                            else:
                                result[key] = value

        return result

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print("Invalid JSON format in file.")
        return None

# Example usage
if __name__ == "__main__":
    file_path = '/Users/saurav.kumar3/Downloads/eval_dataset (3).json'  # Replace with your JSON file path
    result_json = get_union_of_json_fields_from_file(file_path)
    
    if result_json:
        print("Sample JSON with fields having values:")
        print(json.dumps(result_json, indent=4))  # Pretty-print the result
