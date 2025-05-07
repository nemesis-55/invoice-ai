import json

fieldnames = ['Description', 'HsCode', 'HsCodeExport', 'Quantity', 'ArticleNumber', 'GrossWeight', 'NetWeight', 'CountryOfOrigin', 'NumberOfUnits', 'TypeOfUnit', 'PricePerPiece', 'NetAmount']

def embed_order_items_list_in_json(order_json):
    """
    Embeds the 'OrderItems' list from the JSON as a list of lists under the key 'OrderItemsList'.

    Parameters:
    ----------
    order_json : dict
        The input JSON dictionary containing an 'OrderItems' key with a list of dictionaries.

    Returns:
    -------
    dict
        The modified JSON with 'OrderItemsList' added and 'OrderItems' removed.
    """
    
    order_items = order_json.get("OrderItems", [])
    
    # If no OrderItems, set 'OrderItemsList' to empty list
    if not order_items:
        order_json["OrderItemsList"] = []
    else:
        # Convert OrderItems list to list of lists
        order_json["OrderItemsList"] = convert_json_list_to_list_of_list(order_items)
    
    # Remove 'OrderItems' field
    order_json.pop("OrderItems", None)
    return order_json

def convert_json_list_to_list_of_list(json_list):
    """
    Converts a list of dictionaries (typically parsed from a JSON array) into a list of lists,
    where each inner list contains the values from a dictionary in insertion order.

    Parameters:
    ----------
    json_list : list of dict
        A list where each element is a dictionary representing a JSON object.

    Returns:
    -------
    list of list
        A list of lists, where each inner list contains the values of the corresponding dictionary.
    """
    
    if not json_list:
        return []
    
    # Convert each dictionary in the list to a list of values
    list_of_lists = [list(item.values()) for item in json_list]
    return list_of_lists


def expand_order_items_list_to_json(order_json_with_list):
    """
    Expands the 'OrderItemsList' field in the input JSON into a list of dictionaries under 'OrderItems'.

    Parameters:
    ----------
    order_json_with_list : str
        A JSON string containing 'OrderItemsList', to be parsed and converted.

    Returns:
    -------
    str
        A JSON-formatted string with 'OrderItems' populated and 'OrderItemsList' removed.
    """
    
    order_json_with_list = json.loads(order_json_with_list)
    order_items_list = order_json_with_list.get("OrderItemsList", [])
    
    if not order_items_list:
        order_json_with_list.pop("OrderItemsList", None)
        return json.dumps(order_json_with_list, ensure_ascii=False, indent=2)
    
    try:
        order_json_with_list["OrderItems"] = convert_list_of_list_to_json_list(order_items_list, fieldnames)
        order_json_with_list.pop("OrderItemsList", None)
    except Exception as e:
        print(f"ERROR: Exception during list processing: {e}")
        return f"Error processing list: {str(e)}"
    
    return json.dumps(order_json_with_list, ensure_ascii=False, indent=2)

def convert_list_of_list_to_json_list(list_of_lists, fieldnames):
    """
    Converts a list of lists into a list of dictionaries using the provided fieldnames as keys.

    Parameters:
    ----------
    list_of_lists : list of list
        A list where each inner list represents a row of values.
    fieldnames : list of str
        A list of field names to be used as dictionary keys.

    Returns:
    -------
    list of dict
        A list of dictionaries where each dictionary maps fieldnames to corresponding values.
    """
    
    if not list_of_lists or not fieldnames:
        return []
    
    json_list = [dict(zip(fieldnames, sublist)) for sublist in list_of_lists]
    
    return json_list