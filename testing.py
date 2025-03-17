import requests
import time
import base64
import fitz  # PyMuPDF
import os
import json
from concurrent.futures import ThreadPoolExecutor


# API endpoint and headers
endpoint_id = '1yxgq2n20w99oo'
post_url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
get_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/"

headers = {
    'Content-Type': 'application/json',
    'Authorization': 'rpa_IVK8I095G3K2YB26IJ39Y5W5WXRVMBCQ0EASQ8ECg4rvx9',
}

final_response = {}


def page_to_pdf_bytes(page):
    """
    Convert a single PDF page to PDF bytes.

    Args:
        page (fitz.Page): The page object from a PDF.

    Returns:
        bytes: PDF bytes for the single page.
    """
    single_page_doc = fitz.open()  # Create an empty PDF
    single_page_doc.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
    pdf_bytes = single_page_doc.tobytes()
    return pdf_bytes

def make_post_request(pdf_bytes: bytes):
    """
    Sends a POST request with the PDF bytes for a single page.

    Args:
        pdf_bytes (bytes): The PDF file in byte format for a single page.

    Returns:
        str: Task ID received from the API response.
    """
    post_data = {
        "input": {
            "pdf_data": base64.b64encode(pdf_bytes).decode('utf-8')  # Convert PDF bytes to base64 string
        }
    }

    print(f"Sending POST request with PDF bytes...")
    response = requests.post(post_url, headers=headers, json=post_data)
    response_data = response.json()
    task_id = response_data['id']
    print(f"Received Task ID: {task_id}")
    return task_id

def poll_task_status(task_id):
    """
    Polls the API to check the status of a task based on the task ID.

    Args:
        task_id (str): The task ID to poll for status.

    Returns:
        dict: The response data when the task is completed.
    """
    status_url = f"{get_url}{task_id}"

    response = requests.get(status_url, headers=headers)
    response_data = response.json()
    return response_data

def process_pdf(file_path, pages_to_process, pickup_id):
    """
    Loads a PDF file, converts each page into PDF bytes, and sends it page by page to the API.

    Args:
        file_path (str): Path to the PDF file to be processed.
        pages_to_process (int): The number of pages to process in the PDF.
    """
    # Open the PDF document
    pdf_document = fitz.open(file_path)
    complete_response = {}
    # Store the task details in a dictionary
    tasks = {}

    # Only process the first 'pages_to_process' pages
    for page_num in range(min(pages_to_process, len(pdf_document))):
        page = pdf_document.load_page(page_num)  # Load each page
        pdf_bytes = page_to_pdf_bytes(page)  # Convert the page to PDF bytes

        print(f"Processing page {page_num + 1}")

        # Get the task ID after sending the PDF page bytes to the API
        task_id = make_post_request(pdf_bytes)

        # Store task_id and page_number in the dictionary
        tasks[task_id] = {
            "page_number": page_num + 1
        }
    count = -1
    # Polling indefinitely with a 1-minute gap
    while tasks:
        print("\nPolling for task results...")
        for task_id in list(tasks.keys()):
            print(f"Polling for Task ID: {task_id} (Page {tasks[task_id]['page_number']})")
            response_data = poll_task_status(task_id)

            if response_data:
                page_number = tasks[task_id]['page_number']
                status = response_data.get('status')

                if status == "COMPLETED":
                    try:
                        complete_response[page_number] = json.loads(response_data.get('output').get('response'))
                    except Exception as e:
                        print(f"failure while converting taskId:{task_id} page_num: {page_number} response: {response_data.get('output')}")
                        complete_response[page_number] = response_data.get('output').get('response')
                    del tasks[task_id]  # Remove the completed task from the dictionary
                else:
                    print(f"Task {task_id} is still in progress (status: {status}). Waiting 1 minute...")
            else:
                print(f"No response for Task ID: {task_id}")
        count = count + 1
        if tasks:
            time.sleep(60)  # Wait for 1 minute before checking again
    print(f"total time taken in sec: ", count * 10)
    final_response[pickup_id] = complete_response

if __name__ == "__main__":
    # Path to your PDF file
    output_file_path = "./data/model_output_dentalspar_72951.json"

    pdf_data = [
        {
            "pdf_file_path": "/Users/saurav.kumar3/invoice-ai/data/pdf/72951/1_VK1963434_20240313050241.pdf",
            "output_id": "72951"
        }
    ]

    # Input for number of pages to process
    pages_to_process = 1000

    with ThreadPoolExecutor(max_workers=24) as executor:
        for data in pdf_data:
            executor.submit(process_pdf, data.get("pdf_file_path"), pages_to_process, data.get("output_id"))

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(final_response, f, indent=4)
