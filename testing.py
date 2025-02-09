import requests
import time
import base64
import fitz  # PyMuPDF
import os


# API endpoint and headers
endpoint_id = '21m7yr1wkorqc6'
post_url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
get_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/"

headers = {
    'Content-Type': 'application/json',
    'Authorization': 'rpa_IVK8I095G3K2YB26IJ39Y5W5WXRVMBCQ0EASQ8ECg4rvx9',
}

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

def process_pdf(file_path, pages_to_process):
    """
    Loads a PDF file, converts each page into PDF bytes, and sends it page by page to the API.

    Args:
        file_path (str): Path to the PDF file to be processed.
        pages_to_process (int): The number of pages to process in the PDF.
    """
    # Open the PDF document
    pdf_document = fitz.open(file_path)

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
                    print(f"Page {page_number} completed! Response: {response_data}")
                    del tasks[task_id]  # Remove the completed task from the dictionary
                else:
                    print(f"Task {task_id} is still in progress (status: {status}). Waiting 1 minute...")
            else:
                print(f"No response for Task ID: {task_id}")
        count = count + 1
        time.sleep(30)  # Wait for 1 minute before checking again
    print(f"total time taken in sec: ", count * 10)

if __name__ == "__main__":
    # Path to your PDF file
    pdf_file_path = "/Users/saurav.kumar3/Downloads/[Untitled]_20250117084540.pdf"  # Replace with the actual file path
    
    # Input for number of pages to process
    pages_to_process = int(input("Enter the number of pages to process: "))
    
    # Process the PDF and get the results
    process_pdf(pdf_file_path, pages_to_process)

    print(f"Processing completed for {os.path.basename(pdf_file_path)}")
