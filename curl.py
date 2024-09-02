import base64
import requests
from PIL import Image
import io

def convert_image_to_base64(image_path):
    """Convert an image to Base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_handler(image_base64, prompt, url, auth_key):
    """Simulate an event and call the handler function."""
    # Simulate the event structure expected by the handler
    event = {
        "input": {
            "image": image_base64,
            "prompt": prompt
        }
    }

    headers = {
        "Authorization": f"Bearer {auth_key}",
        "Content-Type": "application/json",
    }
    
    response = requests.post(url, json=event, headers=headers)
    
    # Check for successful response
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Request failed with status code {response.status_code}"}

if __name__ == "__main__":
    # Input: Image file path and prompt
    image_path = "/Users/saurav.kumar3/Desktop/image.png"
    prompt = "What is this image?"

    # Convert the image to Base64
    image_base64 = convert_image_to_base64(image_path)

    # Optionally display the image to verify it's correct
    image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
    image.show()

    # The URL of your API endpoint where the handler is running
    url = "https://api.runpod.ai/v2/i5hcmz42bmoqc2/run"

    # The Authorization key for the API
    auth_key = "GDUKZCHH9CNQF3X44CU53Z5WVX18PD4OWXRH1CGS"

    # Execute the request to the handler
    response = call_handler(image_base64, prompt, url, auth_key)

    # Print the response from the handler
    print("Response from handler:", response)
