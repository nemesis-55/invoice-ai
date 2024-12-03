import base64
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import pytesseract
import runpod
from huggingface_hub import login
import fitz  # PyMuPDF

# Hugging Face Login
def authenticate_huggingface(token):
    try:
        login(token)
        print("Logged in to Hugging Face successfully.")
    except Exception as e:
        print(f"Error logging in to Hugging Face: {e}")
        raise

# Load Model and Tokenizer
def load_model_and_tokenizer(model_type):
    try:
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
        print("tokenizer loaded")
        model = AutoModel.from_pretrained(model_type, trust_remote_code=True, device_map="cuda").cuda().eval()
        print("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        raise

# Convert PDF Bytes to Images
def pdf_bytes_to_images(pdf_bytes, dpi=600):
    try:
        print("Converting PDF bytes to images...")
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        images_with_page = {
            str(page_num + 1): Image.frombytes(
                "RGB", [pixmap.width, pixmap.height], pixmap.samples
            )
            for page_num, pixmap in enumerate(
                [pdf_document.load_page(i).get_pixmap(dpi=dpi) for i in range(len(pdf_document))]
            )
        }
        print(f"Converted {len(images_with_page)} pages to images.")
        return images_with_page
    except Exception as e:
        print(f"Error converting PDF bytes to images: {e}")
        raise

# Extract Text from PDF Bytes
def extract_text_from_pdf_bytes(pdf_bytes, dpi=600):
    try:
        print("Extracting text from PDF bytes...")
        images = pdf_bytes_to_images(pdf_bytes, dpi)
        text = "\n".join(
            pytesseract.image_to_string(image) for page_num, image in images.items()
        )
        print("Text extraction completed.")
        return text
    except Exception as e:
        print(f"Error during text extraction: {e}")
        raise

# Generate Detailed Prompt
def generate_detailed_prompt(ocr_data):
    try:
        print("Generating detailed prompt...")
        question = (
            "You are an AI model specialized in data extraction from invoices. "
            "Below, you are provided with OCR-extracted text from an invoice. "
            "Your task is to analyze the OCR data and extract key details to structure them as a JSON object.\n\n"
            f"### OCR Data:\n{ocr_data}\n\n"
            "### Instructions:\n"
            "1. Extract all the required fields as specified in the JSON structure below.\n"
            "2. Ensure the output is a syntactically valid JSON string.\n"
            "3. If a field is missing or unavailable in the OCR text, set its value to an empty string \"\".\n"
            "4. Maintain the exact formatting of numeric values and dates as found in the input.\n"
            "5. Do not include additional explanations or comments in your output.\n\n"
            "### JSON Structure:\n"
            "{...}\n"
        )
        return [{"role": "user", "content": question}]
    except Exception as e:
        print(f"Error generating detailed prompt: {e}")
        raise

# Perform Inference
def handle_inference(prompt, model, tokenizer):
    try:
        print("Performing inference...")
        with torch.no_grad():
            response = model.chat(image=None, msgs=prompt, tokenizer=tokenizer, max_new_tokens=8192)
        print("Inference completed.")
        return response
    except Exception as e:
        print(f"Error during inference: {e}")
        return {"error": f"Inference failed: {e}"}

# Main Request Handler
def run(request):
    try:
        print("Processing request...")
        input_data = request.get("input", {})
        pdf_data = input_data.get("pdf_data")
        ocr_data = input_data.get("ocr_data")

        if not pdf_data:
            return {"error": "Missing PDF data."}

        pdf_bytes = base64.b64decode(pdf_data)
        if not ocr_data:
            print("No OCR data provided. Extracting from PDF...")
            ocr_data = extract_text_from_pdf_bytes(pdf_bytes)

        prompt = generate_detailed_prompt(ocr_data)
        response = handle_inference(prompt, model, tokenizer)

        print("Request processed successfully.")
        return {"response": response}
    except Exception as e:
        print(f"Error processing request: {e}")
        return {"error": f"Exception during processing: {e}"}

# Authenticate and Load Resources
if __name__ == "__main__":
    HUGGINGFACE_TOKEN = "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"
    authenticate_huggingface(HUGGINGFACE_TOKEN)
    model, tokenizer = load_model_and_tokenizer("openbmb/MiniCPM-V-2_6")
    runpod.serverless.start({"handler": run})
