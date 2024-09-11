from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from PIL import Image
import torch
import io
import base64
import runpod
import bitsandbytes as bnb  # Required for 8-bit model loading

# Define model and adapter paths
base_model_name = "openbmb/MiniCPM-Llama3-V-2_5"
adapter_name = "Zorro123444/xylem_invoice_extracter"

# Load tokenizer and base model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
print("Tokenizer loaded.")

print("Loading base model with 8-bit precision onto GPU...")
# Use device_map={"gpu": 0} to explicitly load model onto GPU 0
model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map={"": "cuda:0"})
print("Base model loaded with 8-bit precision onto GPU.")

# Set model to evaluation mode
model.eval()
print("Model set to evaluation mode.")

# Load and apply the adapter
print(f"Loading adapter from {adapter_name}...")
adapter = PeftModel.from_pretrained(model, adapter_name)
adapter_config = PeftConfig.from_pretrained(adapter_name)
print(f"Adapter loaded with config: {adapter_config}")
model = adapter

def handler(event):
    try:
        # Extract the prompt and image data from the event
        print(f"event: {event}")
        event_data = event.get("input")
        print(f"event_data: {event_data}")
        prompt = event_data.get("prompt", "")
        print(f"prompt: {prompt}")
        image_data = event_data.get("image", None)
        print(f"image_data: {image_data}")
        
        if image_data:
            print("Processing image...")
            # If image_data is coming as a string (assumed to be Base64-encoded)
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            print("Image processed.")
            msgs = [{
                "role": "user",
                "content": [image, prompt]
            }]
        else:
            msgs = [{
                "role": "user",
                "content": [prompt]
            }]
        
        print(f"Messages prepared: {msgs}")
        
        # Get the model's response
        print("Generating response...")
        with torch.no_grad():
            output = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                max_new_tokens=8192
            )
        print("Response generated.")
        return {"response": output}

    except Exception as e:
        print(f"Error in handler: {e}")
        return {"error": str(e)}

def health_check(event):
    print("Health check hit.")
    return {"status": "ok"}

# Mapping functions to the event names (optional if you have multiple handlers)
handler_functions = {
    "handler": handler,
    "health_check": health_check
}

runpod.serverless.start({"handler": handler})
