from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import io
import base64
import runpod
import torch

# Define model and adapter paths
model_name = "openbmb/MiniCPM-Llama3-V-2_5"
adapter_name = "Zorro123444/xylem_invoice_extracter"

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print("Tokenizer loaded.")

print("Loading  model in 16bit precision onto GPU...")
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="balanced"  # Automatically balance model across available GPUs
    ).cuda().eval()
    
print(" model loaded in 16bit precision.")


def handler(event):
    try:
        # Extract the prompt and image data from the event
        print(f"event: {event}")
        event_data = event.get("input")
        prompt = event_data.get("prompt", "")
        print(f"prompt: {prompt}")
        image_data = event_data.get("image", None)
        print(f"image_data: {image_data}")

        if image_data:
            print("Processing image...")
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            print("Image processed.")
            msgs = [{"role": "user", "content": [image, prompt]}]
        else:
            msgs = [{"role": "user", "content": [prompt]}]

        print(f"Messages prepared: {msgs}")
        
        # Tokenize and generate response
        print("Generating response...")
        with torch.no_grad():
            output = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                max_new_tokens=8192
            )
        return {"response": output}

    except Exception as e:
        print(f"Error in handler: {e}")
        return {"error": str(e)}

def health_check(event):
    print("Health check hit.")
    return {"status": "ok"}

# Start RunPod serverless function
runpod.serverless.start({"handler": handler})
