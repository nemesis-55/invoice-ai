from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from PIL import Image
import torch
import io

# Define model and adapter paths
base_model_name = "openbmb/MiniCPM-Llama3-V-2_5"
adapter_name = "Zorro123444/xylem_invoice_extracter"

# Load tokenizer and base model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
print("Tokenizer loaded.")

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
print("Base model loaded.")

# Move model to GPU and set to evaluation mode
model = model.cuda()
model.eval()
print("Model moved to GPU and set to evaluation mode.")

# Load and apply the adapter
print(f"Loading adapter from {adapter_name}...")
adapter = PeftModel.from_pretrained(model, adapter_name)
adapter_config = PeftConfig.from_pretrained(adapter_name)
print(f"Adapter loaded with config: {adapter_config}")
model = adapter

def handler(event):
    try:
        # Extract the prompt and image data from the event
        prompt = event.get("prompt", "")
        image_data = event.get("image", None)
        
        if image_data:
            print("Processing image...")
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
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
                max_new_tokens=4096
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
