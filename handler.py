import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from peft import PeftModel, PeftConfig
from PIL import Image
import io
import base64
import runpod
from bitsandbytes import BitsAndBytesConfig  # Import the new config for 8-bit quantization

# Define model and adapter paths
base_model_name = "openbmb/MiniCPM-Llama3-V-2_5"
adapter_name = "Zorro123444/xylem_invoice_extracter"

# Choose precision mode: '8bit' or '16bit'
precision_mode = "16bit"  # Change this to '16bit' to use 16-bit precision

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
print("Tokenizer loaded.")

# Load model based on precision mode
if precision_mode == "8bit":
    print("Loading base model with 8-bit precision onto GPU using BitsAndBytesConfig...")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = LlamaForCausalLM.from_pretrained(base_model_name, quantization_config=quantization_config, device_map="cuda", trust_remote_code=True)
    print("Base model loaded with 8-bit precision.")
elif precision_mode == "16bit":
    print("Loading base model in bfloat16 precision onto GPU...")
    model = LlamaForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    print("Base model loaded in bfloat16 precision.")

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
        input_ids = tokenizer.encode(f"<用户>{prompt}<AI>", return_tensors='pt', add_special_tokens=True).cuda()
        responds = model.generate(input_ids, temperature=0.3, top_p=0.8, repetition_penalty=1.02, max_length=4096)
        output = tokenizer.decode(responds[0], skip_special_tokens=True)
        print("Response generated.")
        
        return {"response": output}

    except Exception as e:
        print(f"Error in handler: {e}")
        return {"error": str(e)}

def health_check(event):
    print("Health check hit.")
    return {"status": "ok"}

# Start RunPod serverless function
runpod.serverless.start({"handler": handler})
