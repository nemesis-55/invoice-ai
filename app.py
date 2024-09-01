from fastapi import FastAPI, HTTPException, File, UploadFile
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from PIL import Image
import torch
import io

app = FastAPI()

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

@app.post("/predict")
async def predict(prompt: str, image: UploadFile = File(...)):
    try:
        print("inside predict")

        # Process image
        print("Processing image...")
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print("Image processed.")

        # Prepare conversation messages
        msgs = [{
            "role": "user",
            "content": [image, prompt]
        }]
        print(f"Messages prepared: {msgs}")

    
        # Get the model's response
        print("Generating response...")
        with torch.no_grad():
            output = model.chat(
                image=None,  # If image is processed separately, set to None
                msgs=msgs,
                tokenizer=tokenizer,
                max_new_tokens=4096
            )
        print("Response generated.")

        return {"response": output}
    except Exception as e:
        print(f"Error in /predict endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-text")
async def predict_text(prompt: str):
    try:
        print("inside predict-text")
        # Prepare conversation messages
        msgs = [{
            "role": "user",
            "content": [prompt]
        }]
        print(f"Messages prepared: {msgs}")


        # Get the model's response
        print("Generating response...")
        with torch.no_grad():
            output = model.chat(
                image=None,  # No image input for this endpoint
                msgs=msgs,
                tokenizer=tokenizer,
                max_new_tokens=4096
            )
        print("Response generated.")

        return {"response": output}
    except Exception as e:
        print(f"Error in /predict-text endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health():
    print("Health check endpoint hit.")
    return {"status": "ok"}
