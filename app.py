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
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True).eval().cuda()

# Load and apply the adapter
adapter = PeftModel.from_pretrained(model, adapter_name)
adapter_config = PeftConfig.from_pretrained(adapter_name)
print(f"adapter_config : {adapter_config}")
model = adapter

@app.post("/predict")
async def predict(prompt: str, image: UploadFile = File(...)):
    try:
        # Process image
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Prepare conversation messages
        msgs = [{
            "role": "user",
            "content": [image, prompt]
        }]
        
        # Get the model's response
        output = model.chat(
            image=None,  # If image is processed separately, set to None
            msgs=msgs,
            tokenizer=tokenizer,
            max_new_tokens=4096
        )
        
        return {"response": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-text")
async def predict_text(prompt: str):
    try:
        # Prepare conversation messages
        msgs = [{
            "role": "user",
            "content": [prompt]
        }]
        
        # Get the model's response
        output = model.chat(
            image=None,  # No image input for this endpoint
            msgs=msgs,
            tokenizer=tokenizer,
            max_new_tokens=4096
        )
        
        return {"response": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}
