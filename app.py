from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Load model and tokenizer
model_name = "openbmb/MiniCPM-Llama3-V-2_5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.post("/predict")
async def predict(prompt: str):
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}
