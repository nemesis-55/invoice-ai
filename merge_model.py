from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import torch
from huggingface_hub import login


# Define model and adapter paths
MODEL_TYPE = "openbmb/MiniCPM-V-2_6"
ADAPTOR_TYPE = "Zorro123444/invoice_extracter_2"
NEW_MODEL_NAME = "Zorro123444/invoice-ai-2_6-0.1"

# Step 1: Load the base model and tokenizer
print("Loading the tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE, trust_remote_code=True)
print("Tokenizer loaded.")

print("Loading the base model...")
model = AutoModel.from_pretrained(MODEL_TYPE, trust_remote_code=True)
print("Base model loaded successfully.")

# Step 2: Load the LoRA adapter with the model using PeftModel
print(f"Loading the LoRA adapter {ADAPTOR_TYPE}...")
lora_model = PeftModel.from_pretrained(
    model, 
    ADAPTOR_TYPE, 
    trust_remote_code=True
)

# Step 4: Authenticate to Hugging Face (you will need to log in)
print("Logging into Hugging Face...")

login(token="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")  

# Step 5: Push the merged model and tokenizer to Hugging Face
print(f"Pushing the model {NEW_MODEL_NAME} to Hugging Face...")
lora_model.push_to_hub(NEW_MODEL_NAME)
tokenizer.push_to_hub(NEW_MODEL_NAME)
print(f"Model {NEW_MODEL_NAME} pushed successfully.")
