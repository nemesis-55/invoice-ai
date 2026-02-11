import os
# MUST be set before any CUDA operations to prevent memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

import base64
from pydantic import ValidationError
from models.payloads.PromptPayload import PromptPayload
from models.payloads.InvoiceExtractionPayload import InvoiceExtractionPayload
from models.payloads.AssistantPayload import AssistantPayload
import torch
from PIL import Image
import fitz  # PyMuPDF for handling PDFs
from transformers import AutoTokenizer, AutoModel
import runpod
from huggingface_hub import login, scan_cache_dir
from helper.order_csv_utils import expand_order_items_list_to_json
import time
import json
import io

# Cache config: Ensure Hugging Face cache uses mounted volume (not /root)
cache_name_env = os.getenv("INVOICE_AI_CACHE_DIR", "cache").strip()
adaptor_type_env = os.getenv("MODEL_ADAPTOR", "openbmb/MiniCPM-V-4_5").strip()
DEEP_THINKING = os.getenv("DEEP_THINKING", "false").strip().lower() == "true"

CACHE_DIR = f"/runpod-volume/{cache_name_env}"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

# Constants
MODEL_DPI = 300
ADAPTOR_TYPE = adaptor_type_env
cache = os.environ["HF_HOME"]
MODEL_LOAD_ERROR = None

CLASSIFICATION_SYSTEM_PROMPT = (
    "You extract customer_name and waybill from email subject lines. "
    "Respond with ONLY a JSON object, no other text.\n"
    "Format: {\"customer_name\": \"...\", \"waybill\": \"...\"}\n"
    "Rules:\n"
    "- Use null if a value is not found\n"
    "- Never guess or hallucinate values\n"
    "- No explanation, no markdown, just raw JSON"
)

# One-time cache cleanup (remove old unreferenced revisions to free space)
try:
    cache_info = scan_cache_dir(cache_dir=cache)
    delete_hashes = []
    for repo in cache_info.repos:
        # Sort revisions newest first (keep newest always)
        revisions = sorted(
            list(repo.revisions),
            key=lambda r: getattr(r, "last_modified", 0),
            reverse=True
        )
        for rev in revisions[1:]:  # skip most recent
            # Some versions expose refs on revision, some on repo; be defensive
            rev_refs = getattr(rev, "refs", None)
            if rev_refs in (None, set(), frozenset()):
                delete_hashes.append(rev.commit_hash)
    if delete_hashes:
        strategy = cache_info.delete_revisions(*delete_hashes)
        print(f"Cache cleanup: will free {strategy.expected_freed_size_str} removing {len(delete_hashes)} old revisions")
        strategy.execute()
    else:
        print("Cache cleanup: nothing to remove")
except Exception as e:
    print(f"Cache cleanup skipped: {e}")

# Hugging Face login
login(os.getenv("HF_TOKEN"))

# Helper function for device map configuration
def get_device_load_kwargs(device_map, retry=False):
    """Build load_kwargs with device map and memory budget configuration."""
    GPU_MAX_MEMORY = os.getenv("GPU_MAX_MEMORY", "40GiB").strip()
    CPU_MAX_MEMORY = os.getenv("CPU_MAX_MEMORY", "16GiB").strip()
    NUM_GPUS = int(os.getenv("NUM_GPUS", "0").strip())  # 0 = auto-detect
    
    load_kwargs = {
        "device_map": device_map,
        "attn_implementation": "sdpa",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "cache_dir": cache,
    }
    
    if device_map == "auto":
        # Auto-detect GPU count or use user-specified value
        if NUM_GPUS <= 0:
            detected_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        else:
            detected_gpus = NUM_GPUS
        
        retry_label = " (retry)" if retry else ""
        print(f"Multi-GPU mode{retry_label}: detected/configured {detected_gpus} GPU(s)")
        
        # Build max_memory dict for ALL available GPUs
        max_mem = {i: GPU_MAX_MEMORY for i in range(detected_gpus)}
        max_mem["cpu"] = CPU_MAX_MEMORY
        load_kwargs["max_memory"] = max_mem
        
        print(f"Memory budget{retry_label}: {max_mem}")
    
    return load_kwargs

# Load Model and Tokenizer
def load_model_and_tokenizer():
    """Load the main model and tokenizer."""

    global MODEL_LOAD_ERROR
    try:
        start_time = time.time()
        
        print(f"Loading tokenizer and model for adaptor: {ADAPTOR_TYPE}")
        print("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(ADAPTOR_TYPE, trust_remote_code=True)
        print("Tokenizer loaded successfully")
    except Exception as e:
        MODEL_LOAD_ERROR = f"Failed to load tokenizer: {str(e)}"
        print(f"Failed to load tokenizer: {MODEL_LOAD_ERROR}")
        return None, None
    
    try:
        print("Loading model...")
        
        # Determine device map
        GPU_DEVICE = os.getenv("GPU_DEVICE", "single").strip()
        
        if GPU_DEVICE == "single":
            device_map = "cuda:0"
        elif GPU_DEVICE == "auto":
            device_map = "auto"
        else:
            device_map = GPU_DEVICE

        # Get load kwargs with device map and memory configuration
        load_kwargs = get_device_load_kwargs(device_map)

        model = AutoModel.from_pretrained(ADAPTOR_TYPE, **load_kwargs).eval()
        print("Model loaded successfully")
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
        
        # Log which devices the model landed on
        if hasattr(model, 'hf_device_map'):
            devices_used = set(str(v) for v in model.hf_device_map.values())
            print(f"Model distributed across devices: {devices_used}")

    except Exception as e:
        print(f"Initial model load failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        # free cached memory to reduce chance of OOM due to fragmentation
        if torch.cuda.is_available():
            print("Clearing CUDA cache...")
            torch.cuda.empty_cache()

            print("Attempting to load model again...")
            try:
                # Get load kwargs with retry flag
                load_kwargs = get_device_load_kwargs(device_map, retry=True)

                model = AutoModel.from_pretrained(ADAPTOR_TYPE, **load_kwargs).eval()
                print("Model loaded successfully on second attempt")
                print(f"Model loaded in {time.time() - start_time:.2f} seconds")
                
                # Log which devices the model landed on
                if hasattr(model, 'hf_device_map'):
                    devices_used = set(str(v) for v in model.hf_device_map.values())
                    print(f"Model distributed across devices: {devices_used}")
            except Exception as e2:
                MODEL_LOAD_ERROR = f"Failed to load model on both attempts: {str(e2)}"
                print(f"Failed to load model: {MODEL_LOAD_ERROR}")
                return None, None
        else:
            MODEL_LOAD_ERROR = str(e)
            print(f"Failed to load model and CUDA not available: {MODEL_LOAD_ERROR}")
            return None, None
    
    try:
        messages = [
            {"role": "user", "content": "hey"}
        ]
        print(f"Test message: {messages}")
        response = model.chat(image=None, msgs=messages, tokenizer=tokenizer, max_new_tokens=512)
        print(f"Test response: {response}")
    except Exception as e:
        print(f"Test inference failed: {str(e)}")
        print("But model and tokenizer loaded, continuing...")

    print("Model loading complete")
    return model, tokenizer


# Convert PDF Page to Image
def pdf_to_image(pdf_bytes, dpi=MODEL_DPI):
    """Convert a single-page PDF to an image."""
    pdf_document = None
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = pdf_document.load_page(0)
        pix = page.get_pixmap(dpi=dpi)
        mode = "RGBA" if pix.alpha else "RGB"
        image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        return image
    except Exception as e:
        print(f"Error converting PDF to image: {e}")
        raise ValueError(f"Error converting PDF to image: {e}")
    finally:
        if pdf_document:
            pdf_document.close()

# Deep Thinking Helper
def prepare_messages_with_thinking(messages, deep_thinking=False):
    """Prepare messages with optional deep thinking system prompt."""
    if deep_thinking:
        system_msg = {"role": "system", "content": "You are a helpful assistant. Think step by step carefully before responding."}
        return [system_msg] + messages
    return messages

# Clear Image References Helper
def clear_image_references(messages):
    """Clear image references from messages to allow garbage collection."""
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            msg["content"] = [c for c in content if isinstance(c, str)]

# JSON Response Parser
def parse_json_response(response_text):
    """
    Robustly parse JSON from model response with multiple fallback strategies.
    
    Handles:
    - Direct JSON parsing
    - Stripping markdown code blocks (```json ... ```)
    - Finding bare JSON objects via regex
    - Returns fallback dict with raw response on parse failure
    """
    import re
    
    # Try direct parsing first
    try:
        return json.loads(response_text)
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Try stripping markdown code blocks
    stripped = response_text.strip()
    if stripped.startswith("```"):
        # Remove opening ```json or ```
        lines = stripped.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove closing ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
        
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Try finding JSON object with regex
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, response_text)
    
    for match in matches:
        try:
            return json.loads(match)
        except (json.JSONDecodeError, ValueError):
            continue
    
    # All parsing attempts failed - return fallback
    return {
        "raw_response": response_text,
        "parse_error": True
    }

# Generate Detailed Prompt
def generate_prompt(pdf_bytes):
    """Create the detailed prompt for the model."""
    try:
        image = pdf_to_image(pdf_bytes)
        question = (
            "Extract the following fields from the invoice image and return a JSON object:\n"
            "- OrderNumber\n"
            "- InvoiceNumber\n"
            "- BuyerName\n"
            "- BuyerAddress1\n"
            "- BuyerZipCode\n"
            "- BuyerCity\n"
            "- BuyerCountry\n"
            "- ReceiverName\n"
            "- ReceiverAddress1\n"
            "- ReceiverZipCode\n"
            "- ReceiverCity\n"
            "- ReceiverCountry\n"
            "- SellerName\n"
            "- NetAmount\n"
            "- OrderDate (YYYY-MM-DD)\n"
            "- Currency\n"
            "- TermsOfDelCode\n"
            "- ActualFreight\n"
            "- OrderItemsList: a list of lists. Each inner list represents one item and follows the column order:\n"
            "  ['Description', 'HsCode', 'HsCodeExport', 'Quantity', 'ArticleNumber', 'GrossWeight', "
            "'NetWeight', 'CountryOfOrigin', 'NumberOfUnits', 'TypeOfUnit', 'PricePerPiece', 'NetAmount']\n"
            "- NetWeight\n"
            "- OtherAmount\n"
            "- NumberOfUnits\n"
            "Use exact text from the image. If a value is missing, set it to an empty string \"\".\n"
            "Respond with only the JSON object."
        )
        return [{"role": "user", "content": [image, question]}]
    except Exception as e:
        print(f"Error generating prompt: {e}")
        raise RuntimeError(f"Error generating prompt: {e}")

# Handle Inference
def perform_inference(messages, model, tokenizer, max_new_tokens=512):
    """Perform model inference."""
    try:
        with torch.no_grad():
            print(f"Inference messages: {messages}")
            response = model.chat(image=None, msgs=messages, tokenizer=tokenizer, max_new_tokens=max_new_tokens)
            print(f"Inference response: {response}")
        return response
    except RuntimeError as e:
        # detect CUDA OOM and provide actionable message
        if "out of memory" in str(e).lower():
            print(f"Inference failed (OOM): {e}")
            # try one more time after clearing cache (best-effort)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                response = model.chat(image=None, msgs=messages, tokenizer=tokenizer, max_new_tokens=max_new_tokens)
                return response
            except Exception:
                raise RuntimeError("Inference failed due to CUDA OOM.")
        print(f"Inference failed: {e}")
        raise RuntimeError(f"Inference failed: {e}")
    finally:
        # ALWAYS clean up after inference regardless of success/failure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Main Request Handler
def run(request):
    """Process incoming requests."""
    try:
        global model, tokenizer, MODEL_LOAD_ERROR

        # Check for model load errors and attempt to reload
        if MODEL_LOAD_ERROR:
            print(f"Model load error detected: {MODEL_LOAD_ERROR}. Attempting to reload...")
            
            # Explicitly free old model to prevent VRAM leak during reload
            if 'model' in globals() and model is not None:
                del model
            if 'tokenizer' in globals() and tokenizer is not None:
                del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            model, tokenizer = load_model_and_tokenizer()
            if model is None or tokenizer is None:
                print(f"Model reload failed: {MODEL_LOAD_ERROR}")
                return {"error": f"Model reload failed: {MODEL_LOAD_ERROR}"}

            print("Model successfully reloaded")

        payload = request.get("input", {})
        action = payload.get("action")
        data = payload.get("data", {})

        if action == "INVOICE_EXTRACTION":
            return handle_extract_invoice(data)
        elif action == "PROMPT":
            return handle_prompt(data)
        elif action == "ASSISTANT":
            return handle_assistant_request(data)
        elif action == "CLASSIFICATION":
            return handle_classification(data)
        else:
            return {"error": f"Unknown action: {action}"}
    
    except Exception as e:
        print(f"Exception during processing: {e}")
        return {"error": f"Exception during processing: {e}"}

def handle_classification(data):
    try:
        payload = PromptPayload(**data)
        subject_text = payload.prompt

        messages = [
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": subject_text}
        ]

        response = perform_inference(messages, model, tokenizer, max_new_tokens=128)
        response = parse_json_response(response)

        return {"response": response}
    except Exception as e:
        print(f"Classification error: {e}")
        return {"error": f"Classification error: {e}"}

def handle_extract_invoice(data):
    try:
        payload = InvoiceExtractionPayload(**data)
    except (TypeError, ValidationError) as e:
        print(f"Invalid prompt payload: {e}")
        return {"error": f"Invalid prompt payload: {e}"}

    pdf_data = payload.pdf_data
    page_number = payload.page_number or "0"

    if not pdf_data:
        print("Missing PDF data.")
        return {"error": "Missing PDF data."}

    pdf_bytes = base64.b64decode(pdf_data)
    prompt = generate_prompt(pdf_bytes)
    
    # Determine if deep thinking should be used
    use_deep_thinking = payload.deep_thinking if payload.deep_thinking is not None else DEEP_THINKING
    max_tokens = 2048 if use_deep_thinking else 512
    
    prompt = prepare_messages_with_thinking(prompt, use_deep_thinking)
    response = perform_inference(prompt, model, tokenizer, max_new_tokens=max_tokens)

    # Clear image references from messages to allow GC
    clear_image_references(prompt)

    # this assumes the response is a JSON string, so in the prompt it should be mentioned to return a JSON string
    try:
        response = json.loads(response)
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        return {"error": f"Error parsing JSON response: {e}"}
    
    # add key value pair for page number in response for all order items
    for item in response.get("OrderItemsList", []):
        item.append(int(page_number))

    json_response = expand_order_items_list_to_json(response)
    return {"response": json_response}

def handle_prompt(data):
    try:
        payload = PromptPayload(**data)
    except (TypeError, ValidationError) as e:
        print(f"Invalid prompt payload: {e}")
        return {"error": f"Invalid prompt payload: {e}"}
    
    # Determine if deep thinking should be used
    use_deep_thinking = payload.deep_thinking if payload.deep_thinking is not None else DEEP_THINKING
    max_tokens = 2048 if use_deep_thinking else 512
    
    messages = [{"role": "user", "content": payload.prompt}]
    messages = prepare_messages_with_thinking(messages, use_deep_thinking)
    
    response = perform_inference(messages, model, tokenizer, max_new_tokens=max_tokens)

    # this assumes the response is a JSON string, so in the prompt it should be mentioned to return a JSON string
    try:
        response = json.loads(response)
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        return {"error": f"Error parsing JSON response: {e}"}
    return {"response": response}

# # Create a new handler function to handle assistant requests
def handle_assistant_request(data):
    try:
        payload = AssistantPayload(**data)
    except (TypeError, ValidationError) as e:
        print(f"Invalid prompt payload: {e}")
        return {"error": f"Invalid prompt payload: {e}"}
    
    images = []
    if payload.attachments:
        for attachment in payload.attachments:
            try:
                img_bytes = base64.b64decode(attachment.data)
                image = pdf_to_image(img_bytes)
                images.append(image)
            except Exception as e:
                print(f"Error decoding attachment image: {e}")
                return {"error": f"Error decoding attachment image: {e}"}

    # Determine if deep thinking should be used
    use_deep_thinking = payload.deep_thinking if payload.deep_thinking is not None else DEEP_THINKING
    max_tokens = 2048 if use_deep_thinking else 512

    messages = [{"role": "user", "content": images + [payload.prompt]}]
    messages = prepare_messages_with_thinking(messages, use_deep_thinking)
    
    response = perform_inference(messages, model, tokenizer, max_new_tokens=max_tokens)
    
    # Clear image references from messages to allow GC
    clear_image_references(messages)
    
    return {"response": response}



model, tokenizer = load_model_and_tokenizer()

# Initialize and Start RunPod Handler
if __name__ == "__main__":
    print("Initializing RunPod serverless handler")
    runpod.serverless.start({"handler": run})
