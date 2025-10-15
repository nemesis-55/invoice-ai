import base64
from pydantic import ValidationError
from models.payloads.PromptPayload import PromptPayload
from models.payloads.InvoiceExtractionPayload import InvoiceExtractionPayload
from models.payloads.AssistantPayload import AssistantPayload
import torch
from PIL import Image
import fitz  # PyMuPDF for handling PDFs
from transformers import AutoTokenizer, AutoModel, AutoModelForVision2Seq, Qwen2VLForConditionalGeneration, AutoProcessor
# qwen_vl utilities (may come from qwen2.5-vl repo / package)
try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None
import runpod
from huggingface_hub import login, scan_cache_dir
import base64
import fitz  # PyMuPDF
from peft import PeftModel
import os
from helper.order_csv_utils import expand_order_items_list_to_json
import time
import json
import io

# Cache config: Ensure Hugging Face cache uses mounted volume (not /root)
cache_name_env = os.getenv("INVOICE_AI_CACHE_DIR", "cache").strip()
adaptor_type_env = os.getenv("MODEL_ADAPTOR", "GothiaDigitalSolutions/invoice-extractor-3.0").strip()

CACHE_DIR = f"/runpod-volume/{cache_name_env}"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

# Constants
MODEL_DPI = 300
ADAPTOR_TYPE = adaptor_type_env
cache = os.environ["HF_HOME"]

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

# Load Model and Tokenizer
def load_model_and_tokenizer():
    """Load the main model, tokenizer and (optionally) processor.

    Returns: (model, tokenizer, processor_or_None, backend_str)
    backend_str is one of 'chat' (legacy model.chat) or 'qwen_vl' (processor+generate).
    """
    try:
        print(f"Loading tokenizer and model for adaptor: {ADAPTOR_TYPE}")
        print("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(ADAPTOR_TYPE, trust_remote_code=True, cache_dir=cache)

        # Heuristic: if adaptor name contains 'qwen' use Qwen2VLForConditionalGeneration + AutoProcessor
        adaptor_low = ADAPTOR_TYPE.lower()
        if "qwen" in adaptor_low or "qwen2" in adaptor_low:
            print("Detected Qwen-family adaptor, attempting Qwen2VLForConditionalGeneration + AutoProcessor")

            # Processor: try adapter repo first, then fall back to a base HF model if preprocessor isn't present
            base_model = os.getenv("HF_BASE_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
            try:
                processor = AutoProcessor.from_pretrained(ADAPTOR_TYPE, trust_remote_code=True, cache_dir=cache)
                print(f"Loaded processor from adapter repo: {ADAPTOR_TYPE}")
            except Exception as e:
                print(f"Processor not found in adapter repo ({ADAPTOR_TYPE}): {e}. Falling back to base processor: {base_model}")
                processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True, cache_dir=cache)

            # Model weights: adapter repos often only contain adapter/safetensors.
            # Try loading a full model from the adapter repo; if that fails, load the base
            # Qwen model and attach the adapter with PeftModel.from_pretrained.
            try:
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    ADAPTOR_TYPE,
                    device_map="cuda",
                    attn_implementation="sdpa",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    cache_dir=cache,
                ).cuda().eval()
                print(f"Loaded full model from {ADAPTOR_TYPE}")
            except Exception as e:
                print(f"Failed to load full model from adapter repo ({ADAPTOR_TYPE}): {e}\nLoading base model {base_model} and applying adapter via PEFT")
                base = Qwen2VLForConditionalGeneration.from_pretrained(
                    base_model,
                    device_map="cuda",
                    attn_implementation="sdpa",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    cache_dir=cache,
                )
                # Wrap base model with adapter weights (adapter repo path)
                try:
                    model = PeftModel.from_pretrained(base, ADAPTOR_TYPE, device_map="cuda")
                    model = model.cuda().eval()
                    print(f"Applied adapter from {ADAPTOR_TYPE} on base model {base_model}")
                except Exception as e2:
                    print(f"Failed to apply PEFT adapter from {ADAPTOR_TYPE}: {e2}")
                    # As a last resort, expose base model (may not have adapter behavior)
                    model = base.cuda().eval()

            backend = "qwen_vl"
        else:
            print("Loading legacy chat-style model (AutoModel)")
            processor = None
            model = AutoModel.from_pretrained(
                ADAPTOR_TYPE,
                device_map="cuda",
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir=cache,
            ).cuda().eval()
            backend = "chat"

        print("Model loading complete")
        return model, tokenizer, processor, backend
    except Exception as e:
        print(f"Failed to load model and tokenizer: {e}")
        return None, None, None, None


# Convert PDF Page to Image
def pdf_to_image(pdf_bytes, dpi=MODEL_DPI):
    """Convert a single-page PDF to an image."""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = pdf_document.load_page(0)
        pix = page.get_pixmap(dpi = dpi)
        mode = "RGBA" if pix.alpha else "RGB"
        image =  Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        return image
    except Exception as e:
        print(f"Error converting PDF to image: {e}")
        raise ValueError(f"Error converting PDF to image: {e}")

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
def perform_inference(messages, model, tokenizer, processor=None, backend="chat", max_new_tokens=8192):
    """Perform model inference.

    Supports two backends:
    - 'chat': legacy models that implement model.chat(msgs=..., tokenizer=...)
    - 'qwen_vl': Qwen2-VL style models that require a processor and process_vision_info -> model.generate
    """
    try:
        # free cached memory to reduce chance of OOM due to fragmentation
        if torch.cuda.is_available():
            print("Clearing CUDA cache...")
            torch.cuda.empty_cache()

        print(f"Inference backend: {backend}")

        if backend == "chat":
            with torch.no_grad():
                print(f"Inference messages (chat): {messages}")
                response = model.chat(image=None, msgs=messages, tokenizer=tokenizer, max_new_tokens=max_new_tokens)
                print(f"Inference response: {response}")
            return response

        elif backend == "qwen_vl":
            if processor is None or process_vision_info is None:
                raise RuntimeError("Qwen VL backend requested but processor or process_vision_info is unavailable.")

            # Normalize messages to the qwen expected format: list of dicts where content is list of dicts
            def normalize_for_qwen(msgs):
                qwen_msgs = []
                for m in msgs:
                    role = m.get("role", "user")
                    content = m.get("content")
                    # if content is a single string (legacy handler_prompt), wrap
                    if isinstance(content, str):
                        q_content = [{"type": "text", "text": content}]
                    elif isinstance(content, list):
                        q_content = []
                        for part in content:
                            # PIL.Image -> image entry
                            if isinstance(part, Image.Image):
                                q_content.append({"type": "image", "image": part})
                            elif isinstance(part, dict) and part.get("type") in ("image", "text", "video"):
                                q_content.append(part)
                            else:
                                # default to text
                                q_content.append({"type": "text", "text": str(part)})
                    else:
                        # unknown content type
                        q_content = [{"type": "text", "text": str(content)}]

                    qwen_msgs.append({"role": role, "content": q_content})
                return qwen_msgs

            qwen_messages = normalize_for_qwen(messages)
            print(f"Qwen-formatted messages: {qwen_messages}")

            # apply chat template
            text = processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=True)

            # process images/videos
            image_inputs, video_inputs, video_kwargs = process_vision_info(qwen_messages, return_video_kwargs=True)

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            # prepare inputs for model.generate
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **(video_kwargs or {}),
            )

            inputs = inputs.to(model.device)
            print("Running model.generate for qwen_vl backend...")
            generated_ids = model.generate(**inputs, max_new_tokens=min(max_new_tokens, 2048))

            # trim prompt tokens and decode
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(f"Qwen output_texts: {output_texts}")
            # single message -> return first element
            return output_texts[0] if isinstance(output_texts, (list, tuple)) and len(output_texts) > 0 else output_texts

        else:
            raise RuntimeError(f"Unknown backend: {backend}")

    except RuntimeError as e:
        # detect CUDA OOM and provide actionable message
        if "out of memory" in str(e).lower():
            print(f"Inference failed (OOM): {e}")
            # try one more time after clearing cache (best-effort)
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if backend == "chat":
                    response = model.chat(image=None, msgs=messages, tokenizer=tokenizer, max_new_tokens=512)
                    return response
                else:
                    raise RuntimeError("OOM on qwen_vl backend; reduce image size or use a smaller model.")
            except Exception:
                raise RuntimeError(
                    "Inference failed due to CUDA OOM. Reduce max_new_tokens, use model/device offloading or a smaller model."
                )
        print(f"Inference failed: {e}")
        raise RuntimeError(f"Inference failed: {e}")

# Main Request Handler
def run(request):
    """Process incoming requests."""
    try:
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
    
    except Exception as e:
        print(f"Exception during processing: {e}")
        return {"error": f"Exception during processing: {e}"}

def handle_classification(data):
    try:
        payload = PromptPayload(**data)
    except (TypeError, ValidationError) as e:
        raise TypeError(f"Invalid prompt payload: {e}")

    image_b64 = payload.image
    
    if not image_b64:
        raise ValueError("Missing image data.")

    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Error decoding image: {e}")
    
    messages = [{"role": "user", "content": [image, payload.prompt]}]
    response = perform_inference(messages, model, tokenizer, processor=processor, backend=backend)
    try:
        response = json.loads(response)
    except Exception:
        pass
    return {"response": response}

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
    response = perform_inference(prompt, model, tokenizer, processor=processor, backend=backend)

    # this assumes the response is a JSON string, so in the prompt it should be mentioned to return a JSON string
    response = json.loads(response)
    
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
    
    messages = [{"role": "user", "content": payload.prompt}]
    response = perform_inference(messages, model, tokenizer, processor=processor, backend=backend)

    # this assumes the response is a JSON string, so in the prompt it should be mentioned to return a JSON string
    response = json.loads(response)
    return {"response": response}

# # Create a new handler function to handle assistant requests
def handle_assistant_request(data):
    try:
        payload = AssistantPayload(**data)
    except (TypeError, ValidationError)  as e:
        print(f"Invalid prompt payload: {e}")
        return{"error": f"Invalid prompt payload: {e}"}
    
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

    messages = [{"role":"user", "content": images + [payload.prompt]}]
    response = perform_inference(messages, model, tokenizer, processor=processor, backend=backend)
    return {"response": response}



start_time = time.time()
model, tokenizer, processor, backend = load_model_and_tokenizer()
print(f"Model loaded in {time.time() - start_time:.2f} seconds (backend={backend})")


# Initialize and Start RunPod Handler
if __name__ == "__main__":
    print("Initializing RunPod serverless handler")
    runpod.serverless.start({"handler": run})
