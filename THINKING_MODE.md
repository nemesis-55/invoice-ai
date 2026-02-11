# Thinking Mode Integration for MiniCPM-V-4.5

## Overview

The invoice-ai handler now supports the **thinking mode** feature available in the MiniCPM-V-4.5 model. This mode enables the model to perform deeper reasoning before generating responses, which can improve accuracy for complex tasks.

## Feature Details

- **Parameter**: `enable_thinking` (boolean)
- **Default**: `false` (disabled by default)
- **Availability**: All API endpoints (INVOICE_EXTRACTION, PROMPT, ASSISTANT, CLASSIFICATION)

## Usage

### API Request Format

Add the `enable_thinking` parameter to your request payload:

#### Invoice Extraction
```json
{
  "input": {
    "action": "INVOICE_EXTRACTION",
    "data": {
      "pdf_data": "base64_encoded_pdf",
      "page_number": "0",
      "enable_thinking": true
    }
  }
}
```

#### Prompt
```json
{
  "input": {
    "action": "PROMPT",
    "data": {
      "prompt": "Your prompt here",
      "enable_thinking": true
    }
  }
}
```

#### Assistant Request
```json
{
  "input": {
    "action": "ASSISTANT",
    "data": {
      "prompt": "Your prompt here",
      "attachments": [...],
      "enable_thinking": true
    }
  }
}
```

#### Classification
```json
{
  "input": {
    "action": "CLASSIFICATION",
    "data": {
      "prompt": "Your prompt here",
      "image": "base64_encoded_image",
      "enable_thinking": true
    }
  }
}
```

## When to Use Thinking Mode

**Enable thinking mode when:**
- Dealing with complex invoice structures
- Requiring high accuracy for critical fields
- Processing documents with ambiguous information
- Need step-by-step reasoning for extraction

**Keep thinking mode disabled (default) when:**
- Processing simple, standardized invoices
- Speed is more important than marginal accuracy gains
- Working with well-structured documents
- Running bulk processing with tight time constraints

## Performance Considerations

- **Response Time**: Thinking mode may increase inference time as the model performs additional reasoning steps
- **Quality**: Can improve accuracy for complex or ambiguous cases
- **Cost**: May use more tokens/compute resources per request

## Implementation Details

### Payload Models

All payload models now include the `enable_thinking` parameter:
- `InvoiceExtractionPayload`
- `PromptPayload`
- `AssistantPayload`

### Handler Function

The `perform_inference()` function has been updated to accept and pass the `enable_thinking` parameter to the model's `chat()` function:

```python
def perform_inference(messages, model, tokenizer, enable_thinking=False):
    response = model.chat(
        image=None, 
        msgs=messages, 
        tokenizer=tokenizer, 
        max_new_tokens=8192,
        enable_thinking=enable_thinking
    )
    return response
```

## Backward Compatibility

The feature is fully backward compatible:
- Existing API calls without `enable_thinking` will default to `false`
- No changes required to existing client code
- Optional parameter can be added when needed

## Example Response Behavior

### Without Thinking Mode (default)
```json
{
  "response": {
    "OrderNumber": "12345",
    "InvoiceNumber": "INV-001",
    ...
  }
}
```

### With Thinking Mode
The response format remains the same, but the model may:
- Take slightly longer to respond
- Provide more accurate field extraction
- Better handle edge cases and ambiguous data

## Testing

To test thinking mode:

```python
import requests
import base64

# Read your PDF
with open("invoice.pdf", "rb") as f:
    pdf_data = base64.b64encode(f.read()).decode('utf-8')

# Test without thinking mode
response_normal = requests.post(endpoint_url, json={
    "input": {
        "action": "INVOICE_EXTRACTION",
        "data": {
            "pdf_data": pdf_data,
            "enable_thinking": False  # or omit this line
        }
    }
})

# Test with thinking mode
response_thinking = requests.post(endpoint_url, json={
    "input": {
        "action": "INVOICE_EXTRACTION",
        "data": {
            "pdf_data": pdf_data,
            "enable_thinking": True
        }
    }
})

# Compare results
print("Normal mode:", response_normal.json())
print("Thinking mode:", response_thinking.json())
```

## Logging

When thinking mode is enabled, the handler logs:
```
Thinking mode enabled: True
```

This helps with debugging and monitoring which requests use the enhanced reasoning capability.

## References

- [MiniCPM-V-4.5 Model Documentation](https://huggingface.co/openbmb/MiniCPM-V-4_5)
- MiniCPM-V-4.5 supports hybrid fast/deep thinking modes for enhanced reasoning
