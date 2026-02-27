# invoice-ai

## Organization

This repository belongs to **[Arindam2002](https://github.com/Arindam2002)** on GitHub.

## Overview

`invoice-ai` is a serverless AI-powered invoice data extraction service built on [RunPod](https://www.runpod.io/). It uses a fine-tuned multimodal language model (`Zorro123444/invoice_extracter_xylem2.1.1`) to parse invoice images and OCR text, returning structured JSON with fields such as order number, invoice number, buyer/seller details, line items, and shipping information.

## Components

| File | Description |
|------|-------------|
| `handler.py` | RunPod serverless handler – loads the model and processes inference requests |
| `curl.py` | Helper script for sending test requests to the RunPod endpoint |
| `Dockerfile` | Container image definition for the serverless worker |
| `requirements.txt` | Python dependencies |

## Usage

The handler accepts a JSON payload with:
- `image` – Base64-encoded invoice image
- `ocr_data` – OCR text extracted from the invoice page

It returns a JSON object containing the extracted invoice fields.
