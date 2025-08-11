#!/usr/bin/env python3

import os

def generate_pdf_identifier(pdf_path, pickup_id):
    """Generate a simple PDF identifier using the full filename with a prefix."""
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Remove the pickup_id prefix if it exists to avoid redundancy
    if pdf_filename.startswith(pickup_id):
        # Remove pickup_id and any following underscore or dash
        identifier = pdf_filename[len(pickup_id):].lstrip('_-')
    else:
        identifier = pdf_filename
    
    # Clean the identifier: replace spaces and dashes with underscores, remove special chars
    identifier = identifier.replace(' ', '_').replace('-', '_')
    identifier = ''.join(c for c in identifier if c.isalnum() or c == '_')
    
    # Remove leading/trailing underscores
    identifier = identifier.strip('_')
    
    # If empty after cleaning, use the original filename
    if not identifier:
        identifier = pdf_filename.replace(' ', '_').replace('-', '_')
        identifier = ''.join(c for c in identifier if c.isalnum() or c == '_').strip('_')
    
    # Add a simple prefix to make it clear this is a PDF identifier
    return f"pdf_{identifier}"

def test_pdf_identifiers():
    # Test cases with real examples
    test_cases = [
        ("185691/xylemtest1_20250813075701_20250814040434.pdf", "185691"),
        ("185691/xylemtest1_20250813075701_test_20250814100714.pdf", "185691"),
        ("12345/invoice_document.pdf", "12345"),
        ("12345/receipt_final.pdf", "12345"),
        ("54321/54321_delivery_note.pdf", "54321"),
        ("99999/special-document with spaces.pdf", "99999"),
    ]
    
    print("Testing simplified PDF identifier generation:")
    print("=" * 60)
    
    for pdf_path, pickup_id in test_cases:
        identifier = generate_pdf_identifier(pdf_path, pickup_id)
        filename = os.path.basename(pdf_path)
        print(f"File: {filename}")
        print(f"Pickup ID: {pickup_id}")
        print(f"Generated ID: {identifier}")
        print("-" * 40)

if __name__ == "__main__":
    test_pdf_identifiers()
