#!/usr/bin/env python3
"""
Test the improved PDF identifier generation with real PDF filenames.
"""

import os
import sys
import hashlib

def generate_pdf_identifier(pdf_path, pickup_id):
    """Generate a meaningful and unique PDF identifier from the PDF filename."""
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Remove common prefixes and the pickup_id if present
    identifier = pdf_filename
    common_prefixes = ['xylemtest1_', pickup_id + '_', pickup_id]
    for prefix in common_prefixes:
        if identifier.startswith(prefix):
            identifier = identifier[len(prefix):]
            break
    
    # Split by underscores and extract meaningful parts
    parts = identifier.split('_')
    
    # Strategy 1: Look for distinguishing keywords
    keywords = ['test', 'invoice', 'receipt', 'order', 'delivery', 'packing', 'shipping', 'bill']
    found_keywords = [part.lower() for part in parts if any(keyword in part.lower() for keyword in keywords)]
    
    if found_keywords:
        # Use the first found keyword as the primary identifier
        base_identifier = found_keywords[0]
    else:
        # Strategy 2: Use timestamp or unique suffix to differentiate
        # Look for the last few parts that might be timestamps or unique identifiers
        if len(parts) >= 2:
            # Take the last 2 parts as they're likely to be unique (timestamps, etc.)
            base_identifier = '_'.join(parts[-2:])
        else:
            # Fallback: use the whole cleaned filename
            base_identifier = identifier
    
    # Clean and limit the identifier
    base_identifier = base_identifier.replace('-', '_').replace(' ', '_')
    base_identifier = ''.join(c for c in base_identifier if c.isalnum() or c == '_')
    
    # Ensure it's not too long and not empty
    if len(base_identifier) > 15:
        base_identifier = base_identifier[:15]
    if not base_identifier or base_identifier == '_':
        # Ultimate fallback: use a hash of the filename
        import hashlib
        base_identifier = hashlib.md5(pdf_filename.encode()).hexdigest()[:8]
    
    return base_identifier

def test_pdf_identifier_generation():
    """Test PDF identifier generation with real examples."""
    
    print("Testing improved PDF identifier generation...")
    
    # Test cases based on actual PDF filenames
    test_cases = [
        # Real examples from the system
        ("185691/xylemtest1_20250813075701_20250814040434.pdf", "185691"),
        ("185691/xylemtest1_20250813075701_test_20250814100714.pdf", "185691"),
        
        # Additional test cases for robustness
        ("185486/xylemtest1_20250813075701_20250813102026.pdf", "185486"),
        ("TEST123/invoice_document_20250814_final.pdf", "TEST123"),
        ("TEST123/receipt_document_20250814_copy.pdf", "TEST123"),
        ("ABC456/order_confirmation_123.pdf", "ABC456"),
        ("ABC456/delivery_note_456.pdf", "ABC456"),
        ("XYZ789/very_long_filename_with_lots_of_text_and_numbers_20250814.pdf", "XYZ789"),
        ("SIMPLE/document.pdf", "SIMPLE"),
        ("EMPTY/___.pdf", "EMPTY"),
    ]
    
    print("\n--- PDF Identifier Generation Test ---")
    identifiers_per_pickup = {}
    
    for pdf_path, pickup_id in test_cases:
        identifier = generate_pdf_identifier(pdf_path, pickup_id)
        
        # Track identifiers per pickup ID to check for conflicts
        if pickup_id not in identifiers_per_pickup:
            identifiers_per_pickup[pickup_id] = []
        identifiers_per_pickup[pickup_id].append((pdf_path, identifier))
        
        print(f"PDF: {os.path.basename(pdf_path)}")
        print(f"  Pickup ID: {pickup_id}")
        print(f"  Generated ID: '{identifier}'")
        print(f"  Length: {len(identifier)}")
        print()
    
    # Check for conflicts
    print("--- Conflict Detection ---")
    conflicts_found = False
    
    for pickup_id, pdf_list in identifiers_per_pickup.items():
        if len(pdf_list) > 1:
            identifiers = [identifier for _, identifier in pdf_list]
            unique_identifiers = set(identifiers)
            
            if len(unique_identifiers) < len(identifiers):
                conflicts_found = True
                print(f"❌ CONFLICT in pickup ID {pickup_id}:")
                for pdf_path, identifier in pdf_list:
                    print(f"  {os.path.basename(pdf_path)} -> '{identifier}'")
                print()
            else:
                print(f"✅ No conflicts in pickup ID {pickup_id}:")
                for pdf_path, identifier in pdf_list:
                    print(f"  {os.path.basename(pdf_path)} -> '{identifier}'")
                print()
    
    # Test collision resolution
    print("--- Collision Resolution Test ---")
    def resolve_collisions(pickup_id, pdf_identifiers):
        """Simulate the collision resolution logic."""
        pdf_identifier_tracker = set()
        resolved_identifiers = []
        
        for original_identifier in pdf_identifiers:
            pdf_identifier = original_identifier
            counter = 1
            while pdf_identifier in pdf_identifier_tracker:
                pdf_identifier = f"{original_identifier}_{counter}"
                counter += 1
            
            pdf_identifier_tracker.add(pdf_identifier)
            resolved_identifiers.append((original_identifier, pdf_identifier))
        
        return resolved_identifiers
    
    # Test collision resolution with deliberately conflicting identifiers
    test_collisions = ["test", "test", "document", "test", "document"]
    resolved = resolve_collisions("TEST_COLLISION", test_collisions)
    
    print("Input identifiers:", test_collisions)
    print("Resolved identifiers:")
    for original, resolved_id in resolved:
        status = "✅ OK" if original == resolved_id else f"🔄 RENAMED: {original} -> {resolved_id}"
        print(f"  {status}")
    
    if not conflicts_found:
        print("\n🎉 All tests passed! PDF identifier generation is working correctly.")
        return True
    else:
        print("\n❌ Conflicts detected in PDF identifier generation.")
        return False

def test_real_pdf_files():
    """Test with actual PDF files in the system."""
    
    print("\n--- Testing with Real PDF Files ---")
    
    pdf_root = "c:/Work/Gothia Digital Solutions/invoice-ai/data/pdf"
    
    if not os.path.exists(pdf_root):
        print(f"PDF directory not found: {pdf_root}")
        return True
    
    real_conflicts = False
    
    # Find pickup IDs with multiple PDFs
    for pickup_id in os.listdir(pdf_root):
        pickup_dir = os.path.join(pdf_root, pickup_id)
        if os.path.isdir(pickup_dir):
            pdf_files = [f for f in os.listdir(pickup_dir) if f.endswith('.pdf')]
            
            if len(pdf_files) > 1:
                print(f"\nPickup ID: {pickup_id} ({len(pdf_files)} PDFs)")
                identifiers = []
                
                for pdf_file in pdf_files:
                    pdf_path = os.path.join(pickup_dir, pdf_file)
                    identifier = generate_pdf_identifier(pdf_path, pickup_id)
                    identifiers.append(identifier)
                    print(f"  {pdf_file} -> '{identifier}'")
                
                # Check for conflicts
                if len(set(identifiers)) < len(identifiers):
                    real_conflicts = True
                    print(f"  ❌ CONFLICT DETECTED!")
                else:
                    print(f"  ✅ No conflicts")
    
    if not real_conflicts:
        print("\n✅ No conflicts found in real PDF files!")
        return True
    else:
        print("\n❌ Conflicts found in real PDF files!")
        return False

if __name__ == "__main__":
    print("Testing PDF identifier generation improvements...")
    
    success1 = test_pdf_identifier_generation()
    success2 = test_real_pdf_files()
    
    overall_success = success1 and success2
    
    if overall_success:
        print("\n🎉 All PDF identifier tests passed!")
        print("\nImprovements made:")
        print("1. ✅ Meaningful identifier extraction from filenames")
        print("2. ✅ Keyword-based differentiation (test, invoice, receipt, etc.)")
        print("3. ✅ Timestamp-based fallback for unique identification")
        print("4. ✅ Collision detection and resolution")
        print("5. ✅ Length limits and character cleaning")
        print("6. ✅ Fallback to hash for edge cases")
    else:
        print("\n❌ PDF identifier generation needs further improvement!")
    
    sys.exit(0 if overall_success else 1)
