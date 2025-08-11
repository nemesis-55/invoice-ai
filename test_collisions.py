#!/usr/bin/env python3

def test_collision_resolution():
    """Test how collision resolution works with the simplified identifiers"""
    # Simulate what happens in the collision resolution code
    identifiers = [
        "pdf_xylemtest1_20250813075701_20250814040434",
        "pdf_xylemtest1_20250813075701_test_20250814100714", 
        "pdf_invoice_document",
        "pdf_invoice_document",  # Duplicate to test collision
        "pdf_receipt_final"
    ]
    
    pickup_id = "185691"
    used_identifiers = set()
    final_identifiers = []
    
    print(f"Testing collision resolution for pickup ID: {pickup_id}")
    print("=" * 60)
    
    for original_identifier in identifiers:
        identifier = original_identifier
        counter = 1
        
        # Handle collisions
        while identifier in used_identifiers:
            identifier = f"{original_identifier}_{counter}"
            counter += 1
        
        used_identifiers.add(identifier)
        final_identifiers.append(identifier)
        
        if identifier != original_identifier:
            print(f"Collision resolved: {original_identifier} -> {identifier}")
        else:
            print(f"No collision: {identifier}")
    
    print("\nFinal unique identifiers:")
    for i, identifier in enumerate(final_identifiers, 1):
        print(f"{i}. {identifier}")

if __name__ == "__main__":
    test_collision_resolution()
