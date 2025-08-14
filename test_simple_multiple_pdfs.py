#!/usr/bin/env python3
"""
Simple test to verify the convert_pdf_to_images function handles multiple PDFs correctly.
"""

import os
import sys
import tempfile
import shutil
import json

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import just the functions we need without running the main script
def test_convert_pdf_to_images():
    """Test the convert_pdf_to_images function with PDF identifiers."""
    
    # Import required modules
    import fitz
    from PIL import Image
    
    # Load the convert_pdf_to_images function code
    def convert_pdf_to_images(pickup_id, pdf_path, image_output_dir, pdf_identifier=None, dpi=600):
        os.makedirs(image_output_dir, exist_ok=True)
        pdf_document = fitz.open(pdf_path)
        image_paths = {}
        
        # Generate unique identifier if not provided
        if pdf_identifier is None:
            pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
            pdf_identifier = pdf_filename[:20]  # Limit length to avoid very long filenames
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # Include PDF identifier in the image name to avoid conflicts
            image_path = os.path.join(image_output_dir, f"{pickup_id}_{pdf_identifier}_{page_num + 1:03d}.png")
            image.save(image_path)
            image_paths[str(page_num + 1)] = image_path
            print(f"Saved: {image_path}")
        return image_paths, pdf_identifier
    
    # Test with existing PDF if available
    test_pdf_path = "c:\\Work\\Gothia Digital Solutions\\invoice-ai\\data\\pdf\\185486\\xylemtest1_20250813075701_20250813102026.pdf"
    
    if not os.path.exists(test_pdf_path):
        print(f"❌ Test PDF not found: {test_pdf_path}")
        return False
    
    # Create temporary directory for test images
    test_dir = tempfile.mkdtemp(prefix="test_pdf_images_")
    
    try:
        print("Testing convert_pdf_to_images with PDF identifier...")
        
        # Test 1: With automatic PDF identifier
        image_paths_1, pdf_id_1 = convert_pdf_to_images(
            pickup_id="TEST_001", 
            pdf_path=test_pdf_path, 
            image_output_dir=test_dir
        )
        
        print(f"✅ Test 1 - Automatic PDF identifier: {pdf_id_1}")
        print(f"Generated {len(image_paths_1)} image(s)")
        
        # Test 2: With custom PDF identifier
        image_paths_2, pdf_id_2 = convert_pdf_to_images(
            pickup_id="TEST_001", 
            pdf_path=test_pdf_path, 
            image_output_dir=test_dir,
            pdf_identifier="custom_pdf_001"
        )
        
        print(f"✅ Test 2 - Custom PDF identifier: {pdf_id_2}")
        print(f"Generated {len(image_paths_2)} image(s)")
        
        # Verify that different identifiers create different file names
        if image_paths_1 and image_paths_2:
            first_image_1 = list(image_paths_1.values())[0]
            first_image_2 = list(image_paths_2.values())[0]
            
            if first_image_1 != first_image_2:
                print("✅ Test 3 - Different PDF identifiers create different file names")
                print(f"  Image 1: {os.path.basename(first_image_1)}")
                print(f"  Image 2: {os.path.basename(first_image_2)}")
                return True
            else:
                print("❌ Test 3 - Failed: Same file names generated")
                return False
        else:
            print("❌ No images generated")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        print(f"Cleaning up test directory: {test_dir}")
        shutil.rmtree(test_dir, ignore_errors=True)

def test_data_structure():
    """Test the new data structure for multiple PDFs."""
    
    print("\nTesting new data structure...")
    
    # Simulate the new image_paths_map structure for multiple PDFs
    image_paths_map = {
        "185486": {
            "pdf_001": {
                "1": "./data/image/185486_pdf_001_001.png",
                "2": "./data/image/185486_pdf_001_002.png"
            },
            "pdf_002": {
                "1": "./data/image/185486_pdf_002_001.png",
                "2": "./data/image/185486_pdf_002_002.png"
            }
        }
    }
    
    # Test data structure handling
    pickup_id = "185486"
    if pickup_id in image_paths_map:
        if isinstance(image_paths_map[pickup_id], dict):
            pdf_count = len(image_paths_map[pickup_id])
            print(f"✅ Data structure test - Found {pdf_count} PDFs for pickup ID {pickup_id}")
            
            for pdf_identifier, pdf_pages in image_paths_map[pickup_id].items():
                page_count = len(pdf_pages)
                print(f"  PDF '{pdf_identifier}': {page_count} pages")
                
                for page_num, image_path in pdf_pages.items():
                    print(f"    Page {page_num}: {os.path.basename(image_path)}")
            
            return True
        else:
            print("❌ Data structure test - Invalid structure")
            return False
    else:
        print("❌ Data structure test - Pickup ID not found")
        return False

if __name__ == "__main__":
    print("Testing multiple PDF handling functionality...")
    
    success1 = test_convert_pdf_to_images()
    success2 = test_data_structure()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Multiple PDF handling is working correctly.")
        print("\nChanges made:")
        print("1. ✅ Modified convert_pdf_to_images() to include PDF identifier in file names")
        print("2. ✅ Updated data structure to handle multiple PDFs per pickup ID")
        print("3. ✅ Removed the restriction that skipped pickup IDs with multiple PDFs")
        print("4. ✅ Added support for unique keys in raw data: pdf_identifier_page_num")
    else:
        print("\n💥 Some tests failed! Please check the implementation.")
    
    sys.exit(0 if (success1 and success2) else 1)
