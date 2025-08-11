#!/usr/bin/env python3
"""
Test script to verify that the updated create_raw_data.py can handle multiple PDFs per pickup ID.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prepare_data.create_raw_data import parallel_pdf_to_images, create_raw_data

def create_test_environment():
    """Create a test environment with multiple PDFs for a single pickup ID."""
    
    # Create temporary directories
    test_dir = tempfile.mkdtemp(prefix="test_multiple_pdfs_")
    pdf_dir = os.path.join(test_dir, "pdf")
    image_dir = os.path.join(test_dir, "images")
    
    # Create test pickup ID directory with multiple PDFs
    test_pickup_id = "TEST_123"
    pickup_pdf_dir = os.path.join(pdf_dir, test_pickup_id)
    os.makedirs(pickup_pdf_dir, exist_ok=True)
    
    # Copy existing PDFs to create multiple test PDFs
    source_pdf_dir = "c:\\Work\\Gothia Digital Solutions\\invoice-ai\\data\\pdf\\185486"
    if os.path.exists(source_pdf_dir):
        source_files = [f for f in os.listdir(source_pdf_dir) if f.endswith('.pdf')]
        if source_files:
            # Copy the same PDF multiple times with different names to simulate multiple PDFs
            source_pdf = os.path.join(source_pdf_dir, source_files[0])
            
            # Create multiple test PDFs
            test_pdf_1 = os.path.join(pickup_pdf_dir, "invoice_001.pdf")
            test_pdf_2 = os.path.join(pickup_pdf_dir, "invoice_002.pdf")
            
            if os.path.exists(source_pdf):
                shutil.copy2(source_pdf, test_pdf_1)
                shutil.copy2(source_pdf, test_pdf_2)
                print(f"Created test PDFs: {test_pdf_1}, {test_pdf_2}")
            else:
                print(f"Source PDF not found: {source_pdf}")
                return None
        else:
            print(f"No PDF files found in {source_pdf_dir}")
            return None
    else:
        print(f"Source directory not found: {source_pdf_dir}")
        return None
    
    return test_dir, pdf_dir, image_dir, test_pickup_id

def test_multiple_pdfs():
    """Test the multiple PDF handling functionality."""
    
    print("Creating test environment...")
    result = create_test_environment()
    if result is None:
        print("Failed to create test environment")
        return False
    
    test_dir, pdf_dir, image_dir, test_pickup_id = result
    
    try:
        print(f"Testing with pickup ID: {test_pickup_id}")
        print(f"PDF directory: {pdf_dir}")
        print(f"Image directory: {image_dir}")
        
        # Test the parallel_pdf_to_images function
        output_json_path = os.path.join(test_dir, "image_path_map.json")
        
        image_paths_map, skipped_pickup_ids = parallel_pdf_to_images(
            pickup_ids=[test_pickup_id],
            pdf_output_dir=pdf_dir,
            image_output_dir=image_dir,
            output_json_path=output_json_path,
            max_workers=2
        )
        
        print(f"Skipped pickup IDs: {skipped_pickup_ids}")
        print(f"Image paths map: {image_paths_map}")
        
        # Verify that no pickup IDs were skipped
        if test_pickup_id in skipped_pickup_ids:
            print("❌ FAILED: Test pickup ID was skipped")
            return False
        
        # Verify that multiple PDFs were processed
        if test_pickup_id in image_paths_map:
            pdf_count = len(image_paths_map[test_pickup_id])
            print(f"✅ SUCCESS: Processed {pdf_count} PDF(s) for pickup ID {test_pickup_id}")
            
            # Print details of processed PDFs
            for pdf_identifier, pages in image_paths_map[test_pickup_id].items():
                print(f"  PDF '{pdf_identifier}': {len(pages)} pages")
                for page_num, image_path in pages.items():
                    print(f"    Page {page_num}: {image_path}")
            
            return True
        else:
            print("❌ FAILED: No image paths found for test pickup ID")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        print(f"Cleaning up test directory: {test_dir}")
        shutil.rmtree(test_dir, ignore_errors=True)

if __name__ == "__main__":
    print("Testing multiple PDF handling...")
    success = test_multiple_pdfs()
    if success:
        print("\n🎉 All tests passed! Multiple PDF handling is working correctly.")
    else:
        print("\n💥 Tests failed! Please check the implementation.")
    
    sys.exit(0 if success else 1)
