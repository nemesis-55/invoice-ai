#!/usr/bin/env python3
"""
Test backward compatibility with existing single PDF structure.
"""

import os
import sys
import tempfile
import shutil
import json

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_backward_compatibility():
    """Test that the existing single PDF structure still works."""
    
    # Import required modules for the test function
    from prepare_data.create_raw_data import create_raw_data, raw_data
    
    print("Testing backward compatibility with single PDF structure...")
    
    # Test with existing format (like current image_path_map.json)
    # {"185486": {"1": "./data/image\\185486_001.png", "2": "./data/image\\185486_002.png"}}
    legacy_image_paths = {
        "185486": {
            "1": "./data/image/185486_001.png",
            "2": "./data/image/185486_002.png",
            "3": "./data/image/185486_003.png",
            "4": "./data/image/185486_004.png"
        }
    }
    
    # Create temporary directories for test
    test_dir = tempfile.mkdtemp(prefix="test_legacy_")
    json_dir = os.path.join(test_dir, "extractedData")
    pickup_json_dir = os.path.join(json_dir, "185486")
    os.makedirs(pickup_json_dir, exist_ok=True)
    
    # Create dummy JSON files to match the expected structure
    for page_num in ["1", "2", "3", "4"]:
        json_file = os.path.join(pickup_json_dir, f"{page_num}_test.json")
        test_data = {
            "Properties": {
                "OrderNumber": f"TEST-{page_num}",
                "InvoiceNumber": f"INV-{page_num}",
                "BuyerName": "Test Buyer",
                "OrderItems": []
            }
        }
        with open(json_file, 'w') as f:
            json.dump(test_data, f)
    
    try:
        # Clear any existing raw_data
        if "185486" in raw_data:
            del raw_data["185486"]
        
        # Test with legacy structure
        create_raw_data("185486", json_dir, legacy_image_paths)
        
        if "185486" in raw_data:
            pages_processed = len(raw_data["185486"])
            print(f"✅ Legacy structure test - Processed {pages_processed} pages")
            
            # Verify the structure matches legacy expectations
            for page_num in ["1", "2", "3", "4"]:
                if page_num in raw_data["185486"]:
                    entry = raw_data["185486"][page_num]
                    if "image_path" in entry and "data" in entry:
                        print(f"  Page {page_num}: ✅ Correct structure")
                    else:
                        print(f"  Page {page_num}: ❌ Missing required fields")
                        return False
                    
                    # Verify no multi-PDF specific fields in legacy mode
                    if "pdf_identifier" in entry or "page_number" in entry:
                        print(f"  Page {page_num}: ❌ Unexpected multi-PDF fields in legacy mode")
                        return False
                else:
                    print(f"  Page {page_num}: ❌ Not found in raw_data")
                    return False
            
            return True
        else:
            print("❌ Legacy structure test - No data found for pickup ID 185486")
            return False
    
    except Exception as e:
        print(f"❌ Legacy structure test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)

def test_mixed_compatibility():
    """Test that both legacy and new structures can coexist."""
    
    from prepare_data.create_raw_data import create_raw_data, raw_data
    
    print("\nTesting mixed compatibility...")
    
    # New multi-PDF structure
    multi_pdf_image_paths = {
        "TEST_001": {
            "pdf_001": {
                "1": "./data/image/TEST_001_pdf_001_001.png",
                "2": "./data/image/TEST_001_pdf_001_002.png"
            },
            "pdf_002": {
                "1": "./data/image/TEST_001_pdf_002_001.png"
            }
        }
    }
    
    # Legacy single-PDF structure
    legacy_image_paths = {
        "TEST_002": {
            "1": "./data/image/TEST_002_001.png",
            "2": "./data/image/TEST_002_002.png"
        }
    }
    
    # Create temporary directories
    test_dir = tempfile.mkdtemp(prefix="test_mixed_")
    json_dir = os.path.join(test_dir, "extractedData")
    
    for pickup_id in ["TEST_001", "TEST_002"]:
        pickup_json_dir = os.path.join(json_dir, pickup_id)
        os.makedirs(pickup_json_dir, exist_ok=True)
        
        # Create dummy JSON files
        for page_num in ["1", "2"]:
            json_file = os.path.join(pickup_json_dir, f"{page_num}_test.json")
            test_data = {
                "Properties": {
                    "OrderNumber": f"{pickup_id}-{page_num}",
                    "InvoiceNumber": f"INV-{pickup_id}-{page_num}",
                    "OrderItems": []
                }
            }
            with open(json_file, 'w') as f:
                json.dump(test_data, f)
    
    try:
        # Clear existing data
        for pickup_id in ["TEST_001", "TEST_002"]:
            if pickup_id in raw_data:
                del raw_data[pickup_id]
        
        # Test multi-PDF structure
        create_raw_data("TEST_001", json_dir, multi_pdf_image_paths)
        
        # Test legacy structure
        create_raw_data("TEST_002", json_dir, legacy_image_paths)
        
        # Verify both structures work
        success = True
        
        # Check multi-PDF structure
        if "TEST_001" in raw_data:
            multi_entries = len(raw_data["TEST_001"])
            print(f"✅ Multi-PDF test - Created {multi_entries} entries")
            
            # Verify multi-PDF entries have the correct format
            for key, entry in raw_data["TEST_001"].items():
                if "pdf_identifier" in entry and "page_number" in entry:
                    print(f"  Entry {key}: ✅ Multi-PDF format")
                else:
                    print(f"  Entry {key}: ❌ Missing multi-PDF fields")
                    success = False
        else:
            print("❌ Multi-PDF test - No data found")
            success = False
        
        # Check legacy structure
        if "TEST_002" in raw_data:
            legacy_entries = len(raw_data["TEST_002"])
            print(f"✅ Legacy test - Created {legacy_entries} entries")
            
            # Verify legacy entries have the correct format
            for key, entry in raw_data["TEST_002"].items():
                if "pdf_identifier" not in entry and "page_number" not in entry:
                    print(f"  Entry {key}: ✅ Legacy format")
                else:
                    print(f"  Entry {key}: ❌ Unexpected multi-PDF fields")
                    success = False
        else:
            print("❌ Legacy test - No data found")
            success = False
        
        return success
    
    except Exception as e:
        print(f"❌ Mixed compatibility test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)

if __name__ == "__main__":
    print("Testing backward compatibility...")
    
    success1 = test_backward_compatibility()
    success2 = test_mixed_compatibility()
    
    if success1 and success2:
        print("\n🎉 All backward compatibility tests passed!")
        print("\nKey improvements:")
        print("1. ✅ Existing single PDF workflows continue to work unchanged")
        print("2. ✅ New multi-PDF support works alongside legacy format")
        print("3. ✅ Automatic detection of data structure format")
        print("4. ✅ No breaking changes to existing functionality")
    else:
        print("\n💥 Some backward compatibility tests failed!")
    
    sys.exit(0 if (success1 and success2) else 1)
