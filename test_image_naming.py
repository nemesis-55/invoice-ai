#!/usr/bin/env python3
import os
from training.handle_pdf import convert_pdf_to_images

def test_image_naming():
    pickup_id = "185691"
    pdf_paths = [
        "185691/xylemtest1_20250813075701_20250814040434.pdf",
        "185691/xylemtest1_20250813075701_test_20250814100714.pdf",
        "12345/invoice_document.pdf",
        "54321/54321_delivery_note.pdf",
        "99999/special-document with spaces.pdf"
    ]
    image_output_dir = "./test_image_output"
    os.makedirs(image_output_dir, exist_ok=True)
    print("Testing image naming pattern:")
    for pdf_path in pdf_paths:
        # We don't need real PDFs for this test, just check the naming logic
        try:
            # Patch fitz.open to avoid opening real files
            import fitz
            class DummyDoc:
                def __len__(self): return 2
                def load_page(self, n):
                    class DummyPage:
                        def get_pixmap(self, dpi=600):
                            class DummyPix:
                                width = 100
                                height = 100
                                alpha = False
                                samples = b'\x00' * (100*100*3)
                            return DummyPix()
                    return DummyPage()
            fitz.open = lambda path: DummyDoc()
            from PIL import Image
            Image.frombytes = lambda mode, size, samples: Image.new("RGB", size)
            convert_pdf_to_images(pickup_id, pdf_path, image_output_dir)
        except Exception as e:
            print(f"Error: {e}")
    print("Test complete. Check ./test_image_output for results.")

if __name__ == "__main__":
    test_image_naming()
