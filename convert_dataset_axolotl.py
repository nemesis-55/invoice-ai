#!/usr/bin/env python3
"""
Dataset conversion script to adapt existing MiniCPM dataset format to Axolotl format.
Converts multimodal invoice data to chat_template format for Axolotl training.
"""

import json
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import base64
from PIL import Image
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetConverter:
    """Converts existing dataset format to Axolotl chat_template format"""
    
    def __init__(self, input_path: str, output_path: str, image_base_path: str = None):
        self.input_path = input_path
        self.output_path = output_path
        self.image_base_path = image_base_path or "./data/images"
        
    def load_original_dataset(self) -> List[Dict[str, Any]]:
        """Load the original dataset format"""
        logger.info(f"Loading dataset from {self.input_path}")
        with open(self.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples")
        return data
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string for embedding in conversation"""
        try:
            if not os.path.isabs(image_path):
                full_path = os.path.join(self.image_base_path, image_path)
            else:
                full_path = image_path
                
            with open(full_path, 'rb') as img_file:
                img_data = img_file.read()
                
            # Optionally resize large images to save memory
            img = Image.open(io.BytesIO(img_data))
            if img.size[0] > 2048 or img.size[1] > 2048:
                img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=95)
                img_data = buffer.getvalue()
                
            return base64.b64encode(img_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None
    
    def convert_conversation_to_axolotl_format(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a single sample from original format to Axolotl chat_template format.
        
        Original format:
        {
            "image": "path/to/image.jpg" or {"image_00": "path1", "image_01": "path2"},
            "conversations": [
                {"role": "user", "content": "Extract data from this invoice..."},
                {"role": "assistant", "content": "{'invoice_number': '123', ...}"}
            ]
        }
        
        Axolotl format:
        {
            "conversations": [
                {"from": "human", "value": "Analyze this invoice image and extract the required data..."},
                {"from": "gpt", "value": "{'invoice_number': '123', ...}"}
            ],
            "images": ["base64_image_data"]
        }
        """
        
        # Handle image(s)
        images = []
        image_placeholders = []
        
        if isinstance(sample["image"], str):
            # Single image
            image_b64 = self.encode_image_to_base64(sample["image"])
            if image_b64:
                images.append(image_b64)
                image_placeholders.append("<image>")
        elif isinstance(sample["image"], dict):
            # Multiple images
            for img_name, img_path in sample["image"].items():
                image_b64 = self.encode_image_to_base64(img_path)
                if image_b64:
                    images.append(image_b64)
                    image_placeholders.append(f"<{img_name}>")
        
        if not images:
            logger.warning("No valid images found for sample, skipping")
            return None
        
        # Convert conversations
        axolotl_conversations = []
        for conv in sample["conversations"]:
            role_mapping = {
                "user": "human",
                "assistant": "gpt"
            }
            
            axolotl_role = role_mapping.get(conv["role"], conv["role"])
            content = conv["content"]
            
            # For the first user message, ensure image placeholder is included
            if axolotl_role == "human" and len(axolotl_conversations) == 0:
                # Check if image placeholder already exists
                has_image_placeholder = any(placeholder in content for placeholder in image_placeholders)
                if not has_image_placeholder:
                    # Add image placeholder at the beginning
                    content = " ".join(image_placeholders) + "\n" + content
                else:
                    # Replace existing placeholders with proper format
                    for i, placeholder in enumerate(image_placeholders):
                        old_placeholder = f"<image_{i:02d}>" if len(image_placeholders) > 1 else "<image>"
                        content = content.replace(old_placeholder, placeholder)
            
            axolotl_conversations.append({
                "from": axolotl_role,
                "value": content
            })
        
        return {
            "conversations": axolotl_conversations,
            "images": images
        }
    
    def convert_dataset(self) -> None:
        """Convert entire dataset and save to output path"""
        original_data = self.load_original_dataset()
        converted_data = []
        skipped_samples = 0
        
        for i, sample in enumerate(original_data):
            logger.info(f"Converting sample {i+1}/{len(original_data)}")
            
            try:
                converted_sample = self.convert_conversation_to_axolotl_format(sample)
                if converted_sample:
                    converted_data.append(converted_sample)
                else:
                    skipped_samples += 1
                    logger.warning(f"Skipped sample {i+1}")
            except Exception as e:
                logger.error(f"Error converting sample {i+1}: {e}")
                skipped_samples += 1
        
        # Save converted dataset
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Conversion complete!")
        logger.info(f"Original samples: {len(original_data)}")
        logger.info(f"Converted samples: {len(converted_data)}")
        logger.info(f"Skipped samples: {skipped_samples}")
        logger.info(f"Output saved to: {self.output_path}")
        
        # Save a sample for inspection
        if converted_data:
            sample_output = self.output_path.replace('.json', '_sample.json')
            with open(sample_output, 'w', encoding='utf-8') as f:
                # Save first sample without base64 images for readability
                sample_copy = converted_data[0].copy()
                sample_copy['images'] = [f"<base64_image_{i}>" for i in range(len(sample_copy['images']))]
                json.dump(sample_copy, f, indent=2, ensure_ascii=False)
            logger.info(f"Sample output (without base64 images) saved to: {sample_output}")

def main():
    parser = argparse.ArgumentParser(description="Convert MiniCPM dataset to Axolotl format")
    parser.add_argument("--input", required=True, help="Path to input dataset JSON file")
    parser.add_argument("--output", required=True, help="Path to output dataset JSON file")
    parser.add_argument("--image_base_path", default="./data/images", 
                       help="Base path for resolving relative image paths")
    parser.add_argument("--split_ratio", type=float, default=0.9,
                       help="Train/validation split ratio (default: 0.9)")
    
    args = parser.parse_args()
    
    # Convert main dataset
    converter = DatasetConverter(args.input, args.output, args.image_base_path)
    converter.convert_dataset()
    
    # Create train/validation split if requested
    if args.split_ratio < 1.0:
        logger.info(f"Creating train/validation split with ratio {args.split_ratio}")
        
        with open(args.output, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        split_idx = int(len(all_data) * args.split_ratio)
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        # Save train set
        train_output = args.output.replace('.json', '_train.json')
        with open(train_output, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        # Save validation set
        val_output = args.output.replace('.json', '_val.json')
        with open(val_output, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Train set ({len(train_data)} samples) saved to: {train_output}")
        logger.info(f"Validation set ({len(val_data)} samples) saved to: {val_output}")

if __name__ == "__main__":
    main()