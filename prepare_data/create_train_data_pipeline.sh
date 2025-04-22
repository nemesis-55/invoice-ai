#!/bin/bash

# --------- CONFIG ---------
# Blob storage SAS URL
export BLOB_URL="https://saascustomsportalstorage.blob.core.windows.net/pickupfiles?sp=rli&st=2025-01-16T15:04:44Z&se=2026-01-16T23:04:44Z&sv=2022-11-02&sr=c&sig=GmbLCUpv%2F7TsLxvzWS0Y%2BEfYlcHxtxTzgz4hwHJN12c%3D"

# Pickup IDs (comma separated)
export PICKUP_IDS="143419"

# Directory paths
export PDF_OUTPUT_DIR="./data/pdf"
export IMAGE_OUTPUT_DIR="./data/image"
export EXTRACTED_OUTPUT_DIR="./data/extractedData"
export RAW_DATA_OUTPUT="./data/raw_data.json"
export IMAGE_PATH_MAP="./data/image_path_map.json"
export TRAIN_DATA_PATH="./data/train_data.json"
export TEST_DATA_PATH="./data/test_data.json"
export SPLIT_RATIO="0.25"
export FIELDS_TO_REMOVE="PageNumber,ItemNumber"
export ORDER_ITEM_FIELDS="Description,HsCode,HsCodeExport,Quantity,ArticleNumber,GrossWeight,NetWeight,CountryOfOrigin,NumberOfUnits,TypeOfUnit,PricePerPiece,NetAmount"

# Max workers
export MAX_WORKERS="100"

# Python scripts
RAW_DATA_SCRIPT="./prepare_data/create_raw_data.py"
TRAINING_SCRIPT="./prepare_data/create_training_data.py"

# --------- EXECUTION ---------
echo "Step 1: Running raw data extraction script..."
python "$RAW_DATA_SCRIPT"

if [ $? -ne 0 ]; then
  echo "Raw data creation failed. Exiting."
  exit 1
fi

echo "Step 2: Running training data creation script..."
python "$TRAINING_SCRIPT"

if [ $? -ne 0 ]; then
  echo "Training data creation failed. Exiting."
  exit 1
fi

echo "✅ Pipeline completed successfully."
