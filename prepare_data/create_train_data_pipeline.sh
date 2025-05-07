#!/bin/bash

# --------- CONFIG ---------
# Blob storage SAS URL
export BLOB_URL="https://saascustomsportalstorage.blob.core.windows.net/pickupfiles?sp=rli&st=2025-01-16T15:04:44Z&se=2026-01-16T23:04:44Z&sv=2022-11-02&sr=c&sig=GmbLCUpv%2F7TsLxvzWS0Y%2BEfYlcHxtxTzgz4hwHJN12c%3D"

# Pickup IDs (comma separated)
export PICKUP_IDS="160573,160228,159777,159776,159775,159749,159308,159288,159287,159283,158291,158269,161099,160729,160198,159759,159391,158975,158361,158029,157699,157365,157269,157189,160633,159675,159674,159565,159230,158943,158942,158156,157697,157696,157695,156047,160175,159676,159611,159179,158060,157480,157284,157279,157260,156836,156803,156232,155727,155694,160875,159950,158192,157779,156489,156294,154920,161103,160728,160226,159430,159024,158386,158040,155935,155609,155017,155016,155013,155009,154916,160876,159951,156298,155231,154990,153092,152381,151357,150634,150549,160043,158246,157223,157061,156450"

# Directory paths
export PDF_OUTPUT_DIR="./data/pdf"
export IMAGE_OUTPUT_DIR="./data/image"
export EXTRACTED_OUTPUT_DIR="./data/extractedData"
export RAW_DATA_OUTPUT="./data/raw_data.json"
export IMAGE_PATH_MAP="./data/image_path_map.json"
export TRAIN_DATA_PATH="./data/train_data.json"
export TEST_DATA_PATH="./data/test_data.json"
export SPLIT_RATIO="0.8" # 80% train, 20% test
export FIELDS_TO_REMOVE="PageNumber,ItemNumber"
export ORDER_ITEM_FIELDS="Description,HsCode,HsCodeExport,Quantity,ArticleNumber,GrossWeight,NetWeight,CountryOfOrigin,NumberOfUnits,TypeOfUnit,PricePerPiece,NetAmount"

# Max workers
export MAX_WORKERS="24"

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
