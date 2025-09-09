#!/bin/bash

# --------- CONFIG ---------
export BLOB_URL="https://saascustomsportalstorage.blob.core.windows.net/pickupfiles?sp=rli&st=2025-01-16T15:04:44Z&se=2026-01-16T23:04:44Z&sv=2022-11-02&sr=c&sig=GmbLCUpv%2F7TsLxvzWS0Y%2BEfYlcHxtxTzgz4hwHJN12c%3D"

# Pickup IDs (comma separated)
export PICKUP_IDS="188274,187757,185758,189255,189592,190295,193089,193091,193092,193102,193103,193105,189901,189906,171271,174021"

# Directory paths
export PDF_OUTPUT_DIR="./classification_training/pdf"
export IMAGE_OUTPUT_DIR="./classification_training/image"
export RAW_DATA_OUTPUT="./classification_training/raw_data.json"
export TRAIN_DATA_PATH="./classification_training/train_data.json"
export TEST_DATA_PATH="./classification_training/test_data.json"
export SPLIT_RATIO="0.8" # 80% train, 20% test

# Max workers
export MAX_WORKERS="24"

# Python scripts
RAW_DATA_SCRIPT="./classification_training_workflow/create_raw_data.py"
TRAINING_SCRIPT="./classification_training_workflow/create_training_data.py"

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

echo "✅ Classification training pipeline completed successfully."
