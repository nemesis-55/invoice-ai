# Multiple PDF Support Enhancement

## Overview
Enhanced `create_raw_data.py` to support processing multiple PDFs per pickup ID, removing the previous restriction that skipped pickup IDs with multiple PDFs.

## Changes Made

### 1. Modified `convert_pdf_to_images()` Function
- **Added PDF identifier parameter** to create unique image file names
- **Enhanced naming scheme**: `{pickup_id}_{pdf_identifier}_{page_num:03d}.png`
- **Returns PDF identifier** along with image paths for tracking
- **Prevents file naming conflicts** when multiple PDFs exist for the same pickup ID

**Before**: `185486_001.png`, `185486_002.png`
**After**: `185486_invoice_doc_001.png`, `185486_receipt_doc_001.png`

### 2. Updated `parallel_pdf_to_images()` Function
- **Removed multiple PDF restriction** - no longer skips pickup IDs with multiple PDFs
- **Enhanced data structure** to handle multiple PDFs per pickup ID
- **Processes all PDFs** found in each pickup ID directory
- **Creates nested structure**: `{pickup_id: {pdf_identifier: {page_num: image_path}}}`

### 3. Enhanced `create_raw_data()` Function
- **Automatic structure detection** - determines if data is legacy (single PDF) or new (multi-PDF) format
- **Backward compatibility** - continues to work with existing single PDF workflows
- **Multi-PDF support** - creates unique keys using format `{pdf_identifier}_{page_num}`
- **Additional metadata** - includes `pdf_identifier` and `page_number` fields for multi-PDF entries

### 4. Updated Main Processing Logic
- **Removed skipping logic** for multiple PDF pickup IDs
- **Processes all pickup IDs** regardless of PDF count
- **Maintains existing functionality** for single PDF cases

## Data Structure Changes

### Legacy Format (Single PDF)
```json
{
  "185486": {
    "1": "./data/image/185486_001.png",
    "2": "./data/image/185486_002.png"
  }
}
```

**Raw Data Output**:
```json
{
  "185486": {
    "1": {
      "image_path": "./data/image/185486_001.png",
      "data": {...}
    },
    "2": {
      "image_path": "./data/image/185486_002.png", 
      "data": {...}
    }
  }
}
```

### New Format (Multiple PDFs)
```json
{
  "185486": {
    "invoice_doc": {
      "1": "./data/image/185486_invoice_doc_001.png",
      "2": "./data/image/185486_invoice_doc_002.png"
    },
    "receipt_doc": {
      "1": "./data/image/185486_receipt_doc_001.png"
    }
  }
}
```

**Raw Data Output**:
```json
{
  "185486": {
    "invoice_doc_1": {
      "image_path": "./data/image/185486_invoice_doc_001.png",
      "data": {...},
      "pdf_identifier": "invoice_doc",
      "page_number": "1"
    },
    "invoice_doc_2": {
      "image_path": "./data/image/185486_invoice_doc_002.png",
      "data": {...},
      "pdf_identifier": "invoice_doc", 
      "page_number": "2"
    },
    "receipt_doc_1": {
      "image_path": "./data/image/185486_receipt_doc_001.png",
      "data": {...},
      "pdf_identifier": "receipt_doc",
      "page_number": "1"
    }
  }
}
```

## Benefits

1. **✅ No Data Loss**: Previously skipped pickup IDs with multiple PDFs are now processed
2. **✅ Backward Compatibility**: Existing single PDF workflows continue unchanged
3. **✅ Unique Identification**: Each PDF and page combination has a unique identifier
4. **✅ Conflict Prevention**: Image file naming prevents overwrites
5. **✅ Enhanced Metadata**: Additional tracking information for multi-PDF scenarios
6. **✅ Automatic Detection**: No manual configuration needed - detects format automatically

## Impact

- **Increased Data Coverage**: More pickup IDs will be included in training datasets
- **Better Training Data**: Richer dataset with multiple document types per pickup ID
- **Improved Model Performance**: More diverse training examples
- **Future-Proof**: Extensible design for additional multi-document scenarios

## Testing

The implementation has been thoroughly tested with:
- ✅ Legacy single PDF structures
- ✅ New multi-PDF structures  
- ✅ Mixed environments with both formats
- ✅ Backward compatibility verification
- ✅ File naming conflict prevention
- ✅ Data structure integrity checks

## Notes

- PDF identifiers are automatically generated from filenames (first 20 characters)
- The system gracefully handles both old and new data formats
- No breaking changes to existing API or data contracts
- Memory usage and processing time scale linearly with PDF count
