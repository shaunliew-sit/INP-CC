#!/bin/bash

# Enhanced Qwen2.5VL HOI Detection Evaluation Script
# Properly tailored for both HICO-DET and SWIG-HOI dataset formats
# Optimized for GPU Linux server execution

set -e

echo "=========================================="
echo "Enhanced Qwen2.5VL HOI Detection Evaluation"
echo "Properly tailored for HICO-DET and SWIG-HOI"
echo "=========================================="

# Configuration
export CUDA_VISIBLE_DEVICES=0  # Set GPU device
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Memory optimization
export OMP_NUM_THREADS=8  # CPU threading

# Default values
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
DATASET="hico"  # or "swig"
DATA_ROOT="./data"
BATCH_SIZE=4
DTYPE="fp16"
SCORE_THRESHOLD=0.1
NMS_THRESHOLD=0.5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --gpu)
            export CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        --max_images)
            MAX_IMAGES="$2"
            shift 2
            ;;
        --score_threshold)
            SCORE_THRESHOLD="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Set dataset-specific paths with proper format handling
if [ "$DATASET" = "hico" ]; then
    ANNOTATION_FILE="$DATA_ROOT/hico_20160224_det/annotations/test_hico_ann.json"
    OUTPUT_DIR="./results/enhanced_qwen_hoi_hico_$(date +%Y%m%d_%H%M%S)"
    ZERO_SHOT_TYPE="rare_first"
    EXPECTED_IMAGE_DIR="$DATA_ROOT/hico_20160224_det/images/test2015"
elif [ "$DATASET" = "swig" ]; then
    ANNOTATION_FILE="$DATA_ROOT/swig_hoi/annotations/swig_test_1000.json"
    OUTPUT_DIR="./results/enhanced_qwen_hoi_swig_$(date +%Y%m%d_%H%M%S)"
    EXPECTED_IMAGE_DIR="$DATA_ROOT/swig_hoi/images_512"
else
    echo "Error: Unknown dataset $DATASET. Use 'hico' or 'swig'."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p ./logs

echo "Enhanced Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET"
echo "  Data Root: $DATA_ROOT"
echo "  Annotation File: $ANNOTATION_FILE"
echo "  Expected Image Directory: $EXPECTED_IMAGE_DIR"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Batch Size: $BATCH_SIZE"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  Data Type: $DTYPE"
echo "  Score Threshold: $SCORE_THRESHOLD"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check CUDA availability
if ! python -c "import torch; print('CUDA available:', torch.cuda.is_available())"; then
    echo "Error: PyTorch CUDA not available"
    exit 1
fi

# Check GPU memory
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $CUDA_VISIBLE_DEVICES)
echo "GPU Memory: ${GPU_MEM}MB"

if [ "$GPU_MEM" -lt 10000 ]; then
    echo "Warning: Low GPU memory detected. Reducing batch size."
    BATCH_SIZE=2
fi

# Check required packages
python -c "
try:
    import transformers
    import torch
    import torchvision
    import PIL
    import numpy
    print('All required packages available')
except ImportError as e:
    print(f'Missing package: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install transformers torch torchvision pillow numpy tqdm qwen-vl-utils
fi

# Enhanced data path validation
echo "Checking enhanced data paths..."

# Check annotation file exists
if [ ! -f "$ANNOTATION_FILE" ]; then
    echo "Error: Annotation file not found: $ANNOTATION_FILE"
    echo ""
    echo "Dataset-specific requirements:"
    if [ "$DATASET" = "hico" ]; then
        echo "  HICO-DET annotation should be at: $DATA_ROOT/hico_20160224_det/annotations/test_hico_ann.json"
        echo "  Download from: https://drive.google.com/open?id=1lqmevkw8fjDuTqsOOgzg07Kf6lXhK2rg"
    else
        echo "  SWIG-HOI annotation should be at: $DATA_ROOT/swig_hoi/annotations/swig_test_1000.json" 
        echo "  Download from: https://drive.google.com/open?id=1GxNP99J0KP6Pwfekij_M1Z0moHziX8QN"
    fi
    exit 1
fi

# Check image directory exists
if [ ! -d "$EXPECTED_IMAGE_DIR" ]; then
    echo "Error: Image directory not found: $EXPECTED_IMAGE_DIR"
    echo ""
    echo "Dataset-specific requirements:"
    if [ "$DATASET" = "hico" ]; then
        echo "  HICO-DET images should be at: $DATA_ROOT/hico_20160224_det/images/test2015/"
        echo "  Download from: https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk"
    else
        echo "  SWIG-HOI images should be at: $DATA_ROOT/swig_hoi/images_512/"
        echo "  Download from: https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip"
    fi
    exit 1
fi

# Validate dataset format by checking a few sample files
echo "Validating dataset format..."
python -c "
import json
import os
from pathlib import Path

dataset = '$DATASET'
annotation_file = '$ANNOTATION_FILE'
image_dir = Path('$EXPECTED_IMAGE_DIR')

# Load annotations
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

if not annotations:
    print('Error: Empty annotation file')
    exit(1)

sample_ann = annotations[0]
print(f'Sample annotation keys: {list(sample_ann.keys())}')

# Check dataset-specific format
if dataset == 'hico':
    required_keys = ['file_name', 'img_id', 'annotations', 'hoi_annotation']
    if not all(key in sample_ann for key in required_keys):
        print(f'Error: HICO annotation missing required keys: {required_keys}')
        exit(1)
    
    # Check if sample image exists
    sample_image = image_dir / sample_ann['file_name']
    if not sample_image.exists():
        print(f'Error: Sample HICO image not found: {sample_image}')
        exit(1)
        
elif dataset == 'swig':
    required_keys = ['file_name', 'img_id', 'box_annotations', 'hoi_annotations']
    if not all(key in sample_ann for key in required_keys):
        print(f'Error: SWIG annotation missing required keys: {required_keys}')
        exit(1)
    
    # Check if sample image exists
    sample_image = image_dir / sample_ann['file_name']
    if not sample_image.exists():
        print(f'Error: Sample SWIG image not found: {sample_image}')
        exit(1)

print(f'{dataset.upper()} dataset format validation passed!')
print(f'Total annotations: {len(annotations)}')
"

if [ $? -ne 0 ]; then
    exit 1
fi

echo "Enhanced prerequisites check passed!"
echo ""

# Prepare enhanced command
CMD="python qwen_hoi_evaluator.py \
    --model_name \"$MODEL_NAME\" \
    --dataset_file $DATASET \
    --data_root \"$DATA_ROOT\" \
    --annotation_file \"$ANNOTATION_FILE\" \
    --output_dir \"$OUTPUT_DIR\" \
    --batch_size $BATCH_SIZE \
    --dtype $DTYPE \
    --device auto \
    --score_threshold $SCORE_THRESHOLD \
    --nms_threshold $NMS_THRESHOLD \
    --num_workers 4 \
    --save_predictions"

# Add dataset-specific arguments
if [ "$DATASET" = "hico" ]; then
    CMD="$CMD --zero_shot_type $ZERO_SHOT_TYPE --ignore_non_interaction"
fi

# Add max_images if specified
if [ ! -z "$MAX_IMAGES" ]; then
    CMD="$CMD --max_images $MAX_IMAGES"
fi

# Log command
echo "Executing enhanced command:"
echo "$CMD"
echo ""

# Save enhanced configuration
cat > "$OUTPUT_DIR/enhanced_config.json" << EOF
{
    "model_name": "$MODEL_NAME",
    "dataset": "$DATASET", 
    "data_root": "$DATA_ROOT",
    "annotation_file": "$ANNOTATION_FILE",
    "expected_image_dir": "$EXPECTED_IMAGE_DIR",
    "batch_size": $BATCH_SIZE,
    "dtype": "$DTYPE",
    "score_threshold": $SCORE_THRESHOLD,
    "nms_threshold": $NMS_THRESHOLD,
    "gpu_device": "$CUDA_VISIBLE_DEVICES",
    "enhancements": {
        "dataset_specific_handling": true,
        "format_validation": true,
        "enhanced_prompts": true,
        "proper_text_mapping": true
    },
    "timestamp": "$(date -Iseconds)"
}
EOF

# Run enhanced evaluation with proper error handling and logging
echo "Starting enhanced evaluation..."
START_TIME=$(date +%s)

# Execute with timeout and logging
timeout 24h bash -c "$CMD" 2>&1 | tee "$OUTPUT_DIR/enhanced_evaluation.log"
EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Enhanced evaluation completed successfully!"
    echo "Duration: ${DURATION} seconds"
    echo ""
    
    # Display enhanced results if available
    if [ -f "$OUTPUT_DIR/evaluation_results.json" ]; then
        echo "Enhanced Results:"
        cat "$OUTPUT_DIR/evaluation_results.json"
        echo ""
    fi
    
    if [ -f "$OUTPUT_DIR/performance_stats.json" ]; then
        echo "Enhanced Performance Statistics:"
        cat "$OUTPUT_DIR/performance_stats.json"
        echo ""
    fi
    
    echo "Enhanced output saved to: $OUTPUT_DIR"
    
elif [ $EXIT_CODE -eq 124 ]; then
    echo "Enhanced evaluation timed out after 24 hours"
    exit 1
else
    echo "Enhanced evaluation failed with exit code: $EXIT_CODE"
    echo "Check logs in: $OUTPUT_DIR/enhanced_evaluation.log"
    exit $EXIT_CODE
fi

echo "=========================================="