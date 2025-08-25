# Qwen2.5VL HOI Detection Evaluation Instructions

This guide provides complete instructions for running the enhanced Qwen2.5VL HOI detection evaluation pipeline on both HICO-DET and SWIG-HOI datasets.

## 📋 Prerequisites

### System Requirements
- **GPU**: Minimum 12GB VRAM (24GB+ recommended)
- **OS**: Linux (Ubuntu 18.04+ recommended)
- **Python**: 3.8+
- **CUDA**: 11.8+ with PyTorch CUDA support

### Environment Setup

1. **Create Conda Environment**
```bash
conda create -n qwen_hoi python=3.9
conda activate qwen_hoi
```

2. **Install PyTorch with CUDA**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. **Install Required Packages**
```bash
pip install -r requirements_qwen_hoi.txt
```

4. **Verify Installation**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import transformers; print('Transformers version:', transformers.__version__)"
```

## 📁 Dataset Preparation

### HICO-DET Dataset

1. **Create Data Directory**
```bash
mkdir -p data/hico_20160224_det/{images,annotations}
```

2. **Download Dataset**
```bash
cd data/hico_20160224_det

# Download images
wget https://drive.google.com/uc?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk -O hico_20160224_det.tar.gz
tar -xzf hico_20160224_det.tar.gz

# Download annotations
wget https://drive.google.com/uc?id=1lqmevkw8fjDuTqsOOgzg07Kf6lXhK2rg -O hico_annotations.tar.gz
tar -xzf hico_annotations.tar.gz -C annotations/
```

3. **Verify Structure**
```bash
# Expected structure:
# data/hico_20160224_det/
# ├── images/
# │   ├── test2015/       # Test images (~9,658 images)
# │   └── train2015/      # Train images (~38,118 images)
# └── annotations/
#     ├── test_hico_ann.json
#     └── trainval_hico_ann.json
```

### SWIG-HOI Dataset

1. **Create Data Directory**
```bash
mkdir -p data/swig_hoi/{images_512,annotations}
```

2. **Download Dataset**
```bash
cd data/swig_hoi

# Download images
wget https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip
unzip images_512.zip

# Download annotations
wget https://drive.google.com/uc?id=1GxNP99J0KP6Pwfekij_M1Z0moHziX8QN -O swig_annotations.tar.gz
tar -xzf swig_annotations.tar.gz -C annotations/
```

3. **Verify Structure**
```bash
# Expected structure:
# data/swig_hoi/
# ├── images_512/         # All images (~76,000+ images)
# └── annotations/
#     ├── swig_test_1000.json
#     ├── swig_val_1000.json
#     └── swig_train_1000.json
```

## 🚀 Quick Start

### Option 1: Automated Script (Recommended)

```bash
# HICO-DET evaluation
./run_qwen_hoi_eval.sh --dataset hico --data_root ./data

# SWIG-HOI evaluation
./run_qwen_hoi_eval.sh --dataset swig --data_root ./data
```

### Option 2: Custom Configuration

```bash
# HICO-DET with custom settings
./run_qwen_hoi_eval.sh \
    --dataset hico \
    --data_root ./data \
    --batch_size 8 \
    --dtype fp16 \
    --gpu 0 \
    --score_threshold 0.1

# SWIG-HOI with custom settings
./run_qwen_hoi_eval.sh \
    --dataset swig \
    --data_root ./data \
    --batch_size 6 \
    --dtype fp16 \
    --gpu 1 \
    --max_images 1000
```

## ⚙️ Advanced Usage

### Direct Python Execution

#### HICO-DET Evaluation
```bash
python qwen_hoi_evaluator.py \
    --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --dataset_file hico \
    --data_root ./data \
    --annotation_file ./data/hico_20160224_det/annotations/test_hico_ann.json \
    --output_dir ./results/hico_$(date +%Y%m%d_%H%M%S) \
    --batch_size 8 \
    --dtype fp16 \
    --device auto \
    --score_threshold 0.1 \
    --nms_threshold 0.5 \
    --zero_shot_type rare \
    --ignore_non_interaction \
    --save_predictions
```

#### SWIG-HOI Evaluation
```bash
python qwen_hoi_evaluator.py \
    --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --dataset_file swig \
    --data_root ./data \
    --annotation_file ./data/swig_hoi/annotations/swig_test_1000.json \
    --output_dir ./results/swig_$(date +%Y%m%d_%H%M%S) \
    --batch_size 6 \
    --dtype fp16 \
    --device auto \
    --score_threshold 0.1 \
    --nms_threshold 0.5 \
    --save_predictions
```

## 🔧 Parameter Configuration

### Core Parameters
| Parameter | Description | Default | HICO | SWIG |
|-----------|-------------|---------|------|------|
| `--model_name` | Qwen2.5VL model name | `Qwen/Qwen2.5-VL-7B-Instruct` | ✓ | ✓ |
| `--dataset_file` | Dataset type | - | `hico` | `swig` |
| `--data_root` | Dataset root directory | - | `./data` | `./data` |
| `--batch_size` | Inference batch size | 4 | 4-8 | 4-6 |
| `--dtype` | Model precision | `fp16` | `fp16` | `fp16` |

### Performance Parameters
| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--device` | GPU device | `auto` | Auto-selects best GPU |
| `--score_threshold` | Confidence threshold | 0.1 | Filter low-confidence detections |
| `--nms_threshold` | NMS threshold | 0.5 | Reduce duplicate detections |
| `--max_images` | Max images to evaluate | None | For testing/debugging |

### Dataset-Specific Parameters

#### HICO-DET Only
| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--zero_shot_type` | Zero-shot evaluation type | `rare` | `rare`, `non_rare`, `unseen_verb`, `unseen_object` |
| `--ignore_non_interaction` | Ignore non-interaction categories | `True` | Boolean flag |

## 📊 Understanding Results

### Output Structure
```
results/
├── enhanced_config.json          # Evaluation configuration
├── enhanced_evaluation.log       # Detailed execution log
├── evaluation_results.json       # Final mAP scores
├── performance_stats.json        # Timing and throughput
├── preds.pkl                     # Raw predictions (if --save_predictions)
└── dets.pkl                      # Detection data for analysis
```

### HICO-DET Results Format
```json
{
    "zero_shot_mAP": 0.1738,    # Performance on rare/unseen interactions
    "seen_mAP": 0.2474,         # Performance on training interactions
    "full_mAP": 0.2312          # Overall performance (main metric)
}
```

### SWIG-HOI Results Format
```json
{
    "zero_shot_mAP": 0.1102,    # Unseen interactions (frequency=0)
    "rare_mAP": 0.1674,         # Rare interactions (frequency=1) 
    "nonrare_mAP": 0.2284,      # Common interactions (frequency=2)
    "full_mAP": 0.1674          # Overall performance (main metric)
}
```

### Performance Metrics
```json
{
    "avg_inference_time": 2.34,      # Average seconds per image
    "total_images": 9658,            # Total images processed
    "total_detections": 12450,       # Total HOI detections made
    "avg_detections_per_image": 1.29, # Average detections per image
    "images_per_second": 0.427       # Throughput
}
```

## ⚡ Performance Optimization

### GPU Memory Optimization

#### For 12GB GPUs (e.g., RTX 3080 Ti)
```bash
./run_qwen_hoi_eval.sh --dataset hico --batch_size 4 --dtype fp16
```

#### For 24GB GPUs (e.g., RTX 3090, RTX 4090)
```bash
./run_qwen_hoi_eval.sh --dataset hico --batch_size 8 --dtype fp16
```

#### For 40GB+ GPUs (e.g., A100)
```bash
./run_qwen_hoi_eval.sh --dataset hico --batch_size 12 --dtype fp16
```

### Memory Troubleshooting

1. **CUDA Out of Memory**
```bash
# Reduce batch size
--batch_size 2

# Clear cache before running
python -c "import torch; torch.cuda.empty_cache()"
```

2. **Model Loading Issues**
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/

# Use CPU if necessary (much slower)
--device cpu --dtype fp32
```

## 🧪 Testing and Validation

### Quick Test Run
```bash
# Test with 100 images
./run_qwen_hoi_eval.sh --dataset hico --max_images 100

# Test with specific GPU
./run_qwen_hoi_eval.sh --dataset swig --gpu 1 --max_images 50
```

### Validate Dataset Format
```bash
# The script automatically validates dataset format
# Check logs for validation results:
tail -f results/*/enhanced_evaluation.log | grep -i "validation"
```

## 🐛 Troubleshooting

### Common Issues

1. **Dataset Not Found**
```bash
# Error: Annotation file not found
# Solution: Check data paths and download datasets
ls -la data/hico_20160224_det/annotations/
ls -la data/swig_hoi/annotations/
```

2. **GPU Memory Issues**
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Reduce batch size in script
export BATCH_SIZE=2
./run_qwen_hoi_eval.sh --dataset hico
```

3. **Model Download Issues**
```bash
# Set Hugging Face cache directory
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache

# Pre-download model
python -c "
from transformers import Qwen2VLForConditionalGeneration
model = Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')
"
```

### Performance Expectations

| GPU Model | VRAM | Batch Size | HICO Speed | SWIG Speed |
|-----------|------|------------|------------|------------|
| RTX 3080 Ti | 12GB | 4 | ~2 img/s | ~2 img/s |
| RTX 4090 | 24GB | 8 | ~4 img/s | ~4 img/s |
| A100 40GB | 40GB | 12 | ~6 img/s | ~6 img/s |
| V100 32GB | 32GB | 8 | ~3 img/s | ~3 img/s |

## 📝 Additional Notes

### Model Variants
- **Qwen2.5-VL-7B-Instruct**: Balanced performance (recommended)
- **Qwen2.5-VL-14B-Instruct**: Better accuracy, slower inference
- **Qwen2.5-VL-3B-Instruct**: Faster inference, lower accuracy

### Evaluation Time Estimates
- **HICO-DET** (~9,658 images): 4-8 hours depending on GPU
- **SWIG-HOI** (~76,000 images): 24-48 hours depending on GPU

### File Permissions
```bash
# Ensure scripts are executable
chmod +x run_qwen_hoi_eval.sh

# Check Python script permissions
ls -la qwen_hoi_evaluator.py
```

This evaluation pipeline provides a comprehensive, fair comparison against the INP-CC baseline while leveraging Qwen2.5VL's multimodal capabilities for HOI detection.