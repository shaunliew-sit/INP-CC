# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is INP-CC (Interaction-aware Prompting with Concept Calibration), an open-vocabulary Human-Object Interaction (HOI) detection system built on PyTorch and CLIP. The project implements a novel approach for detecting interactions between humans and objects while generalizing to novel interaction classes beyond the training set.

## Core Architecture

The codebase follows a transformer-based architecture with several key components:

### Main Components
- **main.py**: Entry point with distributed training setup, model building, and training/evaluation loops
- **engine.py**: Training and evaluation functions including `train_one_epoch()` and `evaluate()`
- **models/model.py**: Core model implementation with HOI-specific attention blocks and CLIP integration
- **arguments.py**: Comprehensive argument parser with all model, training, and dataset configurations

### Key Modules
- **models/**: Contains the core model architecture
  - `model.py`: Main HOI detection model with interaction-aware prompting
  - `criterion.py`: Loss functions and matching criteria
  - `transformer.py`: Custom transformer decoder layers
  - `matcher.py`: Bipartite matching for object detection
- **datasets/**: Data loading and evaluation
  - `hico.py`, `swig.py`: Dataset implementations for HICO-DET and SWIG-HOI
  - `hico_evaluator.py`, `swig_evaluator.py`: Evaluation metrics
- **clip/**: Modified CLIP implementation for HOI detection
- **utils/**: Utility functions for training, scheduling, and visualization

## Training Commands

### HICO-DET Training
```bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port 3996 --use_env main.py \
    --batch_size 32 \
    --output_dir ckpts/hico \
    --epochs 80 \
    --lr 1e-4 --min-lr 1e-7 \
    --hoi_token_length 20 \
    --enable_dec \
    --dataset_file hico --multi_scale false --use_aux_text true \
    --enable_focal_loss --description_file_path hico_hoi_descriptions.json \
    --VPT_length 4 --img_scene_num 8 \
    --instruction_embedding_file InstructEmbed/1108/hico_embeddings_1108.pkl
```

### SWIG-HOI Training
```bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port 5786 --use_env main.py \
    --batch_size 64 \
    --output_dir ckpts/swig \
    --epochs 80 \
    --lr 1e-4 --min-lr 1e-7 \
    --hoi_token_length 30 \
    --enable_dec \
    --dataset_file swig \
    --enable_focal_loss --description_file_path swig_hoi_descriptions_6bodyparts.json \
    --VPT_length 4 --img_scene_num 128 --additional_hoi_num 10 --add_hoi_strategy hard \
    --cluster_assignmen_file InstructEmbed/1108/swig_cluster_assignment_64.npy \
    --use_aux_text true --instruction_embedding_file InstructEmbed/1108/swig_embeddings_1108.pkl
```

## Evaluation Commands

### HICO-DET Evaluation
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 3996 --use_env main.py \
    --batch_size 32 \
    --output_dir ckpts/hico \
    --epochs 80 \
    --lr 1e-4 --min-lr 1e-7 \
    --hoi_token_length 20 \
    --enable_dec \
    --dataset_file hico --multi_scale false --use_aux_text true \
    --enable_focal_loss --description_file_path hico_hoi_descriptions.json \
    --VPT_length 4 --img_scene_num 8 \
    --instruction_embedding_file InstructEmbed/1108/hico_embeddings_1108.pkl \
    --eval --pretrained [path to ckpt]
```

### SWIG-HOI Evaluation
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 5786 --use_env main.py \
    --batch_size 64 \
    --output_dir ckpts/swig \
    --epochs 80 \
    --lr 1e-4 --min-lr 1e-7 \
    --hoi_token_length 30 \
    --enable_dec \
    --dataset_file swig \
    --enable_focal_loss --description_file_path swig_hoi_descriptions_6bodyparts.json \
    --VPT_length 4 --img_scene_num 128 --additional_hoi_num 10 --add_hoi_strategy hard \
    --cluster_assignmen_file InstructEmbed/1108/swig_cluster_assignment_64.npy \
    --use_aux_text true --instruction_embedding_file InstructEmbed/1108/swig_embeddings_1108.pkl \
    --eval --pretrained [path to ckpt]
```

## Dependencies

The project requires:
- PyTorch with CUDA support
- torchvision
- Standard ML libraries: numpy, Pillow, matplotlib
- Text processing: ftfy, regex, tqdm

Install with:
```bash
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install ftfy regex tqdm numpy Pillow matplotlib
```

## Key Configuration Files

- `hico_hoi_descriptions.json`: HOI descriptions for HICO-DET dataset
- `swig_hoi_descriptions_6bodyparts.json`: HOI descriptions for SWIG-HOI dataset
- `InstructEmbed/1108/`: Pre-processed instruction embeddings
- `huggingface-checkpoint/`: Model checkpoints and image embeddings

## Dataset Structure

The project expects datasets in the `data/` directory:
- HICO-DET: `data/hico_20160224_det/`
- SWIG-HOI: `data/swig_hoi/`

## Important Implementation Notes

- Uses distributed training with PyTorch's DistributedDataParallel
- Implements custom attention mechanisms for HOI detection
- Integrates pre-trained CLIP models with modifications for region-level interaction detection
- Supports both HICO-DET and SWIG-HOI benchmarks with different configurations
- Uses cluster assignments and instruction embeddings for improved performance

## Qwen2.5VL HOI Evaluation Implementation

A complete Qwen2.5VL-based HOI detection evaluation pipeline has been implemented to compare against INP-CC using the same evaluation metrics:

### Key Files
- **qwen_hoi_evaluator.py**: Main Qwen2.5VL HOI detection evaluation script with dataset-specific handling
- **run_qwen_hoi_eval.sh**: Automated runner script with GPU optimization and validation
- **HOI_Evaluation_Framework.md**: Complete technical documentation of evaluation metrics and pipeline
- **HOI_Evaluation_Instructions.md**: User guide with setup instructions and troubleshooting

### Current Status
- ✅ Implemented enhanced Qwen2.5VL HOI detection with proper dataset format handling
- ✅ Fixed model architecture issues (using `Qwen2_5_VLForConditionalGeneration`)
- ✅ Resolved FlashAttention2 installation and zero-shot type configuration (`rare_first`)
- ✅ Successfully loads 10 test images and processes through evaluation pipeline

### Usage
```bash
# Quick test with 10 images
bash ./run_qwen_hoi_eval.sh --dataset hico --data_root ./data --max_images 10

# Full HICO-DET evaluation
bash ./run_qwen_hoi_eval.sh --dataset hico --data_root ./data

# SWIG-HOI evaluation  
bash ./run_qwen_hoi_eval.sh --dataset swig --data_root ./data
```

### Next Implementation Goal: Loss Computation Framework

**Objective**: Add INP-CC-compatible loss computation to Qwen2.5VL pipeline for comprehensive comparison

**INP-CC Loss Components**:
- `loss_ce`: Cross-entropy classification loss
- `loss_bbox`: Bounding box regression loss (L1/Smooth L1)  
- `loss_giou`: Generalized IoU loss
- `loss_conf`: Confidence/score loss
- Auxiliary losses for training

**Implementation Approach**:
1. **Post-hoc Loss Computation**: Compute losses between Qwen2.5VL predictions and ground truth after inference
2. **Detection Quality Metrics**: Implement L1, GIoU losses for human/object bboxes
3. **Classification Loss**: Cross-entropy between predicted and GT HOI categories  
4. **Confidence Calibration**: Measure correlation between confidence scores and actual detection accuracy

**Technical Considerations**:
- Different architecture: INP-CC uses detection transformer, Qwen2.5VL is generative VLM
- No direct training loss available during Qwen2.5VL inference
- Need to define meaningful "loss" metrics for generative model HOI detection
- Convert evaluation metrics (IoU, classification accuracy) into loss-like values

**Expected Output**: Enhanced evaluation results showing both mAP metrics and loss comparisons with INP-CC baseline