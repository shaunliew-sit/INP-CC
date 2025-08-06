# INP-CC


This repository contains the official PyTorch implementation for the paper: 

> Ting Lei, Shaofeng Yin, Qingchao Chen, Yuxin Peng, Yang Liu; Open-Vocabulary HOI Detection with Interaction-aware Prompt and Concept Calibration; In Proceedings of the IEEE Conference on International Conference on Computer Vision (ICCV), 2025 


## Overview

> Open Vocabulary Human-Object Interaction (HOI) detection aims to detect interactions between humans and objects while generalizing to novel interaction classes beyond the training set. Current methods often rely on Vision and Language Models (VLMs) but face challenges due to suboptimal image encoders, as image-level pre-training does not align well with the fine-grained region-level interaction detection required for HOI. Additionally, effectively encoding textual descriptions of visual appearances remains difficult, limiting the model’s ability to capture detailed HOI relationships. To address these issues, we propose Interaction-aware Prompting with Concept Calibration (INP-CC), an end-to-end open-vocabulary HOI detector that integrates interaction-aware prompts and concept calibration. Specifically, we propose an interaction-aware prompt generator that dynamically generates a compact set of prompts based on the input scene, enabling selective sharing among similar interactions. This approach directs the model’s attention to key interaction patterns rather than generic image-level semantics, enhancing HOI detection. Furthermore, we refine HOI concept representations through language model-guided calibration, which helps distinguish diverse HOI concepts by leveraging structured semantic knowledge. A negative sampling strategy is also employed to improve inter-modal similarity modeling, enabling the model to better differentiate visually similar but semantically distinct actions. Extensive experimental results demonstrate that INP-CC significantly outperforms state-of-the-art models on the SWIG-HOI and HICO-DET datasets.

## Preparation

### Installation

Our code is built upon [CLIP](https://github.com/openai/CLIP). This repo requires to install [PyTorch](https://pytorch.org/get-started/locally/) and torchvision, as well as small additional dependencies.

```bash
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install ftfy regex tqdm numpy Pillow matplotlib
```

### Dataset

The experiments are mainly conducted on **HICO-DET** and **SWIG-HOI** dataset. We follow [this repo](https://github.com/YueLiao/PPDM) to prepare the HICO-DET dataset. And we follow [this repo](https://github.com/scwangdyd/large_vocabulary_hoi_detection) to prepare the SWIG-HOI dataset.

#### HICO-DET

HICO-DET dataset can be downloaded [here](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk). After finishing downloading, unpack the tarball (`hico_20160224_det.tar.gz`) to the `data` directory. We use the annotation files provided by the [PPDM](https://github.com/YueLiao/PPDM) authors. We re-organize the annotation files with additional meta info, e.g., image width and height. The annotation files can be downloaded from [here](https://drive.google.com/open?id=1lqmevkw8fjDuTqsOOgzg07Kf6lXhK2rg). The downloaded files have to be placed as follows. Otherwise, please replace the default path to your custom locations in [datasets/hico.py](./datasets/hico.py).

``` plain
 |─ data
 │   └─ hico_20160224_det
 |       |- images
 |       |   |─ test2015
 |       |   |─ train2015
 |       |─ annotations
 |       |   |─ trainval_hico_ann.json
 |       |   |─ test_hico_ann.json
 :       :
```

#### SWIG-DET

SWIG-DET dataset can be downloaded [here](https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip). After finishing downloading, unpack the `images_512.zip` to the `data` directory. The annotation files can be downloaded from [here](https://drive.google.com/open?id=1GxNP99J0KP6Pwfekij_M1Z0moHziX8QN). The downloaded files to be placed as follows. Otherwise, please replace the default path to your custom locations in [datasets/swig.py](./datasets/swig.py).

``` plain
 |─ data
 │   └─ swig_hoi
 |       |- images_512
 |       |─ annotations
 |       |   |─ swig_train_1000.json
 |       |   |- swig_val_1000.json
 |       |   |─ swig_trainval_1000.json
 |       |   |- swig_test_1000.json
 :       :
```

### Pre-processed Features

Download the preprocess image features from [link](https://disk.pku.edu.cn/link/AA55FCFC5B31CE4F649AF62BD15E6498C2). The downloaded files have to be placed as follows.

``` plain
 |─ INP-CC
 │   |- swig_image_embeddings.pkl
 │   |- hico_image_embeddings.pkl
 :       :
```

Download the preprocess features from [link](https://disk.pku.edu.cn/link/AA4EC02DC76B0141F19716726FBB253751). The downloaded files have to be placed as follows.

``` plain
 |─ INP-CC
 │   └─ InstructEmbed/1108/
 │       |- swig_embeddings_1108.pkl
 │       |- hico_embeddings_1108.pkl
 :       :
```

## Training

Run this command to train the model in HICO-DET dataset

``` bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port 3996 --use_env main.py \
    --batch_size 32 \
    --output_dir  ckpts/hico \
    --epochs 80 \
    --lr 1e-4 --min-lr 1e-7 \
    --hoi_token_length 20 \
    --enable_dec \
    --dataset_file hico --multi_scale false --use_aux_text true \
    --enable_focal_loss --description_file_path hico_hoi_descriptions.json --VPT_length 4 --img_scene_num 8 --instruction_embedding_file InstructEmbed/1108/hico_embeddings_1108.pkl
```

Run this command to train the model in SWIG-HOI dataset

``` bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port 5786 --use_env main.py \
    --batch_size 64 \
    --output_dir ckpts/swig \
    --epochs 80 \
    --lr 1e-4 --min-lr 1e-7 \
    --hoi_token_length 30 \
    --enable_dec \
    --dataset_file swig \
    --enable_focal_loss --description_file_path swig_hoi_descriptions_6bodyparts.json --VPT_length 4 --img_scene_num 128 --additional_hoi_num 10 --add_hoi_strategy hard --cluster_assignmen_file InstructEmbed/1108/swig_cluster_assignment_64.npy --use_aux_text true --instruction_embedding_file InstructEmbed/1108/swig_embeddings_1108.pkl
```

## Inference

Run this command to evaluate the model on HICO-DET dataset
``` bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 3996 --use_env main.py \
    --batch_size 32 \
    --output_dir  ckpts/hico \
    --epochs 80 \
    --lr 1e-4 --min-lr 1e-7 \
    --hoi_token_length 20 \
    --enable_dec \
    --dataset_file hico --multi_scale false --use_aux_text true \
    --enable_focal_loss --description_file_path hico_hoi_descriptions.json --VPT_length 4 --img_scene_num 8 --instruction_embedding_file InstructEmbed/1108/hico_embeddings_1108.pkl \
    --eval --pretrained [path to ckpt]
```

Run this command to evaluate the model in SWIG-HOI dataset

``` bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 5786 --use_env main.py \
    --batch_size 64 \
    --output_dir ckpts/swig \
    --epochs 80 \
    --lr 1e-4 --min-lr 1e-7 \
    --hoi_token_length 30 \
    --enable_dec \
    --dataset_file swig \
    --enable_focal_loss --description_file_path swig_hoi_descriptions_6bodyparts.json --VPT_length 4 --img_scene_num 128 --additional_hoi_num 10 --add_hoi_strategy hard --cluster_assignmen_file InstructEmbed/1108/swig_cluster_assignment_64.npy --use_aux_text true --instruction_embedding_file InstructEmbed/1108/swig_embeddings_1108.pkl \
    --eval --pretrained [path to ckpt]
```

## Models

| Dataset  | Unseen | Seen  | Full  | Checkpoint |
|----------|--------|-------|-------|------------|
| HICO-DET | 17.38  | 24.74 | 23.12 | [Params](https://disk.pku.edu.cn/link/AADA08FCAA771B4FABB78B674BBC77C287)     |



| Dataset  | Non-rare | Rare  | Unseen | Full  | Checkpoint |
|----------|----------|-------|--------|-------|------------|
| SWIG-HOI | 22.84    | 16.74 | 11.02  | 16.74 | [Params](https://disk.pku.edu.cn/link/AA03E10E19BC1B4A0CAE1CF1CE78FAC09E)  |



## Acknowledgement
We would also like to thank the anonymous reviewers for their constructive feedback.


