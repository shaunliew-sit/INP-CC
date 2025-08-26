#!/bin/bash
# INP-CC Inference Script with Enhanced Visualization Support
# Usage: 
#   bash ./inference.sh hico        # Run HICO-DET evaluation
#   bash ./inference.sh swig        # Run SWIG-HOI evaluation
#   bash ./inference.sh hico-vis    # Run HICO-DET with enhanced visualization
#   bash ./inference.sh swig-vis    # Run SWIG-HOI with enhanced visualization

DATASET=${1:-"hico"}

case $DATASET in
    "hico")
        echo "🚀 Running HICO-DET evaluation..."
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
            --instruction_embedding_file checkpoints/1108/hico_embeddings_1108.pkl \
            --eval --pretrained checkpoints/hico_checkpoint.pth
        ;;
    
    "hico-vis")
        echo "🎨 Running HICO-DET evaluation with Enhanced Visualization..."
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
            --instruction_embedding_file checkpoints/1108/hico_embeddings_1108.pkl \
            --eval --pretrained checkpoints/hico_checkpoint.pth \
            --enhanced_vis --vis_max_images 50 --vis_threshold 0.1 --vis_max_detections 10
        ;;
    
    "swig")
        echo "🚀 Running SWIG-HOI evaluation..."
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
            --use_aux_text true --instruction_embedding_file checkpoints/1108/swig_embeddings_1108.pkl \
            --eval --pretrained checkpoints/swig_checkpoint.pth
        ;;
    
    "swig-vis")
        echo "🎨 Running SWIG-HOI evaluation with Enhanced Visualization..."
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
            --use_aux_text true --instruction_embedding_file checkpoints/1108/swig_embeddings_1108.pkl \
            --eval --pretrained checkpoints/swig_checkpoint.pth \
            --enhanced_vis --vis_max_images 50 --vis_threshold 0.1 --vis_max_detections 10
        ;;
    
    *)
        echo "❌ Invalid dataset option: $DATASET"
        echo "📋 Available options:"
        echo "   hico        - HICO-DET evaluation only"
        echo "   hico-vis    - HICO-DET with enhanced visualization"
        echo "   swig        - SWIG-HOI evaluation only"
        echo "   swig-vis    - SWIG-HOI with enhanced visualization"
        echo ""
        echo "💡 Examples:"
        echo "   ./inference.sh hico-vis     # HICO with visualization"
        echo "   ./inference.sh swig-vis     # SWIG with visualization"
        exit 1
        ;;
esac

echo "✅ Inference completed! Check output directory for results."