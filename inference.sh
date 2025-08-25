# hico_det
python -m torch.distributed.launch --nproc_per_node=1 --master_port 3996 --use_env main.py \
    --batch_size 32 \
    --output_dir  ckpts/hico \
    --epochs 80 \
    --lr 1e-4 --min-lr 1e-7 \
    --hoi_token_length 20 \
    --enable_dec \
    --dataset_file hico --multi_scale false --use_aux_text true \
    --enable_focal_loss --description_file_path hico_hoi_descriptions.json --VPT_length 4 --img_scene_num 8 --instruction_embedding_file checkpoints/1108/hico_embeddings_1108.pkl \
    --eval --pretrained checkpoints/hico_checkpoint.pth

# swig_hoi
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 5786 --use_env main.py \
#     --batch_size 64 \
#     --output_dir ckpts/swig \
#     --epochs 80 \
#     --lr 1e-4 --min-lr 1e-7 \
#     --hoi_token_length 30 \
#     --enable_dec \
#     --dataset_file swig \
#     --enable_focal_loss --description_file_path swig_hoi_descriptions_6bodyparts.json --VPT_length 4 --img_scene_num 128 --additional_hoi_num 10 --add_hoi_strategy hard --cluster_assignmen_file InstructEmbed/1108/swig_cluster_assignment_64.npy --use_aux_text true --instruction_embedding_file checkpoints/1108/swig_embeddings_1108.pkl \
#     --eval --pretrained checkpoints/swig_checkpoint.pth