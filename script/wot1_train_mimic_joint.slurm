#!/bin/bash
#SBATCH -J iu_xray # Job name
#SBATCH -o ./log/wot1_mimic_freezet_joint_8layer.out
#SBATCH -p bigTiger        # Partition name (e.g., gpu, compute)
#SBATCH --mem=256G        # Memory CPU
#SBATCH -t 60:00:00             # Maximum runtime (HH:MM:SS)
#SBATCH -N 1                         # Number of nodes
#SBATCH --gres=gpu:h100_80gb       # Request k A6000 GPUs
#SBATCH --cpus-per-task=8      # 每个任务分配16个CPU核心 <<< 核心参数
#SBATCH --ntasks-per-node=1     # 单节点任务数固定为1

# Activate the conda environment
source ~/miniconda3/bin/activate jima

base_dir=~/gitpj/JIMA
python $base_dir/main_train2.py \
--image_dir /project/wli5/JIMA/data/mimic_cxr/images/ \
--ann_path /project/wli5/JIMA/data/mimic_cxr/annotation_label_with_filtered_tokens.json \
--dataset_name mimic_cxr \
--max_seq_length 100 \
--threshold 10 \
--batch_size 128 \
--epochs 100 \
--save_dir /project/wli5/JIMA/results/mimic_cxr/0403 \
--step_size 1 \
--gamma 0.8 \
--seed 456789 \
--num_layers 8 \
--beam_size 5 \
--d_model 768 \
--d_ff 2048 \
--num_heads 12 \
--dropout 0.2 \
--lr_ed 5e-5 \
--weight_decay 1e-4