#!/bin/bash
#SBATCH -J mimic_jima # Job name
#SBATCH -o mimic_resplit.out
#SBATCH -p bigTiger        # Partition name (e.g., gpu, compute)
#SBATCH --mem=256G        # Memory CPU
#SBATCH -t 24:00:00             # Maximum runtime (HH:MM:SS)
#SBATCH -N 1                         # Number of nodes
#SBATCH --gres=gpu:rtx_5000       # Request k A6000 GPUs

# Activate the conda environment
source ~/miniconda3/bin/activate jima

python main_train.py \
--image_dir /project/wli5/JIMA/data/mimic_cxr/images/ \
--ann_path /project/wli5/JIMA/data/mimic_cxr/annotation_label_with_filtered_tokens.json \
--dataset_name mimic_cxr \
--max_seq_length 100 \
--threshold 10 \
--batch_size 128 \
--epochs 60 \
--save_dir /project/wli5/JIMA/results/mimic_cxr \
--step_size 1 \
--gamma 0.8 \
--freeze_visual_extractor_on_task2 \
--seed 456789 \
--split_seed 121212