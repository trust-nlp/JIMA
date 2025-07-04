#!/bin/bash
#SBATCH -J iu_xray # Job name
#SBATCH -o ./log/iu_freezet2.out
#SBATCH -p bigTiger        # Partition name (e.g., gpu, compute)
#SBATCH --mem=256G        # Memory CPU
#SBATCH -t 48:00:00             # Maximum runtime (HH:MM:SS)
#SBATCH -N 1                         # Number of nodes
#SBATCH --gres=gpu:rtx_6000       # Request k A6000 GPUs
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

# Activate the conda environment
source ~/miniconda3/bin/activate jima

base_dir=~/gitpj/JIMA
python $base_dir/main_train.py \
--image_dir /project/wli5/JIMA/data/iu_xray/images/ \
--ann_path /project/wli5/JIMA/data/iu_xray/annotation_label_with_filtered_tokens.json \
--dataset_name iu_xray \
--max_seq_length 60 \
--threshold 3 \
--threshold 3 \
--batch_size 32 \
--epochs 100 \
--save_dir /project/wli5/JIMA/results/iu_xray/0402 \
--step_size 50 \
--gamma 0.1 \
--seed 9 \
--num_layers 6 \
--beam_size 5 \
--dropout 0.1 
# --joint  # joint training, otherwise alternating training