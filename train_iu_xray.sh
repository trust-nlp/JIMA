nohup python main_train.py \
--image_dir /project/wli5/JIMA/data/iu_xray/images/ \
--ann_path /project/wli5/JIMA/data/iu_xray/annotation_label_with_filtered_tokens.json \
--dataset_name iu_xray \
--max_seq_length 60 \
--threshold 3 \
--batch_size 32 \
--epochs 100 \
--save_dir results/iu_xray \
--step_size 50 \
--gamma 0.1 \
--seed 9 > train_iu_nofreeze.log 2>&1 &
# --freeze_visual_extractor_on_task2 \
