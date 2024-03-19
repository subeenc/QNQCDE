#!/bin/bash

echo "Start Training"

python main.py \
  --test False \
  --max_len 50 \
  --batch_size 4 \
  --epochs 3 \
  --eval_steps 250 \
  --lr 0.0001 \
  --warmup_ratio 0.05 \
  --temperature 0.2 \
  --path_to_data ../../../cl_research/OurModel/datasets \
  --train_data sgd_train_with_pairs.csv \
  --valid_data sgd_clustering_dev_with_pairs.csv 

# echo "Start Testing"

# python main.py \
#  --train False \
#  --test True \
#  --max_len 50 \
#  --batch_size 256 \
#  --epochs 3 \
#  --eval_steps 250 \
#  --lr 0.00005 \
#  --warmup_ratio 0.05 \
#  --temperature 0.2 \
#  --path_to_data ../../../cl_research/OurModel/datasets \
#  --test_data sgd_clustering_test_with_pairs.csv  \
#  --path_to_saved_model output/best_checkpoint.pt

#echo "Semantic Search"

#python SemanticSearch.py

