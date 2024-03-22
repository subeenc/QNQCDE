#!/bin/bash

echo "Start Training"

python experiment2/main.py \
  --test False \
  --max_len 50 \
  --batch_size 4 \
  --seq_len 64 \
  --epochs 3 \
  --eval_steps 250 \
  --lr 0.0001 \
  --warmup_ratio 0.05 \
  --temperature 0.2 \
  --path_to_data  /home/subeen40/cl_research/dial2vec_40/datasets/mwoz \
  --train_data train.tsv \
  --valid_data clustering_dev.tsv

#../../../../cl_research/dial2vec40/datasets/doc2dial
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

