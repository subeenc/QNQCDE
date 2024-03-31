#!/bin/bash

echo "Start Training"

python main.py \
  --test False \
  --init_checkpoint "/home/jihyeon41/SimCDE/experiment2/model/PLATO.pt" \
  --config_file "/home/jihyeon41/SimCDE/experiment2/model/plato/config.json" \
  --max_len 50 \
  --batch_size 4 \
  --seq_len 64 \
  --epochs 3 \
  --eval_steps 250 \
  --lr 0.0001 \
  --warmup_ratio 0.05 \
  --temperature 0.2 \
  --path_to_data /home/jihyeon41/research_dial_embedding/dial2vec_git/dial2vec/datasets/mwoz \
  --train_data train.tsv \
  --valid_data clustering_dev.tsv

# echo "Start Testing"

# python main.py \
#  --train False \
#  --test True \
#  --max_len 50 \
#  --batch_size 256 \
#  --seq_len 16 \
#  --epochs 3 \
#  --eval_steps 250 \
#  --lr 0.00005 \
#  --warmup_ratio 0.05 \
#  --temperature 0.2 \
#  --path_to_data /home/jihyeon41/datasets \
#  --test_data sgd_clustering_test_with_pairs.csv  \
#  --path_to_saved_model output/best_checkpoint.pt

#echo "Semantic Search"

#python SemanticSearch.py