#!/bin/bash

server=43
gpuno="2,3" # multi GPU 설정 
IFS=',' read -ra ADDR <<< "$gpuno"  # gpuno 문자열을 쉼표를 구분자로 사용하여 배열로 변환
n_gpu=${#ADDR[@]}

echo "Using ${n_gpu} GPU with DDP Training."

backbone='plato'
dataset=$1
stage='test'
temperature=0.2
window=2
init_checkpoint="${backbone}.${dataset}.${temperature}t.${window}.best_model.pkl"
# init_checkpoint="PLATO.pt"

CUDA_VISIBLE_DEVICES=${gpuno} \
torchrun run.py \
  --stage ${stage} \
  --backbone ${backbone} \
  --temperature ${temperature} \
  --max_turn_view_range 10 \
  --test_batch_size 5 \
  --dev_batch_size 5 \
  --use_turn_embedding True \
  --use_role_embedding True \
  --use_response False \
  --sampler greedy_coreset \
  --percentage 0.5 \
  --window ${window} \
  --dataset ${dataset:-"mwoz"} \
  --init_checkpoint ${init_checkpoint:-"PLATO.pt"} \
  --config_file "plato/config.json" \
  --data_dir "/home/subeen${server}/mllab/research/dial2vec/datasets/${dataset}" \
# > ./logs/dial2vec_${backbone}_${dataset}_${stage}_-1Epochs_GPU${gpuno}.log 2>&1 &

