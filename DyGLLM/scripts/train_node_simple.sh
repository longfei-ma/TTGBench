#!/bin/bash

dataset=$1
model_name=$2
llm_name=$3
device=$4
cpunum=$5
num_runs=2
num_neighbors=2
empty_ndim=384

mkdir logs/hy_node
# 1. 去除文本
taskset -c $cpunum  nohup python train_node_simple.py --dataset_name $dataset  --model_name $model_name --llm_name $llm_name --gpu $device --load_best_configs --empty --empty_ndim $empty_ndim --num_runs $num_runs --num_neighbors $num_neighbors --time_gap $num_neighbors --max_input_sequence_length $num_neighbors --patch_size $num_neighbors > logs/hy_node/$dataset.$model_name.empty$empty_ndim 2>&1 &

# 2. 不同的语言模型
# taskset -c $cpunum nohup python train_node_simple.py --dataset_name $dataset  --model_name $model_name --llm_name $llm_name --gpu $device --load_best_configs --num_runs $num_runs --num_neighbors $num_neighbors --max_input_sequence_length $num_neighbors --patch_size $num_neighbors > logs/hy_node/$dataset.$model_name.$llm_name 2>&1 &
