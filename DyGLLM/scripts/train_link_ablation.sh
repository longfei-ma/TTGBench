#!/bin/bash

dataset=$1
model_name=$2
llm_name=$3
device=$4
cpunum=$5
num_neighbors=$6
num_runs=5
train_ratio=0.2

mkdir logs/hy_ablation
# 1. 去除文本
# taskset -c $cpunum  nohup python train_link_ablation_runs.py --dataset_name $dataset  --model_name $model_name --llm_name $llm_name --gpu $device --load_best_configs --empty --empty_ndim $empty_ndim --num_runs $num_runs > logs/hy_ablation/$dataset.$model_name.empty$empty_ndim 2>&1 &
# 2. 不同的语言模型
taskset -c $cpunum  nohup python train_link_ablation_runs.py --dataset_name $dataset  --model_name $model_name --llm_name $llm_name --gpu $device --num_runs $num_runs --num_neighbors $num_neighbors --time_gap $num_neighbors --max_input_sequence_length $num_neighbors --patch_size $num_neighbors > logs/hy_ablation/$dataset.$model_name.$llm_name-n$num_neighbors 2>&1 &
# 2.1 不同语言模型不同训练集比例
# taskset -c $cpunum  nohup python train_link_ablation_runs.py --dataset_name $dataset --train_ratio $train_ratio --model_name $model_name --llm_name $llm_name --gpu $device --num_runs $num_runs --num_neighbors $num_neighbors --max_input_sequence_length $num_neighbors --patch_size $num_neighbors > logs/hy_ablation/${dataset}${train_ratio}.$model_name.$llm_name-n$num_neighbors 2>&1 &
# 3.不同模型使用user-aug
# taskset -c $cpunum  nohup python train_link_ablation_runs.py --dataset_name $dataset  --model_name $model_name --llm_name $llm_name --gpu $device --num_runs $num_runs --num_neighbors $num_neighbors --max_input_sequence_length $num_neighbors --patch_size $num_neighbors --user_aug > logs/hy_ablation/$dataset.$model_name.$llm_name-n$num_neighbors-useraug 2>&1 &
# taskset -c $cpunum  nohup python train_link_ablation_runs.py --dataset_name $dataset  --model_name $model_name --llm_name $llm_name --gpu $device --num_runs $num_runs > logs/hy_ablation/$dataset.$model_name.$llm_name 2>&1 &
# taskset -c $cpunum  nohup python train_link_ablation_runs.py --dataset_name $dataset  --model_name $model_name --llm_name $llm_name --gpu $device --load_best_configs --num_runs $num_runs > logs/hy_ablation/$dataset.$model_name.$llm_name 2>&1 &
# 以下为WalkLM+经典动态图方法脚本
# taskset -c $cpunum  nohup python train_link_ablation_runs.py --dataset_name $dataset  --model_name $model_name --llm_name $llm_name --gpu $device --load_best_configs --walklm > logs/hy_ablation/$dataset.walklm-$model_name.$llm_name 2>&1 &
