#!/bin/bash

dataset=$1
device=$2
llm_name=$3
cpunum=$4
num_runs=2
num_neighbors=2
empty_ndim=384

mkdir logs/node_predict

for model_name in "JODIE" "DyRep" "TGAT" "TGN" "CAWN" "TCL" "GraphMixer" "DyGFormer";
# for model_name in "DyGFormer";
do  
    # 不同的语言模型
    taskset -c $cpunum python test_node_simple.py --dataset_name $dataset  --model_name $model_name --llm_name $llm_name  --num_runs $num_runs --gpu $device --num_neighbors $num_neighbors --max_input_sequence_length $num_neighbors --patch_size $num_neighbors > logs/node_predict/$dataset.$model_name.$llm_name 2>&1
done 

for model_name in "JODIE" "DyRep" "TGAT" "TGN" "CAWN" "TCL" "GraphMixer" "DyGFormer";
# for model_name in "DyGFormer";
do  
    # 去除文本
    taskset -c $cpunum python test_node_simple.py --dataset_name $dataset  --model_name $model_name --llm_name $llm_name  --num_runs $num_runs --gpu $device --num_neighbors $num_neighbors --time_gap $num_neighbors --max_input_sequence_length $num_neighbors --patch_size $num_neighbors --empty --empty_ndim ${empty_ndim} > logs/node_predict/$dataset.$model_name.empty$empty_ndim 2>&1
done 


