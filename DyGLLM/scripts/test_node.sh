#!/bin/bash

dataset=$1
device=$2
llm_name=$3
cpunum=$4
num_runs=3

mkdir logs

for model_name in "JODIE" "DyRep" "TGAT" "TGN" "CAWN" "TCL" "GraphMixer" "DyGFormer";
do  
    taskset -c $cpunum python test_node.py --dataset_name $dataset  --model_name $model_name --llm_name $llm_name  --num_runs $num_runs --gpu $device > logs/testnode-$dataset.$model_name.$llm_name 2>&1
done 