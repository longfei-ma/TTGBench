#!/bin/bash

dataset=$1
model_name=$2
llm_name=$3
device=$4
cpunum=$5
num_runs=3

mkdir logs

taskset -c $cpunum  nohup python train_link.py --dataset_name $dataset  --model_name $model_name --llm_name $llm_name --gpu $device --num_runs $num_runs > logs/trainlink-$dataset.$model_name.$llm_name 2>&1 &
