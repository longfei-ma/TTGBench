#!/bin/bash

dataset=$1
device=$2
cpunum=$3
llm_name=$4
num_runs=3

mkdir logs

for model_name in "JODIE" "DyRep" "TGAT" "TGN" "CAWN" "TCL" "GraphMixer" "DyGFormer";
do
    taskset -c $cpunum python test_link.py --dataset_name $dataset --model_name $model_name --llm_name $llm_name --transductive --num_runs $num_runs --gpu $device > logs/testlink-$dataset.$model_name.$llm_name 2>&1
done
