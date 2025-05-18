#!/bin/bash

dataset=${1-"FOOD"}
task=${2:-"lp"}
emb=${3:-"sbert"}
cpunum=$4
sample_size=2
pretra_gnn=clip_gcn
output_path="results"
num_runs=2


# 非empty
taskset -c $cpunum python graphgpt/eval/eval_res.py --dataset ${dataset} --raw --task ${task} --res_path ${output_path} --llm_name ${emb} --num_runs ${num_runs} > logs/test_${dataset}_stage2_${task}_${emb}.log 2>&1 

# empty
taskset -c $cpunum python graphgpt/eval/eval_res.py --dataset ${dataset} --raw --empty --task ${task} --res_path ${output_path} --llm_name ${emb} --num_runs ${num_runs} > logs/test_${dataset}_stage2_${task}_empty.log 2>&1 