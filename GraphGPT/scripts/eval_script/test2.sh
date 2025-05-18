#!/bin/bash

dataset=${1-"FOOD"}
task=${2:-"lp"}
emb=${3:-"sbert"}
cpunum=$4
sample_size=
pretra_gnn=
output_path=
num_runs=


taskset -c $cpunum python graphgpt/eval/eval_res.py --dataset ${dataset} --task ${task} --res_path ${output_path} --llm_name ${emb} --num_runs ${num_runs} > logs/test_${dataset}_stage2_${task}_${emb}.log 2>&1 