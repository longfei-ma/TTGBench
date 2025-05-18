#!/bin/bash

dataset=${1-"FOOD"}
task=${2:-"lp"}
cpunum=$3
sample_size=2
output_path="temprompt"
num_runs=2


# 非empty
for variant in "nondst2" "dst2-v1" "dst2-v2";
do
    taskset -c $cpunum python graphgpt/eval/eval_temprompt.py --dataset ${dataset} --variant ${variant} --task ${task} --res_path ${output_path} --num_runs ${num_runs} > logs/temprompt_${dataset}_stage2_${task}.log 2>&1
done

# empty
for variant in "nondst2" "dst2-v1" "dst2-v2";
do
    taskset -c $cpunum python graphgpt/eval/eval_temprompt.py --dataset ${dataset} --empty --variant ${variant}  --task ${task} --res_path ${output_path} --num_runs ${num_runs} > logs/temprompt_${dataset}_stage2_${task}_empty.log 2>&1
done