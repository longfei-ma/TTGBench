#!/bin/bash

model=${1:-"vicuna"}
task=${2:-"lp"}
dataset=${3-"FOOD"}
bs=${4:-16}
emb=${5:-"sbert-64"}
cpunum=$6
use_hop=2
sample_size=2
template="ND"
num_runs=3

if [ ${model} = "vicuna" ]; then
  projector_type="linear"
  prefix=llaga-vicuna-7b-${emb}-${use_hop}-${sample_size}-${projector_type}-projector
  model_base=lmsys/vicuna-7b-v1.5-16k
  mode="v1"
elif [ ${model} = "vicuna_2layer" ]; then
  use_hop=2
  template="ND"
  projector_type="2-layer-mlp"
  prefix=llaga-vicuna-7b-${emb}-${use_hop}-${sample_size}-${projector_type}-projector
  model_base=lmsys/vicuna-7b-v1.5-16k
  mode="v1"
elif [ ${model} = "vicuna_4hop" ]; then
  use_hop=4
  template="HO"
  projector_type="linear"
  prefix=llaga-vicuna-7b-${emb}-${use_hop}-hop-token-${projector_type}-projector
  model_base=lmsys/vicuna-7b-v1.5-16k
  mode="v1"
elif [ ${model} = "vicuna_4hop_2layer" ]; then
  use_hop=4
  template="HO"
  projector_type="2-layer-mlp"
  prefix=llaga-vicuna-7b-${emb}-${use_hop}-hop-token-${projector_type}-projector
  model_base=lmsys/vicuna-7b-v1.5-16k
  mode="v1"
elif [ ${model} = "llama" ]; then
  projector_type="linear"
  prefix=llaga-llama-2-7b-hf-${emb}-${use_hop}-${sample_size}-${projector_type}-projector
  model_base=meta-llama/Llama-2-7b-hf
  mode="llaga_llama_2"
elif [ ${model} = "llama_4hop" ]; then
  projector_type="linear"
  prefix=llaga-llama-2-7b-hf-${emb}-${use_hop}-hop-token-${projector_type}-projector
  model_base=meta-llama/Llama-2-7b-hf
  mode="llaga_llama_2"
elif [ ${model} = "opt_2.7b" ]; then
  use_hop=2
  template="ND"
  projector_type="linear"
  prefix=llaga-opt-2.7b-${emb}-${use_hop}-${sample_size}-${projector_type}-projector
  model_base=facebook/opt-2.7b
  max_len=1536
  mode="v1"
elif [ ${model} = "opt_2.7b_4hop" ]; then
  use_hop=4
  template="HO"
  projector_type="linear"
  prefix=llaga-opt-2.7b-${emb}-${use_hop}-hop-token-${projector_type}-only-train-pretrain
  model_base=facebook/opt-2.7b
  max_len=1536
  mode="v1"
fi

model_path="./checkpoints/${dataset}/${prefix}_${task}"
model_base="vicuna-7b-v1.5-16k" #meta-llama/Llama-2-7b-hf
output_path="results"

taskset -c $cpunum python eval/eval_pretrain.py \
--dataset ${dataset} \
--template ${template} \
--task ${task} \
--model_path ${model_path} \
--model_base ${model_base} \
--conv_mode  ${mode} \
--pretrained_embedding_type ${emb} \
--use_hop ${use_hop} \
--sample_neighbor_size ${sample_size} \
--output_path ${output_path} \
--num_runs ${num_runs} \
--cache_dir "checkpoint/${model_base}" > logs/test_${dataset}_stage1.log 2>&1 && taskset -c $cpunum python eval/eval_res.py --dataset ${dataset} --task ${task} --res_path ${output_path} --num_runs ${num_runs} > logs/test_${dataset}_stage2.log 2>&1 
