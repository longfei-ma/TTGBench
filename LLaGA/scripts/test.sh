#!/bin/bash

model=${1:-"vicuna"}
task=${2:-"lp"}
dataset=${3:-"FOOD"}
bs=${4:-16}
emb=${5:-"sbert"}
cpunum=$6
run=$7
empty=${8:-"empty"}
use_hop=2
sample_size=2
num_runs=3


if [ ${model} = "vicuna" ]; then
  template="ND"
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

if [ "${task}" = "lp" ];then
  taskset -c $cpunum python eval/eval_pretrain_link.py --dataset ${dataset} --run ${run} --template ${template} --task ${task} --pretrained_embedding_type ${emb} --model_path ${model_path} --model_base ${model_base} --conv_mode  ${mode}  --use_hop ${use_hop} --sample_neighbor_size ${sample_size} --output_path ${output_path} --cache_dir "checkpoint/${model_base}" --task ${task} --template ${template} --res_path ${output_path} --llm_name ${emb} --num_runs ${num_runs}
elif [ "${task}" = "nc" ];then
  taskset -c $cpunum python eval/eval_pretrain_node.py --dataset ${dataset} --run ${run} --template ${template} --task ${task} --pretrained_embedding_type ${emb} --model_path ${model_path} --model_base ${model_base} --conv_mode  ${mode}  --use_hop ${use_hop} --sample_neighbor_size ${sample_size} --output_path ${output_path} --cache_dir "checkpoint/${model_base}" 
fi

taskset -c $cpunum python eval/eval_res.py --dataset ${dataset} --task ${task} --template ${template} --res_path ${output_path} --llm_name ${emb} --num_runs ${num_runs}