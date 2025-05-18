#!/bin/bash

dataset=$1
device=$2
cpunum=$3
llm_name=$4
# empty_type=$5
num_runs=1
num_neighbors=2
train_ratio=0.2

mkdir logs/link_predict
# for negative_sample_strategy in random historical inductive;
# for negative_sample_strategy in historical inductive;

for negative_sample_strategy in random;
do
    for model_name in "JODIE" "DyRep" "TGAT" "TGN" "CAWN" "TCL" "GraphMixer" "DyGFormer";
    # for model_name in "JODIE" "DyRep" "TGAT" "GraphMixer" "DyGFormer";
    # for model_name in "GraphMixer";
    do
        # 再跑各embedding方法的结果
        taskset -c $cpunum python evaluate_link_prediction_ablation.py --dataset_name $dataset  --model_name $model_name --llm_name $llm_name --negative_sample_strategy $negative_sample_strategy --transductive --num_runs $num_runs --gpu $device --num_neighbors $num_neighbors --time_gap $num_neighbors --max_input_sequence_length $num_neighbors --patch_size $num_neighbors > logs/link_predict/$dataset.$model_name.$llm_name.$negative_sample_strategy-n$num_neighbors 2>&1
        # 2.1 不同语言模型不同训练集比例
        # taskset -c $cpunum  nohup python evaluate_link_prediction_ablation.py --dataset_name $dataset --train_ratio $train_ratio --model_name $model_name --llm_name $llm_name --negative_sample_strategy $negative_sample_strategy --transductive --num_runs $num_runs --gpu $device --num_neighbors $num_neighbors --max_input_sequence_length $num_neighbors --patch_size $num_neighbors > logs/link_predict/${dataset}${train_ratio}.$model_name.$llm_name.$negative_sample_strategy-n$num_neighbors 2>&1

        # 使用用户画像增强
        # taskset -c $cpunum python evaluate_link_prediction_ablation.py --dataset_name $dataset  --model_name $model_name --llm_name $llm_name --negative_sample_strategy $negative_sample_strategy --transductive --num_runs $num_runs --gpu $device --num_neighbors $num_neighbors --max_input_sequence_length $num_neighbors --patch_size $num_neighbors --user_aug > logs/link_predict/$dataset.$model_name.$llm_name.$negative_sample_strategy-n$num_neighbors-useraug 2>&1
    done
done

# device=$1
# llm_name=$2
# cpunum=$3

# for dataset in FOOD IMDB Librarything;
# do
#     for negative_sample_strategy in random historical inductive;
#     # for negative_sample_strategy in historical inductive;
#     do
#         # for model_name in "JODIE" "DyRep" "TGAT" "TGN" "CAWN" "TCL" "GraphMixer" "DyGFormer";
#         # 下面是WalkLM
#         for model_name in "TGAT" "DyGFormer";
#         # for model_name in "DyGFormer";
#         do
#             # taskset -c $cpunum python evaluate_link_prediction_simple.py --dataset_name $dataset  --model_name $model_name --llm_name $llm_name --negative_sample_strategy $negative_sample_strategy --transductive --num_runs 5 --gpu $device --load_best_configs > logs/link_predict/$dataset.$model_name.$llm_name.$negative_sample_strategy 2>&1

#             # 以下为WalkLM+经典动态图方法脚本
#             taskset -c $cpunum python evaluate_link_prediction_simple.py --dataset_name $dataset  --model_name $model_name --llm_name $llm_name --negative_sample_strategy $negative_sample_strategy --transductive --num_runs 5 --gpu $device --load_best_configs --walklm > logs/link_predict/$dataset.walklm-$model_name.$llm_name.$negative_sample_strategy 2>&1
#         done 
#     done
# done