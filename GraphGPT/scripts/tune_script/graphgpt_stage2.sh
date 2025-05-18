# to fill in the following path to run the second stage of our GraphGPT!
model_path=../LLaGA/checkpoint/vicuna-7b-v1.5-16k
graph_data_path=./graph_data/all_graph_data.pt
pretra_gnn=clip_gcn
tuned_proj=./checkpoints/stage_1_projector/stage_1_projector.bin
dataset=$1
task=${2:-"lp"}
emb=${3:-"sbert-64"}
cpunum=$4
run=$5
empty=${6:-"empty"}
sample_size=2
empty_ndim=384
prefix=llaga-vicuna-7b-${emb}-2-${sample_size}
output_dir=./checkpoints/stage_2/${dataset}/${prefix}_${task}

wandb offline
# sbert
# taskset -c $cpunum python  graphgpt/train/train_mem.py --dataset ${dataset} --task ${task} --empty False --run $run --pretrained_embedding_type ${emb} --model_name_or_path ${model_path} --version v1 --graph_tower ${pretra_gnn} --pretrain_graph_model_path text-graph-grounding/res/${dataset} --tune_graph_mlp_adapter True --graph_select_layer -2 --use_graph_start_end True --bf16 True --output_dir ${output_dir} --num_train_epochs 1 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 1 --evaluation_strategy no --save_strategy "epoch" --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --tf32 True --model_max_length 4096 --gradient_checkpointing True --lazy_preprocess True --report_to wandb --sample_neighbor_size ${sample_size} > logs/train_${dataset}_${task}_${emb}_${run}.log 2>&1

if [ "${dataset}" = "Amazon-Kindle" ];then
    train_bs=8
else
    train_bs=8
fi

# empty raw
if [ "${empty}" = "empty" ];then
    taskset -c $cpunum python  graphgpt/train/train_mem.py --dataset ${dataset} --task ${task} --raw True --empty True --run $run --pretrained_embedding_type ${emb} --model_name_or_path ${model_path} --version v1 --graph_tower ${pretra_gnn} --pretrain_graph_model_path text-graph-grounding/res/${dataset} --tune_graph_mlp_adapter True --graph_select_layer -2 --use_graph_start_end True --bf16 True --output_dir ${output_dir} --num_train_epochs 1 --per_device_train_batch_size ${train_bs} --per_device_eval_batch_size 8 --gradient_accumulation_steps 1 --evaluation_strategy no --save_strategy "steps" --save_steps 50000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --tf32 True --model_max_length 2048 --gradient_checkpointing True --lazy_preprocess True --report_to wandb --sample_neighbor_size ${sample_size} > logs/train_${dataset}_${task}_empty-${empty_ndim}_${run}.log 2>&1
# sbert raw
else
    taskset -c $cpunum python  graphgpt/train/train_mem.py --dataset ${dataset} --task ${task} --raw True --empty False --run $run --pretrained_embedding_type ${emb} --model_name_or_path ${model_path} --version v1 --graph_tower ${pretra_gnn} --pretrain_graph_model_path text-graph-grounding/res/${dataset} --tune_graph_mlp_adapter True --graph_select_layer -2 --use_graph_start_end True --bf16 True --output_dir ${output_dir} --num_train_epochs 1 --per_device_train_batch_size ${train_bs} --per_device_eval_batch_size 8 --gradient_accumulation_steps 1 --evaluation_strategy no --save_strategy "steps" --save_steps 50000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --tf32 True --model_max_length 2048 --gradient_checkpointing True --lazy_preprocess True --report_to wandb --sample_neighbor_size ${sample_size} > logs/train_${dataset}_${task}_${emb}_${run}-raw.log 2>&1
fi



# python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=20001 \
#     graphgpt/train/train_mem.py \
#     --run ${run} \
#     --pretrained_embedding_type  \
#     --model_name_or_path ${model_path} \
#     --version v1 \
#     --data_path ${instruct_ds} \
#     --graph_content ./arxiv_ti_ab.json \
#     --graph_data_path ${graph_data_path} \
#     --graph_tower ${pretra_gnn} \
#     --pretrain_graph_mlp_adapter ${tuned_proj} \
#     --tune_graph_mlp_adapter True \
#     --graph_select_layer -2 \
#     --use_graph_start_end True\
#     --bf16 True \
#     --output_dir ${output_model} \
#     --num_train_epochs 2 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb
