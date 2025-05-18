# to fill in the following path to run the second stage of our GraphGPT!
model_path=
graph_data_path=
pretra_gnn=
tuned_proj=
task=${2:-"lp"}
emb=${3:-"sbert"}
cpunum=$4
run=$5
sample_size=
prefix=
output_dir=

wandb offline

if [ "${dataset}" = "Amazon-Kindle" ];then
    train_bs=8
else
    train_bs=8
fi


taskset -c $cpunum python  graphgpt/train/train_mem.py --dataset ${dataset} --task ${task} --run $run --pretrained_embedding_type ${emb} --model_name_or_path ${model_path} --version v1 --graph_tower ${pretra_gnn} --pretrain_graph_model_path text-graph-grounding/res/${dataset} --tune_graph_mlp_adapter True --graph_select_layer -2 --use_graph_start_end True --bf16 True --output_dir ${output_dir} --num_train_epochs 1 --per_device_train_batch_size ${train_bs} --per_device_eval_batch_size 8 --gradient_accumulation_steps 1 --evaluation_strategy no --save_strategy "steps" --save_steps 50000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --tf32 True --model_max_length 2048 --gradient_checkpointing True --lazy_preprocess True --report_to wandb --sample_neighbor_size ${sample_size} > logs/train_${dataset}_${task}_${emb}_${run}-raw.log 2>&1