# to fill in the following path to extract projector for the second tuning stage!
dataset=${1-"FOOD"}
task=${2:-"lp"}
emb=${3:-"sbert"}
cpunum=$4
run=$5
sample_size=
pretra_gnn=
prefix=
model_path=
output_path=
cache_dir=

taskset -c $cpunum python graphgpt/eval/run_graphgpt.py --dataset ${dataset} --run ${run} --task ${task} --pretrained_embedding_type ${emb} --model_path ${model_path} --pretrain_graph_model_path text-graph-grounding/res/${dataset} --graph_tower ${pretra_gnn} --sample_neighbor_size ${sample_size} --output_path ${output_path} --cache_dir $cache_dir > logs/test_${dataset}_stage1_${run}_${task}_${emb}.log 2>&1