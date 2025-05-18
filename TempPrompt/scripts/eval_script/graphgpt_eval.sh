# to fill in the following path to extract projector for the second tuning stage!
dataset=${1-"FOOD"}
task=${2:-"lp"}
emb=${3:-"sbert"}
cpunum=$4
run=$5
empty=${6:-"empty"}
sample_size=2
pretra_gnn=clip_gcn
prefix=llaga-vicuna-7b-${emb}-2-${sample_size}
model_path=checkpoints/stage_2/${dataset}/${prefix}_${task}
output_path="results"
empty_ndim=384

# empty
if [ "${empty}" = "empty" ];then
    taskset -c $cpunum python graphgpt/eval/run_graphgpt.py --dataset ${dataset} --raw --run ${run} --empty --empty_ndim ${empty_ndim} --task ${task} --pretrained_embedding_type ${emb} --model_path ${model_path} --pretrain_graph_model_path text-graph-grounding/res/${dataset} --graph_tower ${pretra_gnn} --sample_neighbor_size ${sample_size} --output_path ${output_path} --cache_dir ../LLaGA/checkpoint/vicuna-7b-v1.5-16k > logs/test_${dataset}_stage1_${run}_${task}_${emb}.log 2>&1
# sbert
else
    taskset -c $cpunum python graphgpt/eval/run_graphgpt.py --dataset ${dataset} --raw --run ${run} --task ${task} --pretrained_embedding_type ${emb} --model_path ${model_path} --pretrain_graph_model_path text-graph-grounding/res/${dataset} --graph_tower ${pretra_gnn} --sample_neighbor_size ${sample_size} --output_path ${output_path} --cache_dir ../LLaGA/checkpoint/vicuna-7b-v1.5-16k > logs/test_${dataset}_stage1_${run}_${task}_${emb}.log 2>&1
fi