# to fill in the following path to extract projector for the second tuning stage!
dataset=${1:-"FOOD"}
task=${2:-"lp"}
cpunum=$3
run=$4
variant=$5
empty=${6:-"empty"}
num_parts=$7
current_part=$8
llm_name=$9

sample_size=2
output_path="temprompt"
empty_ndim=384


# for variant in "nondst2" "dst2-v1" "dst2-v2";
# do
#     taskset -c $cpunum python graphgpt/eval/run_temprompt.py --dataset ${dataset} --run ${run} --task ${task} --variant ${variant} --sample_neighbor_size ${sample_size} --output_path ${output_path} --cache_dir ../LLaGA/checkpoint/vicuna-7b-v1.5-16k --model_max_length ${model_max_length} > logs/temprompt_${dataset}_stage1_${run}_${task}_${variant}.log 2>&1
# done
# empty 
if [ "${empty}" = "empty" ];then
    taskset -c $cpunum python graphgpt/eval/run_llm_service_split.py --dataset ${dataset} --llm_name ${llm_name} --run ${run} --empty --empty_ndim ${empty_ndim} --task ${task} --variant ${variant} --num_parts ${num_parts} --current_part ${current_part} --sample_neighbor_size ${sample_size} --output_path ${output_path} > logs/${llm_name}_${dataset}_stage1_${run}_${task}_${variant}_empty.log${current_part} 2>&1
else
    taskset -c $cpunum python graphgpt/eval/run_llm_service_split.py --dataset ${dataset} --llm_name ${llm_name} --run ${run} --task ${task} --variant ${variant} --num_parts ${num_parts} --current_part ${current_part} --sample_neighbor_size ${sample_size} --output_path ${output_path} > logs/${llm_name}_${dataset}_stage1_${run}_${task}_${variant}.log${current_part} 2>&1
fi

# empty
# for variant in "nondst2" "dst2-v1" "dst2-v2";
# do
#     taskset -c $cpunum python graphgpt/eval/run_temprompt.py --dataset ${dataset} --run ${run} --empty --empty_ndim ${empty_ndim} --task ${task} --variant ${variant} --sample_neighbor_size ${sample_size} --output_path ${output_path} --cache_dir ../LLaGA/checkpoint/vicuna-7b-v1.5-16k --model_max_length ${model_max_length} > logs/temprompt_${dataset}_stage1_${run}_${task}_${variant}_empty.log 2>&1
# done