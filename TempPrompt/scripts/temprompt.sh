# to fill in the following path to extract projector for the second tuning stage!
dataset=${1:-"FOOD"}
task=${2:-"lp"}
cpunum=$3
run=$4
variant=$5
sample_size=
output_path=
cache_dir=
num_runs=


taskset -c $cpunum python graphgpt/eval/run_temprompt.py --dataset ${dataset} --run ${run} --task ${task} --variant ${variant} --sample_neighbor_size ${sample_size} --output_path ${output_path} --cache_dir  $cache_dir && taskset -c $cpunum python graphgpt/eval/eval_temprompt.py --dataset ${dataset} --variant ${variant} --task ${task} --res_path ${output_path} --num_runs ${num_runs}