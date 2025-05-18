dataset=$1
cpunum=$2
task=$3
num_runs=3

mkdir -p logs
if [ "${task}" = "lp" ];then
    for variant in "nondst2" "dst2-v1" "dst2-v2";
    do
        taskset -c ${cpunum} python generate_temprompt.py --dataset_name ${dataset} --variant ${variant} --num_runs ${num_runs} > logs/get_temprompt_${dataset}_${task}_${variant}.log 2>&1

        # empty
        taskset -c ${cpunum} python generate_temprompt.py --dataset_name ${dataset} --variant ${variant} --empty --num_runs ${num_runs} > logs/get_temprompt_${dataset}_${task}_${variant}_empty.log 2>&1
    done
elif [ "${task}" = "nc" ];then
    for variant in "nondst2" "dst2-v1" "dst2-v2";
    do
        taskset -c ${cpunum} python generate_temprompt_node.py --dataset_name ${dataset} --variant ${variant} --num_runs ${num_runs} > logs/get_temprompt_${dataset}_${task}_${variant}.log 2>&1

        # empty
        taskset -c ${cpunum} python generate_temprompt_node.py --dataset_name ${dataset} --variant ${variant} --empty --num_runs ${num_runs} > logs/get_temprompt_${dataset}_${task}_${variant}_empty.log 2>&1
    done
fi