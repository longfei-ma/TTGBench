dataset=$1
cpunum=$2
task=${3:-"lp"}
empty=${4:-"empty"}
num_runs=3

if [ "${task}" = "lp" ];then
    # empty
    if [ "${empty}" = "empty" ];then
        taskset -c ${cpunum} python generate_graph_data_raw_link.py --dataset_name ${dataset} --empty --num_runs ${num_runs} > logs/lp_data_{dataset}_empty.log 2>&1
    else
        taskset -c ${cpunum} python generate_graph_data_raw_link.py --dataset_name ${dataset} --num_runs ${num_runs} > logs/lp_data_${dataset}.log 2>&1
    fi

elif [ "${task}" = "nc" ];then
    # empty
    if [ "${empty}" = "empty" ];then
        taskset -c ${cpunum} python generate_graph_data_raw_node.py --dataset_name ${dataset} --empty --num_runs ${num_runs} > logs/nc_data_${dataset}_empty.log 2>&1
    else
        taskset -c ${cpunum} python generate_graph_data_raw_node.py --dataset_name ${dataset} --num_runs ${num_runs} > logs/nc_data_${dataset}.log 2>&1
    fi
fi