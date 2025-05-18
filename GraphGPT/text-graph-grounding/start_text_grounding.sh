dataset=$1
llm_name=$2
cpunum=$3
num_runs=3

mkdir logs

taskset -c ${cpunum} python main_train.py --data_name ${dataset} --llm_name ${llm_name} --num_runs ${num_runs} > logs/tg_${dataset}_${llm_name}.log 2>&1 