dataset=$1
llm_name=$2
cpunum=$3
empty=${4:-"empty"}
num_runs=3

mkdir logs

#empty
if [ "${empty}" = "empty" ];then
    taskset -c ${cpunum} python main_train.py --data_name ${dataset} --empty --llm_name ${llm_name} --num_runs ${num_runs} > logs/tg_${dataset}_empty.log 2>&1 

#sbert
else
    taskset -c ${cpunum} python main_train.py --data_name ${dataset} --llm_name ${llm_name} --num_runs ${num_runs} > logs/tg_${dataset}_${llm_name}.log 2>&1 
fi