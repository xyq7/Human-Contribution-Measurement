#!/bin/bash

data=$1
eval_model=$2
model=$3
time=$4

if [ ! -d "./result_ada" ]; then
  mkdir -p ./result_ada
fi


python src/evaluate.py --dataset_name ${data} --llm_config_file ./src/config/${eval_model}.yaml \
--dataset_file ./data_new/${data}.json \
--output_result_dir ./result_ada/ \
--response_file ./ada_output/${model}_${data}_${time}.jsonl \
--output_name ${eval_model}_${model}_${data}_${time}
