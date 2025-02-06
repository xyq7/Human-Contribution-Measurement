#!/bin/bash

data=$1
eval_model=$2
model=$3
time=$4
temperature=$5

if [ ! -d "./result_lens2" ]; then
  mkdir -p ./result_lens2
fi


python src/evaluate.py --dataset_name ${data} --llm_config_file ./src/config/${eval_model}.yaml \
--dataset_file ./data_new/${data}.json \
--output_result_dir ./result_temperature/ \
--response_file ./tempature_output/${model}_${data}_${time}_${temperature}.jsonl \
--output_name ${eval_model}_${model}_${data}_${time}_${temperature}