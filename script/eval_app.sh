#!/bin/bash

data=$1
eval_model=$2
model=$3
time=$4

if [ ! -d "./result_app" ]; then
  mkdir -p ./result_app
fi


python src/evaluate_app.py --llm_config_file ./src/config/${eval_model}.yaml \
--output_result_dir ./result_app/ \
--response_file ./app_output/${model}_${data}_${time}.jsonl \
--output_name ${eval_model}_${model}_${data}_${time}
