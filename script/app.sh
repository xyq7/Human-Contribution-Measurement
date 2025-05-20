#!/bin/bash

data=$1
model=$2
time=$3

if [ ! -d "./app_output" ]; then
  mkdir -p ./app_output
fi

python src/generation_app.py --dataset_name ${data} --llm_config_file ./src/config/${model}.yaml \
--input_file ./data_new/${data}.jsonl \
--output_file ./app_output/${model}_${data}_${time}.jsonl --tensor_parallel_size 4