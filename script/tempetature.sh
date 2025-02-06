#!/bin/bash

data=$1
model=$2
time=$3
temperature=$4

if [ ! -d "./output" ]; then
  mkdir ./output
fi

python src/generation_llm.py --dataset_name ${data} --llm_config_file ./src/config/${model}.yaml \
--input_file ./data_new/${data}.json \
--output_file ./tempature_output/${model}_${data}_${time}_${temperature}.jsonl \
--resume --batch_size 20 --temperature ${temperature}