#!/bin/bash

data=$1
eval_model=$2
model=$3
time=$4

if [ ! -d "./result_lens2" ]; then
  mkdir -p ./result_lens2
fi


python src/evaluate_vary_lens.py --dataset_name ${data} --llm_config_file ./src/config/${eval_model}.yaml \
--dataset_file ./data_new/${data}.json \
--output_result_dir ./result_lens/ \
--response_file ./output_lens/${model}_${data}_${time}.jsonl \
--num_words_list 100 200 300 400 500 600 800 1000 \
--output_name ${eval_model}_${model}_${data}_${time}