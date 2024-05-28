#!/bin/bash

data=$1
model=$2
time=$3

if [ ! -d "./output_lens" ]; then
  mkdir ./output_lens
fi

python src/generation_vary_lens.py --dataset_name ${data} --llm_config_file ./src/config/${model}.yaml \
--input_file ./data_new/${data}.json \
--output_file ./output_lens/${model}_${data}_${time}.jsonl \
--tensor_parallel_size 1 --batch_size 200 \
--num_words_list 100 200 300 400 500 600 800 1000 --modes gen